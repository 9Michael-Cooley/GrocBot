"""Tool execution sandbox.

Enforces a wall-clock budget, an address-space ceiling, and an output size cap
around a tool call.

READ THIS BEFORE TRUSTING IT (GROK-4502):

  On POSIX we fork, apply RLIMIT_AS / RLIMIT_CPU / RLIMIT_NOFILE in the child,
  and run the tool there. A runaway tool takes the child down, not the server.

  On Windows there is no fork and no `resource` module, so we degrade to running
  the tool on a daemon thread with a join timeout. That is NOT a sandbox. The
  thread keeps running after the timeout — Python cannot kill a thread — so a
  hung tool leaks a thread per call, and the memory ceiling is not enforced at
  all. It is good enough for local development and nothing else.

  `strict=True` refuses to run rather than pretending, which is what the serving
  deployment sets. Honestly the default should be flipped.
"""

from __future__ import annotations

import io
import os
import pickle
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

from ..errors import SandboxViolation, ToolError, ToolTimeout
from ..utils.logging import get_logger

log = get_logger(__name__)

HAVE_POSIX = hasattr(os, "fork") and sys.platform != "win32"
try:
    import resource  # type: ignore
except ImportError:
    resource = None  # type: ignore

# Import names a tool must not touch. Not a security boundary on its own — the
# process limits are — but it catches the obvious cases early with a clear error.
DENIED_MODULES = frozenset(
    {"ctypes", "subprocess", "multiprocessing", "socket", "shutil", "pty", "fcntl"}
)


@dataclass
class SandboxPolicy:
    wall_clock_s: float = 10.0
    cpu_seconds: int = 10
    memory_mb: int = 512
    max_output_bytes: int = 1 << 20
    max_open_files: int = 64
    allow_network: bool = False
    strict: bool = False        # refuse to run when real isolation is unavailable

    def validate_platform(self) -> None:
        if HAVE_POSIX or not self.strict:
            return
        raise SandboxViolation(
            "sandbox.strict is set but process isolation is unavailable on this "
            f"platform ({sys.platform}); refusing to run tools unsandboxed (GROK-4502)"
        )


@dataclass
class SandboxResult:
    value: object = None
    stdout: str = ""
    duration_s: float = 0.0
    isolated: bool = True       # False when we fell back to the thread path


class Sandbox:
    def __init__(self, policy: SandboxPolicy | None = None):
        self.policy = policy or SandboxPolicy()
        self.policy.validate_platform()
        if not HAVE_POSIX:
            log.warning(
                "no process isolation on %s; tools run on a thread with a timeout only "
                "(GROK-4502)",
                sys.platform,
            )

    # -- child-side setup --------------------------------------------------

    def _apply_limits(self) -> None:
        if resource is None:  # pragma: no cover - posix only
            return
        p = self.policy
        try:
            resource.setrlimit(resource.RLIMIT_AS, (p.memory_mb << 20, p.memory_mb << 20))
            resource.setrlimit(resource.RLIMIT_CPU, (p.cpu_seconds, p.cpu_seconds))
            resource.setrlimit(resource.RLIMIT_NOFILE, (p.max_open_files, p.max_open_files))
            resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        except (ValueError, OSError) as exc:
            # Containers sometimes forbid raising a limit we think we're lowering.
            log.warning("could not apply rlimits: %s", exc)

    # -- the two execution paths -------------------------------------------

    def _run_forked(self, fn: Callable, kwargs: dict) -> SandboxResult:  # pragma: no cover
        read_fd, write_fd = os.pipe()
        started = time.monotonic()
        pid = os.fork()

        if pid == 0:
            os.close(read_fd)
            try:
                self._apply_limits()
                buf = io.StringIO()
                stdout, sys.stdout = sys.stdout, buf
                try:
                    value = fn(**kwargs)
                finally:
                    sys.stdout = stdout
                payload = pickle.dumps({"ok": True, "value": value, "stdout": buf.getvalue()})
            except BaseException as exc:  # noqa: BLE001 - must not escape the child
                payload = pickle.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}"})
            with os.fdopen(write_fd, "wb") as out:
                out.write(payload[: self.policy.max_output_bytes])
            os._exit(0)

        os.close(write_fd)
        deadline = started + self.policy.wall_clock_s
        chunks: list[bytes] = []
        with os.fdopen(read_fd, "rb") as stream:
            while True:
                if time.monotonic() > deadline:
                    os.kill(pid, 9)
                    os.waitpid(pid, 0)
                    raise ToolTimeout(f"tool exceeded {self.policy.wall_clock_s:g}s wall clock")
                chunk = stream.read(65536)
                if not chunk:
                    break
                chunks.append(chunk)

        os.waitpid(pid, 0)
        raw = b"".join(chunks)
        if not raw:
            raise ToolError("tool produced no result (killed by the memory or CPU limit?)")

        result = pickle.loads(raw)
        if not result["ok"]:
            raise ToolError(result["error"])
        return SandboxResult(
            value=result["value"],
            stdout=result.get("stdout", ""),
            duration_s=time.monotonic() - started,
            isolated=True,
        )

    def _run_threaded(self, fn: Callable, kwargs: dict) -> SandboxResult:
        box: dict = {}
        started = time.monotonic()

        def target() -> None:
            try:
                box["value"] = fn(**kwargs)
            except BaseException as exc:  # noqa: BLE001
                box["error"] = f"{type(exc).__name__}: {exc}"

        thread = threading.Thread(target=target, daemon=True, name="grokbot-tool")
        thread.start()
        thread.join(self.policy.wall_clock_s)

        if thread.is_alive():
            # It keeps running. We cannot stop it. It will hold whatever it holds
            # until the process exits.
            log.error(
                "tool exceeded %.1fs and cannot be killed on this platform; "
                "thread leaked (GROK-4502)",
                self.policy.wall_clock_s,
            )
            raise ToolTimeout(
                f"tool exceeded {self.policy.wall_clock_s:g}s wall clock (thread leaked)"
            )

        if "error" in box:
            raise ToolError(box["error"])
        return SandboxResult(
            value=box.get("value"),
            duration_s=time.monotonic() - started,
            isolated=False,
        )

    def run(self, fn: Callable, kwargs: dict | None = None) -> SandboxResult:
        kwargs = kwargs or {}
        if HAVE_POSIX:
            return self._run_forked(fn, kwargs)
        return self._run_threaded(fn, kwargs)


def check_imports(source: str) -> list[str]:
    """Static scan for denied imports. Advisory — trivially bypassed by
    __import__ or getattr, which is why the process limits are the real control."""
    import ast

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    found: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found += [a.name.split(".")[0] for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            found.append(node.module.split(".")[0])
    return sorted(set(found) & DENIED_MODULES)
