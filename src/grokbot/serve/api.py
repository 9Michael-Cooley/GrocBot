"""HTTP server.

Built on http.server. The production deployment ran this behind an ASGI adapter
with uvloop; the adapter is in //platform/api and is not here. The stdlib path is
what remains, and it is genuinely serviceable for local work and integration
tests — it is not what you want in front of real traffic.

The engine is single-threaded and every request holds the engine lock for its
duration. Concurrency is therefore 1 in this build, which makes the scheduler's
continuous batching invisible from outside. That is a property of this server,
not of the engine; the real adapter drives the same step loop from one thread
and multiplexes streams off it.
"""

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from ..config import Config
from ..errors import GrokBotError, RateLimited, SafetyBlocked
from ..inference.engine import Engine
from ..safety.policy import PolicyConfig, PolicyEngine
from ..telemetry import metrics
from ..utils.logging import get_logger, set_request_id
from . import protocol
from .protocol import SSE_DONE, ChatRequest, ProtocolError, sse
from .ratelimit import Guard, LimitConfig, RateLimiter

log = get_logger(__name__)


class ServerState:
    """Everything the handler needs. Handlers are instantiated per request, so
    shared state cannot live on them."""

    def __init__(self, cfg: Config):
        self.config = cfg
        self.engine = Engine(cfg)
        self.limiter = RateLimiter(LimitConfig.from_dict(cfg.get("limits")))
        self.policy = PolicyEngine(PolicyConfig.from_dict(cfg.get("safety")))
        self.engine_lock = threading.Lock()
        self.started = time.time()

        self.served_names = cfg.get("api.served_model_names") or [self.engine.model_config.name]
        keys = cfg.get("api.api_keys") or []
        self.api_keys = set(keys)
        if not self.api_keys:
            log.warning("api.api_keys is empty — this server accepts unauthenticated requests")


def _status_for(exc: Exception) -> int:
    return getattr(exc, "status_code", 500) if isinstance(exc, GrokBotError) else 500


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "grokbot/0.4.2"
    state: ServerState = None  # type: ignore[assignment]

    # -- plumbing ----------------------------------------------------------

    def log_message(self, fmt: str, *args) -> None:
        log.debug("%s - %s", self.address_string(), fmt % args)

    def _auth_key(self) -> str:
        header = self.headers.get("authorization", "")
        token = header[7:].strip() if header.lower().startswith("bearer ") else ""
        if self.state.api_keys and token not in self.state.api_keys:
            raise RateLimited("invalid or missing API key", retry_after=0.0)
        return token or self.client_address[0]

    def _send_json(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self._cors()
        self.end_headers()
        self.wfile.write(body)

    def _send_text(self, text: str, status: int = 200, ctype: str = "text/plain") -> None:
        body = text.encode()
        self.send_response(status)
        self.send_header("content-type", ctype)
        self.send_header("content-length", str(len(body)))
        self._cors()
        self.end_headers()
        self.wfile.write(body)

    def _cors(self) -> None:
        origins = self.state.config.get("server.cors_origins") or []
        if origins:
            self.send_header("access-control-allow-origin", origins[0] if len(origins) == 1 else "*")
            self.send_header("access-control-allow-headers", "authorization, content-type")

    def _read_body(self) -> dict:
        limit = self.state.config.get("server.max_body_bytes", 8 << 20)
        length = int(self.headers.get("content-length") or 0)
        if length <= 0:
            raise ProtocolError("empty request body")
        if length > limit:
            raise ProtocolError(f"request body is {length} bytes, limit is {limit}")
        try:
            return json.loads(self.rfile.read(length))
        except json.JSONDecodeError as exc:
            raise ProtocolError(f"invalid JSON: {exc}") from exc

    # -- routes ------------------------------------------------------------

    def do_OPTIONS(self) -> None:  # noqa: N802
        self.send_response(204)
        self._cors()
        self.send_header("content-length", "0")
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        path = self.path.split("?")[0]
        if path in ("/health", "/healthz"):
            self._send_json({"status": "ok", "uptime_s": round(time.time() - self.state.started, 1)})
        elif path == "/v1/models":
            self._send_json(
                {
                    "object": "list",
                    "data": [
                        {"id": n, "object": "model", "owned_by": "grokbot"}
                        for n in self.state.served_names
                    ],
                }
            )
        elif path == self.state.config.get("telemetry.metrics_path", "/metrics"):
            metrics.scrape_engine(self.state.engine)
            self._send_text(metrics.REGISTRY.render(), ctype="text/plain; version=0.0.4")
        elif path == "/stats":
            self._send_json(
                {
                    "engine": self.state.engine.stats(),
                    "limiter": self.state.limiter.snapshot(),
                    "policy": self.state.policy.report(),
                    "ignored_request_fields": protocol.ignored_field_report(),
                }
            )
        else:
            self._send_json({"error": {"message": f"no route for GET {path}"}}, 404)

    def do_POST(self) -> None:  # noqa: N802
        path = self.path.split("?")[0]
        request_id = f"req-{int(time.time() * 1000) % 10**9:09d}"
        set_request_id(request_id)

        try:
            if path not in ("/v1/chat/completions", "/v1/completions"):
                self._send_json({"error": {"message": f"no route for POST {path}"}}, 404)
                return

            key = self._auth_key()
            payload = self._read_body()
            request = ChatRequest.parse(payload)
            if path == "/v1/completions":
                request.messages = [{"role": "user", "content": payload.get("prompt", "")}]

            metrics.requests_total.inc()
            estimate = sum(len(m.get("content", "")) for m in request.messages) // 4
            self.state.limiter.check(key, estimate)
            self.state.limiter.validate_size(estimate, request.gen.max_tokens)

            prompt_text = "\n".join(m.get("content", "") for m in request.messages)
            decision = self.state.policy.check_input(prompt_text)
            if not decision.allowed:
                raise SafetyBlocked(decision.reason, stage="input")

            with Guard(self.state.limiter, key):
                if request.stream:
                    self._stream(request, request_id)
                else:
                    self._complete(request, request_id)

        except Exception as exc:  # noqa: BLE001 - boundary: nothing escapes to the socket
            metrics.requests_failed.inc()
            status = _status_for(exc)
            if status >= 500:
                log.exception("unhandled error serving %s", path)
            else:
                log.info("%s -> %d: %s", path, status, exc)
            try:
                self._send_json(protocol.error_response(exc, status), status)
            except (BrokenPipeError, ConnectionResetError):
                pass   # client already gone; nothing to report to

    # -- handlers ----------------------------------------------------------

    def _complete(self, request: ChatRequest, request_id: str) -> None:
        started = time.monotonic()
        with self.state.engine_lock:
            completion = self.state.engine.chat(request.messages, request.gen)

        checked = self.state.policy.check_output(completion.text)
        if not checked.allowed:
            raise SafetyBlocked(checked.reason, stage="output")
        completion.text = checked.text

        metrics.latency.observe(time.monotonic() - started)
        metrics.tokens_generated.inc(completion.completion_tokens)
        metrics.tokens_prefilled.inc(completion.prompt_tokens)
        metrics.prompt_tokens.observe(completion.prompt_tokens)

        self._send_json(protocol.chat_response(completion, request.model, request_id))

    def _stream(self, request: ChatRequest, request_id: str) -> None:
        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.send_header("cache-control", "no-cache")
        self.send_header("connection", "keep-alive")
        self.send_header("x-accel-buffering", "no")   # nginx buffers SSE otherwise
        self._cors()
        self.end_headers()

        started = time.monotonic()
        first = True
        emitted = 0
        finish = "stop"

        try:
            self.wfile.write(sse(protocol.chunk_response("", request.model, request_id, role=True)))
            self.wfile.flush()

            with self.state.engine_lock:
                for chunk in self.state.engine.stream_chat(request.messages, request.gen):
                    if first and chunk.text:
                        metrics.ttft.observe(time.monotonic() - started)
                        first = False
                    if chunk.text:
                        emitted += 1
                        self.wfile.write(
                            sse(protocol.chunk_response(chunk.text, request.model, request_id))
                        )
                        # No backpressure handling here: a slow reader blocks this
                        # write and pins the engine lock for everyone. Known.
                        self.wfile.flush()
                    if chunk.is_final:
                        finish = protocol._FINISH_MAP.get(chunk.finish_reason, "stop")

            self.wfile.write(
                sse(protocol.chunk_response("", request.model, request_id, finish=finish))
            )
            self.wfile.write(SSE_DONE)
            self.wfile.flush()

        except (BrokenPipeError, ConnectionResetError):
            log.info("client disconnected after %d chunks", emitted)
        finally:
            metrics.tokens_generated.inc(emitted)
            metrics.latency.observe(time.monotonic() - started)


def serve(cfg: Config, host: str | None = None, port: int | None = None) -> None:
    state = ServerState(cfg)
    Handler.state = state

    host = host or cfg.get("server.host", "127.0.0.1")
    port = int(port or cfg.get("server.port", 8080))

    httpd = ThreadingHTTPServer((host, port), Handler)
    httpd.daemon_threads = True

    log.info(
        "serving %s on http://%s:%d (models: %s)",
        state.engine.model_config.name,
        host,
        port,
        ", ".join(state.served_names),
    )
    if state.engine.backend.is_synthetic:
        log.warning("backend is synthetic — responses are not model output")

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        log.info("shutting down")
    finally:
        httpd.server_close()
