"""Built-in tools.

Deliberately boring. These exist so the agent loop has something real to call in
tests and demos; anything domain-specific belongs in the caller's registry, not
here.

`http_get` is a stub. The real one goes through the egress proxy, which does DNS
pinning and blocks link-local and RFC1918 ranges — without it, a tool call is an
SSRF primitive pointed at the cluster's metadata endpoint. It is not included in
this tree, so the tool refuses rather than making a direct request.
"""

from __future__ import annotations

import ast
import math
import operator
import time
from datetime import datetime, timezone

from ..errors import ToolError
from .registry import tool

# --------------------------------------------------------------------------
# calculator
# --------------------------------------------------------------------------

_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARYOPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}

_FUNCTIONS = {
    "sqrt": math.sqrt,
    "abs": abs,
    "round": round,
    "floor": math.floor,
    "ceil": math.ceil,
    "log": math.log,
    "log2": math.log2,
    "log10": math.log10,
    "exp": math.exp,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "min": min,
    "max": max,
}
_CONSTANTS = {"pi": math.pi, "e": math.e, "tau": math.tau}

# 2**(2**30) is a one-line way to hang the process. Bound the exponent.
_MAX_EXPONENT = 1024


def _eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)

    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ToolError(f"unsupported literal {node.value!r}")
        return node.value

    if isinstance(node, ast.Name):
        if node.id not in _CONSTANTS:
            raise ToolError(f"unknown name {node.id!r}")
        return _CONSTANTS[node.id]

    if isinstance(node, ast.BinOp):
        op = _BINOPS.get(type(node.op))
        if op is None:
            raise ToolError(f"unsupported operator {type(node.op).__name__}")
        left, right = _eval_node(node.left), _eval_node(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > _MAX_EXPONENT:
            raise ToolError(f"exponent {right} exceeds the limit of {_MAX_EXPONENT}")
        try:
            return op(left, right)
        except ZeroDivisionError:
            raise ToolError("division by zero") from None

    if isinstance(node, ast.UnaryOp):
        op = _UNARYOPS.get(type(node.op))
        if op is None:
            raise ToolError(f"unsupported unary operator {type(node.op).__name__}")
        return op(_eval_node(node.operand))

    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in _FUNCTIONS:
            raise ToolError(
                f"unknown function; available: {', '.join(sorted(_FUNCTIONS))}"
            )
        if node.keywords:
            raise ToolError("keyword arguments are not supported")
        return _FUNCTIONS[node.func.id](*[_eval_node(a) for a in node.args])

    raise ToolError(f"unsupported expression node {type(node).__name__}")


@tool(description="Evaluate an arithmetic expression. Supports + - * / // % **, "
                  "sqrt, log, exp, sin, cos, tan, floor, ceil, round, min, max, "
                  "and the constants pi, e, tau.")
def calculator(expression: str) -> float:
    """Evaluate arithmetic safely.

    Walks a parsed AST with an explicit node allowlist. Never eval() — a
    calculator tool is the single most obvious place a prompt-injected
    expression lands, and eval() there is arbitrary code execution wearing a
    hat.
    """
    expression = expression.strip()
    if not expression:
        raise ToolError("empty expression")
    if len(expression) > 500:
        raise ToolError(f"expression too long ({len(expression)} chars, limit 500)")

    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ToolError(f"could not parse {expression!r}: {exc.msg}") from exc

    result = _eval_node(tree)
    if isinstance(result, complex):
        raise ToolError("complex results are not supported")
    if math.isnan(result) or math.isinf(result):
        raise ToolError(f"result is not finite ({result})")
    return float(result)


# --------------------------------------------------------------------------
# clock
# --------------------------------------------------------------------------


@tool(description="Get the current UTC date and time in ISO 8601 format.")
def clock(format: str = "iso") -> str:
    """Current UTC time.

    Models have no clock and will confidently invent one, so this is worth
    wiring up in almost every deployment.
    """
    now = datetime.now(timezone.utc)
    if format == "iso":
        return now.isoformat()
    if format == "date":
        return now.strftime("%Y-%m-%d")
    if format == "unix":
        return str(int(time.time()))
    if format == "human":
        return now.strftime("%A, %d %B %Y at %H:%M UTC")
    raise ToolError(f"unknown format {format!r}; expected iso, date, unix, or human")


# --------------------------------------------------------------------------
# http_get
# --------------------------------------------------------------------------


@tool(
    description="Fetch the contents of a URL over HTTP GET. Returns the response body as text.",
    timeout_s=15.0,
    dangerous=True,
)
def http_get(url: str, max_bytes: int = 65536) -> str:
    """Fetch a URL through the egress proxy.

    Not functional in this tree. The proxy client is what enforces DNS pinning
    and blocks link-local / RFC1918 destinations; without it a direct request
    from inside the serving pod is an SSRF primitive aimed at the metadata
    endpoint. Failing closed is the correct behaviour here.
    """
    if not url.lower().startswith(("http://", "https://")):
        raise ToolError(f"url must be http(s), got {url!r}")
    raise ToolError(
        "http_get is unavailable: the egress proxy client is not part of this tree. "
        "Register your own fetch tool if you need one, and make sure it validates "
        "the resolved address, not just the hostname."
    )


BUILTINS = ["calculator", "clock", "http_get"]
