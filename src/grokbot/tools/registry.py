"""Tool registry.

A tool is a plain function plus a JSON schema derived from its signature. The
schema is generated rather than hand-written because hand-written schemas drift
from the implementation and the model then calls a signature that doesn't exist.

    @tool(description="Look up a ticker's price.")
    def stock_price(symbol: str, currency: str = "USD") -> float:
        ...

Type hints map to JSON Schema. Anything unmappable raises at decoration time —
at import, not at the moment the model tries to call it.
"""

from __future__ import annotations

import inspect
import json
import typing
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal, Union, get_args, get_origin

from ..errors import ToolError
from ..utils.logging import get_logger

log = get_logger(__name__)

try:                                  # 3.10+
    from types import UnionType as _UNION_TYPE
except ImportError:                   # pragma: no cover
    _UNION_TYPE = None  # type: ignore[assignment]

_PRIMITIVES: dict[Any, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
}


def _schema_for(annotation: Any, name: str) -> dict:
    if annotation is inspect.Parameter.empty:
        raise ToolError(f"parameter {name!r} has no type annotation; tools must be fully annotated")

    if annotation in _PRIMITIVES:
        return {"type": _PRIMITIVES[annotation]}

    origin = get_origin(annotation)
    args = get_args(annotation)

    if origin is Literal:
        if not args:
            raise ToolError(f"{name}: empty Literal")
        kinds = {type(a) for a in args}
        if len(kinds) > 1:
            raise ToolError(f"{name}: Literal mixes types {kinds}")
        return {"type": _PRIMITIVES.get(kinds.pop(), "string"), "enum": list(args)}

    if origin in (list, typing.List):  # noqa: UP006
        item = args[0] if args else str
        return {"type": "array", "items": _schema_for(item, f"{name}[]")}

    if origin in (dict, typing.Dict):  # noqa: UP006
        return {"type": "object", "additionalProperties": True}

    # `int | None` is types.UnionType, `Optional[int]` is typing.Union. They are
    # not the same object and get_origin reports each as itself, so both have to
    # be checked — annotating a tool with modern union syntax raised otherwise.
    if origin is Union or origin is _UNION_TYPE:
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) == 1:
            # Optional[X] — nullability is expressed by absence from `required`.
            return _schema_for(non_none[0], name)
        return {"anyOf": [_schema_for(a, name) for a in non_none]}

    raise ToolError(f"{name}: cannot derive a JSON schema for {annotation!r}")


@dataclass
class Tool:
    name: str
    description: str
    fn: Callable
    parameters: dict = field(default_factory=dict)
    required: list[str] = field(default_factory=list)
    timeout_s: float = 10.0
    dangerous: bool = False        # requires explicit opt-in to be callable

    def schema(self) -> dict:
        """OpenAI function-calling shape. The model sees exactly this."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": self.parameters,
                    "required": self.required,
                },
            },
        }

    def validate_args(self, args: dict) -> dict:
        missing = [r for r in self.required if r not in args]
        if missing:
            raise ToolError(f"{self.name}: missing required argument(s) {missing}")

        unknown = [k for k in args if k not in self.parameters]
        if unknown:
            # Models hallucinate parameters. Dropping is friendlier than failing
            # the whole turn, but it gets logged because a persistent one usually
            # means the description is misleading.
            log.debug("%s: dropping unknown argument(s) %s", self.name, unknown)
            args = {k: v for k, v in args.items() if k in self.parameters}

        coerced = {}
        for key, value in args.items():
            want = self.parameters[key].get("type")
            try:
                if want == "integer" and not isinstance(value, bool):
                    coerced[key] = int(value)
                elif want == "number":
                    coerced[key] = float(value)
                elif want == "boolean":
                    coerced[key] = value if isinstance(value, bool) else str(value).lower() == "true"
                elif want == "string":
                    coerced[key] = value if isinstance(value, str) else json.dumps(value)
                else:
                    coerced[key] = value
            except (TypeError, ValueError) as exc:
                raise ToolError(f"{self.name}: argument {key!r}={value!r} is not a {want}") from exc

            enum = self.parameters[key].get("enum")
            if enum and coerced[key] not in enum:
                raise ToolError(f"{self.name}: {key!r} must be one of {enum}, got {coerced[key]!r}")
        return coerced

    def __call__(self, **kwargs):
        return self.fn(**self.validate_args(kwargs))


class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool, *, replace: bool = False) -> Tool:
        if tool.name in self._tools and not replace:
            # Used to silently last-one-wins. That shipped a registry where a
            # test double shadowed the real tool in production for two days.
            raise ToolError(
                f"tool {tool.name!r} is already registered; pass replace=True to override"
            )
        self._tools[tool.name] = tool
        return tool

    def get(self, name: str) -> Tool:
        try:
            return self._tools[name]
        except KeyError:
            raise ToolError(
                f"unknown tool {name!r}; registered: {sorted(self._tools) or '(none)'}"
            ) from None

    def unregister(self, name: str) -> None:
        self._tools.pop(name, None)

    def subset(self, names: list[str]) -> ToolRegistry:
        sub = ToolRegistry()
        for n in names:
            sub.register(self.get(n))
        return sub

    def schemas(self, *, include_dangerous: bool = False) -> list[dict]:
        return [
            t.schema()
            for t in self._tools.values()
            if include_dangerous or not t.dangerous
        ]

    def names(self) -> list[str]:
        return sorted(self._tools)

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __iter__(self):
        return iter(self._tools.values())


REGISTRY = ToolRegistry()


def tool(
    _fn: Callable | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
    timeout_s: float = 10.0,
    dangerous: bool = False,
    registry: ToolRegistry | None = None,
):
    """Decorator. Registers into REGISTRY unless another registry is passed."""

    def wrap(fn: Callable) -> Tool:
        sig = inspect.signature(fn)
        hints = typing.get_type_hints(fn)

        properties: dict[str, dict] = {}
        required: list[str] = []
        for pname, param in sig.parameters.items():
            if pname == "self":
                continue
            if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
                raise ToolError(f"{fn.__name__}: *args/**kwargs are not supported in tools")
            schema = _schema_for(hints.get(pname, param.annotation), pname)
            if param.default is not inspect.Parameter.empty:
                schema["default"] = param.default
            else:
                required.append(pname)
            properties[pname] = schema

        doc = description or (inspect.getdoc(fn) or "").split("\n\n")[0].strip()
        if not doc:
            raise ToolError(
                f"{fn.__name__}: tools need a description (docstring or description=). "
                f"The model only sees this text."
            )

        built = Tool(
            name=name or fn.__name__,
            description=doc,
            fn=fn,
            parameters=properties,
            required=required,
            timeout_s=timeout_s,
            dangerous=dangerous,
        )
        # `registry or REGISTRY` is wrong here: ToolRegistry defines __len__, so
        # an *empty* registry is falsy and every tool decorated with a fresh
        # registry= silently landed in the global one instead.
        target = registry if registry is not None else REGISTRY
        target.register(built, replace=True)
        return built

    return wrap(_fn) if _fn is not None else wrap
