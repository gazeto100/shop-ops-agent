"""Tool registry: name -> callable, plus the `ToolSpec` list the provider layer sends.

`call()` never raises. An unknown tool, a tool outside the caller's audience, arguments
that fail validation, or a crash inside a tool all come back as a `ToolError`, which the
agent loop (#7) returns to the model like any other result so it can correct itself.
"""

import logging
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, ValidationError

from shop_ops_agent.llm import ToolSpec

log = logging.getLogger(__name__)

# customer: the public /chat. operator: shop staff, who may also use every customer tool.
type Audience = Literal["customer", "operator"]


class ToolError(BaseModel):
    status: Literal["error"] = "error"
    tool: str
    message: str


@dataclass(frozen=True)
class Tool[In: BaseModel]:
    name: str
    description: str
    input_model: type[In]
    run: Callable[[In], BaseModel]
    audience: Audience = "customer"

    def spec(self) -> ToolSpec:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.input_model.model_json_schema(),
        }


class ToolRegistry:
    def __init__(self, tools: Iterable[Tool[Any]]) -> None:
        self._tools = {tool.name: tool for tool in tools}

    def specs(self, audience: Audience) -> list[ToolSpec]:
        return [t.spec() for t in self._tools.values() if _allowed(t, audience)]

    def call(self, name: str, arguments: dict[str, Any], audience: Audience) -> BaseModel:
        tool = self._tools.get(name)
        if tool is None or not _allowed(tool, audience):
            return ToolError(tool=name, message=f"Unknown tool {name!r}.")
        try:
            args = tool.input_model.model_validate(arguments)
        except ValidationError as e:
            problems = "; ".join(
                f"{'.'.join(map(str, err['loc'])) or 'arguments'}: {err['msg']}"
                for err in e.errors()
            )
            return ToolError(tool=name, message=f"Invalid arguments: {problems}")
        try:
            return tool.run(args)
        except Exception:
            log.exception("tool %s failed", name)
            return ToolError(tool=name, message="The tool failed; try again later.")


def _allowed(tool: Tool[Any], audience: Audience) -> bool:
    return audience == "operator" or tool.audience == "customer"
