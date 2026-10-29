"""The tool loop (#7): LLM -> tool calls -> execute -> results back -> LLM, bounded.

Within one turn the conversation is append-only, so Claude's thinking blocks replayed
with the tool results stay valid. Tool results go back as the registry's JSON, verbatim:
the model never sees a paraphrase.
"""

import asyncio
import logging
from dataclasses import dataclass

from pydantic import BaseModel

from shop_ops_agent.agent.tools import Audience, ToolRegistry
from shop_ops_agent.llm import LLM, LLMResponse, Message, ToolCall

log = logging.getLogger(__name__)

# An iteration is one LLM call. Five is enough for search -> check -> answer with room
# for a retry; beyond that the model is usually looping.
MAX_ITERATIONS = 5


@dataclass(frozen=True)
class ToolExchange:
    call: ToolCall
    result: BaseModel
    result_json: str  # exactly what the model received


@dataclass(frozen=True)
class LoopResult:
    final: LLMResponse | None  # None: the iteration budget ran out while still calling tools
    exchanges: tuple[ToolExchange, ...]


class AgentLoop:
    def __init__(
        self,
        llm: LLM,
        registry: ToolRegistry,
        audience: Audience,
        max_iterations: int = MAX_ITERATIONS,
    ) -> None:
        if max_iterations < 1:
            raise ValueError("max_iterations must be at least 1")
        self._llm = llm
        self._registry = registry
        self._audience = audience
        self._max_iterations = max_iterations

    async def run(self, messages: list[Message]) -> LoopResult:
        conversation = list(messages)  # this turn's working copy; only ever appended to
        tools = self._registry.specs(self._audience) or None
        exchanges: list[ToolExchange] = []
        # Every call but the last may run tools and feed the results back...
        for _ in range(self._max_iterations - 1):
            response = await self._llm.complete(conversation, tools)
            if not response.tool_calls:
                return LoopResult(final=response, exchanges=tuple(exchanges))
            conversation.append(response.as_message())
            for call in response.tool_calls:
                exchange = await self._execute(call)
                exchanges.append(exchange)
                conversation.append(
                    Message(role="tool", content=exchange.result_json, tool_call_id=call.id)
                )
        # ...the last one must answer: tools it asks for are never run, since no further
        # LLM call would read their results.
        response = await self._llm.complete(conversation, tools)
        if not response.tool_calls:
            return LoopResult(final=response, exchanges=tuple(exchanges))
        log.warning("agent loop stopped after %d iterations", self._max_iterations)
        return LoopResult(final=None, exchanges=tuple(exchanges))

    async def _execute(self, call: ToolCall) -> ToolExchange:
        # Tools are synchronous (CSV / embedding work); keep them off the event loop.
        # registry.call never raises: failures come back as a ToolError result.
        result = await asyncio.to_thread(
            self._registry.call, call.name, call.arguments, self._audience
        )
        return ToolExchange(call=call, result=result, result_json=result.model_dump_json())
