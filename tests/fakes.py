"""Test doubles. Unit tests must never hit a real LLM — use FakeLLM."""

from typing import Any

from shop_ops_agent.llm import LLMResponse, Message


class FakeLLM:
    """Scripted LLM: returns queued responses in order. Records every call."""

    def __init__(self, responses: list[LLMResponse] | None = None) -> None:
        self._responses = list(responses or [LLMResponse(content="ok")])
        self.calls: list[tuple[list[Message], list[dict[str, Any]] | None]] = []

    async def complete(
        self, messages: list[Message], tools: list[dict[str, Any]] | None = None
    ) -> LLMResponse:
        # Snapshot: callers (the agent loop) keep appending to the same list.
        self.calls.append((list(messages), tools))
        if len(self._responses) > 1:
            return self._responses.pop(0)
        return self._responses[0]
