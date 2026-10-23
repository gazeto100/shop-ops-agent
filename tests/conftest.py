"""Shared fixtures. Unit tests must never hit a real LLM — use FakeLLM."""

from typing import Any

import pytest

from shop_ops_agent.llm import LLMResponse, Message, ToolCall


class FakeLLM:
    """Scripted LLM: returns queued responses in order. Records every call."""

    def __init__(self, responses: list[LLMResponse] | None = None) -> None:
        self._responses = list(responses or [LLMResponse(content="ok")])
        self.calls: list[tuple[list[Message], list[dict[str, Any]] | None]] = []

    async def complete(
        self, messages: list[Message], tools: list[dict[str, Any]] | None = None
    ) -> LLMResponse:
        self.calls.append((messages, tools))
        if len(self._responses) > 1:
            return self._responses.pop(0)
        return self._responses[0]


@pytest.fixture
def fake_llm() -> FakeLLM:
    return FakeLLM()


@pytest.fixture
def fake_llm_with_tool_call() -> FakeLLM:
    return FakeLLM(
        [
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(id="c1", name="check_stock_and_price", arguments={"sku": "X1"})
                ],
            ),
            LLMResponse(content="Наличен е, цената е 199.00 EUR."),
        ]
    )
