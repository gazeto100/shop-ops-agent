"""Shared fixtures. Unit tests must never hit a real LLM — use FakeLLM."""

import pytest

from shop_ops_agent.llm import LLMResponse, ToolCall
from tests.fakes import FakeLLM


@pytest.fixture
def fake_llm() -> FakeLLM:
    return FakeLLM()


@pytest.fixture
def fake_llm_with_tool_call() -> FakeLLM:
    return FakeLLM(
        [
            LLMResponse(
                content="",
                tool_calls=(
                    ToolCall(id="c1", name="check_stock_and_price", arguments={"sku": "X1"}),
                ),
            ),
            LLMResponse(content="Наличен е, цената е 199.00 EUR."),
        ]
    )
