"""The tool loop (#7) with FakeLLM and a spy tool: no LLM, no network."""

from decimal import Decimal
from typing import Any

from pydantic import BaseModel

from shop_ops_agent.agent.loop import MAX_ITERATIONS, AgentLoop
from shop_ops_agent.agent.tools import ToolRegistry
from shop_ops_agent.agent.tools.definitions import (
    CheckStockAndPriceInput,
    StockAndPrice,
)
from shop_ops_agent.agent.tools.registry import Tool, ToolError
from shop_ops_agent.llm import LLMResponse, Message, ToolCall
from tests.fakes import FakeLLM

QUESTION = [
    Message(role="system", content="You are a shop assistant."),
    Message(role="user", content="Колко струва X1?"),
]


class SpyStockTool:
    """A check_stock_and_price that records its calls and returns a fixed result."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def __call__(self, args: CheckStockAndPriceInput) -> BaseModel:
        self.calls.append(args.sku)
        return StockAndPrice(
            sku=args.sku,
            name="Тестов продукт",
            price_eur=Decimal("199.00"),
            price_bgn=Decimal("389.21"),
            stock_qty=3,
            in_stock=True,
        )


def registry_with(spy: SpyStockTool, **tools: Any) -> ToolRegistry:
    stock = Tool(
        name="check_stock_and_price",
        description="price and stock",
        input_model=CheckStockAndPriceInput,
        run=spy,
    )
    return ToolRegistry([stock, *tools.values()])


async def test_tool_is_executed_result_injected_verbatim_and_answer_returned(
    fake_llm_with_tool_call: FakeLLM,
) -> None:
    spy = SpyStockTool()

    outcome = await AgentLoop(fake_llm_with_tool_call, registry_with(spy), "customer").run(QUESTION)

    # 1. the tool ran with the model's arguments
    assert spy.calls == ["X1"]
    # 2. its result went back to the model as the exact JSON it produced, no paraphrase
    [exchange] = outcome.exchanges
    second_call_messages = fake_llm_with_tool_call.calls[1][0]
    assistant, tool = second_call_messages[-2:]
    assert assistant.role == "assistant"
    assert assistant.tool_calls == (exchange.call,)
    assert tool == Message(role="tool", content=exchange.result_json, tool_call_id="c1")
    assert tool.content == exchange.result.model_dump_json()
    assert '"price_eur":"199.00"' in tool.content
    # 3. the model's final answer is returned
    assert outcome.final is not None
    assert outcome.final.content == "Наличен е, цената е 199.00 EUR."


async def test_max_iterations_guard_stops_a_model_that_keeps_calling_tools() -> None:
    call = ToolCall(id="c1", name="check_stock_and_price", arguments={"sku": "X1"})
    llm = FakeLLM([LLMResponse(content="", tool_calls=(call,))])  # repeats forever
    spy = SpyStockTool()

    outcome = await AgentLoop(llm, registry_with(spy), "customer").run(QUESTION)

    assert outcome.final is None
    assert len(llm.calls) == MAX_ITERATIONS == 5
    # Tools ran after each of the first 4 calls; the 5th call's tools are never executed,
    # because no further LLM call would read their results.
    assert len(spy.calls) == MAX_ITERATIONS - 1


async def test_answer_without_tool_calls_ends_the_loop_at_once() -> None:
    llm = FakeLLM([LLMResponse(content="Здравейте!")])

    outcome = await AgentLoop(llm, registry_with(SpyStockTool()), "customer").run(QUESTION)

    assert outcome.final is not None
    assert outcome.final.content == "Здравейте!"
    assert outcome.exchanges == ()
    assert len(llm.calls) == 1


async def test_parallel_tool_calls_run_in_order_and_all_results_go_back() -> None:
    calls = (
        ToolCall(id="a", name="check_stock_and_price", arguments={"sku": "A1"}),
        ToolCall(id="b", name="check_stock_and_price", arguments={"sku": "B2"}),
    )
    llm = FakeLLM([LLMResponse(content="", tool_calls=calls), LLMResponse(content="Готово.")])
    spy = SpyStockTool()

    await AgentLoop(llm, registry_with(spy), "customer").run(QUESTION)

    assert spy.calls == ["A1", "B2"]
    tool_messages = [m for m in llm.calls[1][0] if m.role == "tool"]
    assert [m.tool_call_id for m in tool_messages] == ["a", "b"]


async def test_tool_errors_go_back_to_the_model_instead_of_raising() -> None:
    bad = ToolCall(id="c1", name="check_stock_and_price", arguments={"sku": ""})
    llm = FakeLLM([LLMResponse(content="", tool_calls=(bad,)), LLMResponse(content="Опитах.")])

    outcome = await AgentLoop(llm, registry_with(SpyStockTool()), "customer").run(QUESTION)

    [exchange] = outcome.exchanges
    assert isinstance(exchange.result, ToolError)
    assert llm.calls[1][0][-1].content == exchange.result_json


async def test_the_callers_messages_are_not_mutated() -> None:
    messages = list(QUESTION)
    llm = FakeLLM(
        [
            LLMResponse(
                content="",
                tool_calls=(
                    ToolCall(id="c1", name="check_stock_and_price", arguments={"sku": "X"}),
                ),
            ),
            LLMResponse(content="Готово."),
        ]
    )

    await AgentLoop(llm, registry_with(SpyStockTool()), "customer").run(messages)

    assert messages == QUESTION  # the turn's tool traffic stays inside the loop


async def test_audience_decides_which_tool_specs_are_offered() -> None:
    llm = FakeLLM()

    class Args(BaseModel):
        pass

    secret = Tool(
        name="secret",
        description="x",
        input_model=Args,
        run=lambda _: Args(),
        audience="operator",
    )
    registry = registry_with(SpyStockTool(), secret=secret)

    await AgentLoop(llm, registry, "customer").run(QUESTION)
    await AgentLoop(llm, registry, "operator").run(QUESTION)

    customer_tools, operator_tools = (call[1] or [] for call in llm.calls)
    assert [t["name"] for t in customer_tools] == ["check_stock_and_price"]
    assert [t["name"] for t in operator_tools] == ["check_stock_and_price", "secret"]
