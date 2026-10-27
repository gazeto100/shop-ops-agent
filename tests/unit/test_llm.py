"""Adapters against stub SDK clients that return real SDK response objects (no network)."""

from typing import Any, cast

import anthropic
import openai
import pytest
from anthropic.types import Message as AnthropicMessage
from openai.types.chat import ChatCompletion

from shop_ops_agent.config import Settings
from shop_ops_agent.llm import (
    AnthropicLLM,
    LLMConfigError,
    LLMRefusalError,
    Message,
    OllamaLLM,
    OpenAILLM,
    ToolCall,
    create_llm,
)

TOOLS = [
    {
        "name": "check_stock_and_price",
        "description": "Live price and stock for a sku.",
        "parameters": {"type": "object", "properties": {"sku": {"type": "string"}}},
    }
]
CONVERSATION = [
    Message(role="system", content="You are a shop assistant."),
    Message(role="user", content="Здравей"),
]


class _Recorder:
    def __init__(self, response: object) -> None:
        self.response = response
        self.kwargs: dict[str, Any] = {}

    async def create(self, **kwargs: Any) -> object:
        self.kwargs = kwargs
        return self.response


def anthropic_stub(**message: Any) -> tuple[AnthropicLLM, _Recorder]:
    response = AnthropicMessage.model_validate(
        {
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "model": "claude-haiku-5-5",
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": 12, "output_tokens": 7},
            "content": [{"type": "text", "text": "Здравейте!"}],
        }
        | message
    )
    recorder = _Recorder(response)
    client = type("StubAnthropic", (), {"messages": recorder})()
    return AnthropicLLM("claude-haiku-5-5", cast(anthropic.AsyncAnthropic, client)), recorder


def openai_stub(**message: Any) -> tuple[OpenAILLM, _Recorder]:
    response = ChatCompletion.model_validate(
        {
            "id": "chatcmpl-1",
            "object": "chat.completion",
            "created": 0,
            "model": "gpt-4o-mini",
            "usage": {"prompt_tokens": 9, "completion_tokens": 4, "total_tokens": 13},
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "Hello!", "refusal": None}
                    | message,
                }
            ],
        }
    )
    recorder = _Recorder(response)
    completions = type("StubCompletions", (), {"create": recorder.create})()
    chat = type("StubChat", (), {"completions": completions})()
    client = type("StubOpenAI", (), {"chat": chat})()
    return OpenAILLM("gpt-4o-mini", cast(openai.AsyncOpenAI, client)), recorder


# --- Anthropic -------------------------------------------------------------------------


async def test_anthropic_request_shape() -> None:
    llm, sent = anthropic_stub()

    await llm.complete(CONVERSATION, TOOLS)

    assert sent.kwargs["model"] == "claude-haiku-5-5"
    assert sent.kwargs["system"] == "You are a shop assistant."
    assert sent.kwargs["messages"] == [{"role": "user", "content": "Здравей"}]
    assert sent.kwargs["tools"] == [
        {
            "name": "check_stock_and_price",
            "description": "Live price and stock for a sku.",
            "input_schema": TOOLS[0]["parameters"],
        }
    ]
    assert sent.kwargs["output_config"] == {"effort": "low"}
    # Current Claude models reject sampling parameters.
    assert not {"temperature", "top_p", "top_k"} & sent.kwargs.keys()


async def test_anthropic_omits_tools_and_system_when_absent() -> None:
    llm, sent = anthropic_stub()

    await llm.complete([Message(role="user", content="hi")])

    assert sent.kwargs["tools"] is anthropic.omit
    assert sent.kwargs["system"] is anthropic.omit


async def test_anthropic_reads_blocks_by_type_skipping_thinking() -> None:
    llm, _ = anthropic_stub(
        stop_reason="tool_use",
        content=[
            {"type": "thinking", "thinking": "", "signature": "sig"},
            {"type": "text", "text": "Проверявам."},
            {
                "type": "tool_use",
                "id": "tu_1",
                "name": "check_stock_and_price",
                "input": {"sku": "BOIL-80-A"},
            },
        ],
    )

    response = await llm.complete(CONVERSATION, TOOLS)

    assert response.content == "Проверявам."
    assert response.tool_calls == (
        ToolCall(id="tu_1", name="check_stock_and_price", arguments={"sku": "BOIL-80-A"}),
    )
    assert response.usage == {"input_tokens": 12, "output_tokens": 7}


async def test_anthropic_refusal_raises_before_content_is_read() -> None:
    llm, _ = anthropic_stub(stop_reason="refusal", content=[])

    with pytest.raises(LLMRefusalError):
        await llm.complete(CONVERSATION)


# --- OpenAI / Ollama -------------------------------------------------------------------


async def test_openai_request_shape() -> None:
    llm, sent = openai_stub()

    await llm.complete(CONVERSATION, TOOLS)

    assert sent.kwargs["model"] == "gpt-4o-mini"
    assert sent.kwargs["messages"] == [
        {"role": "system", "content": "You are a shop assistant."},
        {"role": "user", "content": "Здравей"},
    ]
    assert sent.kwargs["tools"] == [
        {
            "type": "function",
            "function": {
                "name": "check_stock_and_price",
                "description": "Live price and stock for a sku.",
                "parameters": TOOLS[0]["parameters"],
            },
        }
    ]


async def test_openai_parses_text_tool_calls_and_usage() -> None:
    llm, _ = openai_stub(
        content=None,
        tool_calls=[
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "check_stock_and_price", "arguments": '{"sku": "KET-17-S"}'},
            }
        ],
    )

    response = await llm.complete(CONVERSATION, TOOLS)

    assert response.content == ""
    assert response.tool_calls == (
        ToolCall(id="call_1", name="check_stock_and_price", arguments={"sku": "KET-17-S"}),
    )
    assert response.usage == {"input_tokens": 9, "output_tokens": 4}


async def test_openai_omits_tools_when_absent() -> None:
    llm, sent = openai_stub()

    await llm.complete(CONVERSATION)

    assert sent.kwargs["tools"] is openai.omit


async def test_openai_refusal_raises() -> None:
    llm, _ = openai_stub(content=None, refusal="I can't help with that.")

    with pytest.raises(LLMRefusalError, match="can't help"):
        await llm.complete(CONVERSATION)


def test_ollama_uses_the_openai_compatible_endpoint() -> None:
    llm = OllamaLLM("llama3.2", "http://localhost:11434/")

    assert str(llm._client.base_url) == "http://localhost:11434/v1/"
    assert llm._model == "llama3.2"


@pytest.mark.parametrize("llm", [anthropic_stub()[0], openai_stub()[0]])
async def test_tool_result_turns_are_rejected_until_the_agent_loop(
    llm: AnthropicLLM | OpenAILLM,
) -> None:
    with pytest.raises(NotImplementedError, match="#7"):
        await llm.complete([*CONVERSATION, Message(role="tool", content="{}", tool_call_id="t")])


# --- Provider selection ----------------------------------------------------------------


@pytest.mark.parametrize(
    ("provider", "adapter", "default_model"),
    [
        ("openai", OpenAILLM, "gpt-4o-mini"),
        ("anthropic", AnthropicLLM, "claude-haiku-5-5"),
        ("ollama", OllamaLLM, "llama3.2"),
    ],
)
def test_provider_selects_adapter_and_its_default_model(
    provider: str, adapter: type, default_model: str
) -> None:
    settings = Settings.model_construct(
        llm_provider=provider, openai_api_key="test-key", anthropic_api_key="test-key"
    )

    llm = create_llm(settings)

    assert type(llm) is adapter
    assert llm._model == default_model  # type: ignore[attr-defined]


def test_explicit_llm_model_wins() -> None:
    settings = Settings.model_construct(
        llm_provider="anthropic", llm_model="claude-opus-5-5", anthropic_api_key="test-key"
    )

    assert create_llm(settings)._model == "claude-opus-5-5"  # type: ignore[attr-defined]


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
def test_missing_api_key_is_a_config_error(provider: str) -> None:
    with pytest.raises(LLMConfigError, match="API_KEY"):
        create_llm(Settings.model_construct(llm_provider=provider))
