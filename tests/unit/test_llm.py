"""Adapters against stub SDK clients that return real SDK response objects (no network)."""

import json
from typing import Any, cast

import anthropic
import httpx2
import openai
import pytest
from anthropic.types import Message as AnthropicMessage
from openai.types.chat import ChatCompletion

from shop_ops_agent.config import Settings
from shop_ops_agent.llm import (
    AnthropicLLM,
    LLMConfigError,
    LLMRefusalError,
    LLMUnavailableError,
    Message,
    OllamaLLM,
    OpenAILLM,
    ToolCall,
    ToolSpec,
    create_llm,
)

TOOLS: list[ToolSpec] = [
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
        if isinstance(self.response, Exception):
            raise self.response
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
    # Never sent: effort 400s on claude-haiku-5-5 (live API); sampling params 400 on
    # current Claude models.
    assert not {"output_config", "temperature", "top_p", "top_k"} & sent.kwargs.keys()


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


async def test_anthropic_max_tokens_cut_off_never_yields_a_tool_call() -> None:
    # A response cut off by max_tokens may carry a truncated tool_use input.
    llm, _ = anthropic_stub(
        stop_reason="max_tokens",
        content=[
            {"type": "text", "text": "Проверявам"},
            {"type": "tool_use", "id": "tu_1", "name": "check_stock_and_price", "input": {}},
        ],
    )

    response = await llm.complete(CONVERSATION, TOOLS)

    assert response.tool_calls == ()
    assert response.content == "Проверявам"


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


# --- Tool round trips (#7) --------------------------------------------------------------

CALL_A = ToolCall(id="tu_a", name="check_stock_and_price", arguments={"sku": "BOIL-80-A"})
CALL_B = ToolCall(id="tu_b", name="check_stock_and_price", arguments={"sku": "KET-17-S"})
RESULT_A = '{"status":"found","sku":"BOIL-80-A","price_eur":"189.00"}'
RESULT_B = '{"status":"not_found","sku":"KET-17-S"}'


def round_trip(assistant: Message) -> list[Message]:
    return [
        *CONVERSATION,
        assistant,
        Message(role="tool", content=RESULT_A, tool_call_id="tu_a"),
        Message(role="tool", content=RESULT_B, tool_call_id="tu_b"),
    ]


async def test_anthropic_replays_its_own_blocks_and_groups_tool_results() -> None:
    first, _ = anthropic_stub(
        stop_reason="tool_use",
        content=[
            {"type": "thinking", "thinking": "", "signature": "SIG"},
            {"type": "tool_use", "id": "tu_a", "name": CALL_A.name, "input": CALL_A.arguments},
            {"type": "tool_use", "id": "tu_b", "name": CALL_B.name, "input": CALL_B.arguments},
        ],
    )
    response = await first.complete(CONVERSATION, TOOLS)
    llm, sent = anthropic_stub()

    await llm.complete(round_trip(response.as_message()), TOOLS)

    _, assistant, results = sent.kwargs["messages"]
    # Claude's own blocks, thinking (signature) included, are sent back unchanged.
    assert assistant == {"role": "assistant", "content": list(response.provider_content)}
    # Both results in ONE user message, verbatim.
    assert results == {
        "role": "user",
        "content": [
            {"type": "tool_result", "tool_use_id": "tu_a", "content": RESULT_A},
            {"type": "tool_result", "tool_use_id": "tu_b", "content": RESULT_B},
        ],
    }


async def test_anthropic_builds_tool_use_blocks_for_a_foreign_assistant_turn() -> None:
    llm, sent = anthropic_stub()
    foreign = Message(role="assistant", content="Проверявам.", tool_calls=(CALL_A, CALL_B))

    await llm.complete(round_trip(foreign), TOOLS)

    assert sent.kwargs["messages"][1]["content"] == [
        {"type": "text", "text": "Проверявам."},
        {"type": "tool_use", "id": "tu_a", "name": CALL_A.name, "input": CALL_A.arguments},
        {"type": "tool_use", "id": "tu_b", "name": CALL_B.name, "input": CALL_B.arguments},
    ]


async def test_anthropic_sdk_sends_the_thinking_signature_back() -> None:
    """End to end through the real SDK (mock HTTP transport, no network)."""
    bodies: list[dict[str, Any]] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        bodies.append(json.loads(request.content))
        content = (
            [
                {"type": "thinking", "thinking": "", "signature": "SIG-42"},
                {"type": "tool_use", "id": "tu_a", "name": CALL_A.name, "input": CALL_A.arguments},
            ]
            if len(bodies) == 1
            else [{"type": "text", "text": "189.00 EUR"}]
        )
        return httpx2.Response(
            200,
            json={
                "id": f"m{len(bodies)}",
                "type": "message",
                "role": "assistant",
                "model": "claude-haiku-5-5",
                "stop_reason": "tool_use" if len(bodies) == 1 else "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": 1, "output_tokens": 1},
                "content": content,
            },
        )

    client = anthropic.AsyncAnthropic(
        api_key="test",
        max_retries=0,
        http_client=anthropic.DefaultAsyncHttpxClient(transport=httpx2.MockTransport(handler)),
    )
    llm = AnthropicLLM("claude-haiku-5-5", client)
    first = await llm.complete(CONVERSATION, TOOLS)

    final = await llm.complete(
        [
            *CONVERSATION,
            first.as_message(),
            Message(role="tool", content=RESULT_A, tool_call_id="tu_a"),
        ],
        TOOLS,
    )

    assert final.content == "189.00 EUR"
    replayed = bodies[1]["messages"][1]["content"]
    assert replayed[0] == {"type": "thinking", "thinking": "", "signature": "SIG-42"}
    assert replayed[1]["type"] == "tool_use"
    assert bodies[1]["messages"][2]["content"][0]["content"] == RESULT_A


async def test_openai_maps_tool_calls_and_tool_results() -> None:
    llm, sent = openai_stub()

    await llm.complete(
        round_trip(Message(role="assistant", content="", tool_calls=(CALL_A, CALL_B)))
    )

    _, _, assistant, result_a, result_b = sent.kwargs["messages"]
    assert assistant == {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "tu_a",
                "type": "function",
                "function": {"name": CALL_A.name, "arguments": '{"sku": "BOIL-80-A"}'},
            },
            {
                "id": "tu_b",
                "type": "function",
                "function": {"name": CALL_B.name, "arguments": '{"sku": "KET-17-S"}'},
            },
        ],
    }
    assert result_a == {"role": "tool", "tool_call_id": "tu_a", "content": RESULT_A}
    assert result_b == {"role": "tool", "tool_call_id": "tu_b", "content": RESULT_B}


async def test_openai_plain_assistant_turn_stays_text() -> None:
    llm, sent = openai_stub()

    await llm.complete([*CONVERSATION, Message(role="assistant", content="Здравейте!")])

    assert sent.kwargs["messages"][-1] == {"role": "assistant", "content": "Здравейте!"}


@pytest.mark.parametrize("llm", [anthropic_stub()[0], openai_stub()[0]])
async def test_tool_result_without_a_call_id_is_rejected(llm: AnthropicLLM | OpenAILLM) -> None:
    with pytest.raises(ValueError, match="tool_call_id"):
        await llm.complete([*CONVERSATION, Message(role="tool", content="{}")])


# --- Provider failures -----------------------------------------------------------------

_REQUEST = httpx2.Request("POST", "https://api.example.test/v1/messages")


def _rate_limited(sdk: Any, retry_after: str | None) -> Exception:
    headers = {"retry-after": retry_after} if retry_after is not None else {}
    response = httpx2.Response(429, headers=headers, request=_REQUEST)
    return cast(Exception, sdk.RateLimitError("rate limited", response=response, body=None))


@pytest.mark.parametrize("sdk", [anthropic, openai])
@pytest.mark.parametrize(("header", "expected"), [("7", 7), ("2.5", 2), (None, None)])
async def test_rate_limit_becomes_unavailable_with_retry_after(
    sdk: Any, header: str | None, expected: int | None
) -> None:
    llm, recorder = (anthropic_stub if sdk is anthropic else openai_stub)()
    recorder.response = _rate_limited(sdk, header)

    with pytest.raises(LLMUnavailableError) as exc:
        await llm.complete(CONVERSATION)

    assert exc.value.retry_after == expected


@pytest.mark.parametrize("sdk", [anthropic, openai])
async def test_connection_error_becomes_unavailable(sdk: Any) -> None:
    llm, recorder = (anthropic_stub if sdk is anthropic else openai_stub)()
    recorder.response = sdk.APIConnectionError(request=_REQUEST)

    with pytest.raises(LLMUnavailableError) as exc:
        await llm.complete(CONVERSATION)

    assert exc.value.retry_after is None


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
