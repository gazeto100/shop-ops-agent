"""Provider-agnostic LLM interface and the OpenAI / Anthropic / Ollama adapters.

Tool definitions use one neutral shape, `ToolSpec`; each adapter translates it. Provider
failures (network, rate limits, 5xx, auth) surface as `LLMUnavailableError`, so callers
never handle SDK-specific exceptions. Tool *results* (role "tool")
need assistant tool-call turns and, on Claude, replayed thinking blocks, which `Message`
can't carry yet; the agent loop (#7) adds them, so adapters reject them for now.
"""

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, NotRequired, Protocol, TypedDict, assert_never

import anthropic
import openai
from anthropic.types import MessageParam as AnthropicMessageParam
from anthropic.types import ToolParam as AnthropicToolParam
from openai.types.chat import ChatCompletionFunctionToolParam, ChatCompletionMessageParam

from shop_ops_agent.config import Settings

type Role = Literal["system", "user", "assistant", "tool"]


@dataclass(frozen=True)
class Message:
    role: Role
    content: str
    tool_call_id: str | None = None


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class LLMResponse:
    content: str
    tool_calls: tuple[ToolCall, ...] = ()
    usage: dict[str, int] = field(default_factory=dict)


class ToolSpec(TypedDict):
    """A tool the model may call; `parameters` is the JSON Schema of its arguments."""

    name: str
    description: NotRequired[str]
    parameters: dict[str, Any]


class LLM(Protocol):
    async def complete(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
    ) -> LLMResponse: ...


class LLMRefusalError(Exception):
    """The provider's safety layer declined the request (HTTP 200, not an API error)."""


class LLMConfigError(Exception):
    """The selected provider can't be used with the current settings."""


class LLMUnavailableError(Exception):
    """The provider call failed (network, rate limit, server or auth error)."""

    def __init__(self, message: str, retry_after: int | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after  # seconds, when the provider said so (429)


def _retry_after(headers: Any) -> int | None:
    value = headers.get("retry-after")
    try:
        return max(0, int(float(value))) if value is not None else None
    except ValueError:
        return None  # HTTP-date form; callers fall back to their own default


def _reject_tool_turns(messages: Sequence[Message]) -> None:
    if any(m.role == "tool" for m in messages):
        raise NotImplementedError("tool-result messages are wired up with the agent loop (#7)")


class AnthropicLLM:
    # No `effort` and no temperature/top_p: the live API rejects `effort` for
    # claude-haiku-5-5 (400, see DECISIONS.md) and current models reject sampling params.
    # Thinking counts toward max_tokens, so leave room beyond the visible answer.
    MAX_TOKENS = 8000

    def __init__(self, model: str, client: anthropic.AsyncAnthropic) -> None:
        self._model = model
        self._client = client

    async def complete(
        self, messages: list[Message], tools: list[ToolSpec] | None = None
    ) -> LLMResponse:
        _reject_tool_turns(messages)
        # System text goes in the top-level `system` field, not in `messages`.
        system = "\n\n".join(m.content for m in messages if m.role == "system")
        turns: list[AnthropicMessageParam] = []
        for m in messages:
            match m.role:
                case "user":
                    turns.append({"role": "user", "content": m.content})
                case "assistant":
                    turns.append({"role": "assistant", "content": m.content})
        tool_params: list[AnthropicToolParam] = [
            {
                "name": t["name"],
                "description": t.get("description", ""),
                "input_schema": t["parameters"],
            }
            for t in tools or []
        ]
        try:
            response = await self._client.messages.create(
                model=self._model,
                max_tokens=self.MAX_TOKENS,
                system=system or anthropic.omit,
                messages=turns,
                tools=tool_params or anthropic.omit,
            )
        # Most specific first; the SDK has already retried 429/5xx/connection errors.
        except anthropic.RateLimitError as e:
            raise LLMUnavailableError(str(e), _retry_after(e.response.headers)) from e
        except anthropic.APIError as e:
            raise LLMUnavailableError(str(e)) from e
        # Check before reading content: a refusal is a normal 200 response.
        if response.stop_reason == "refusal":
            raise LLMRefusalError(str(response.stop_details))
        # Read blocks by type: responses can start with (empty) thinking blocks.
        text = "".join(b.text for b in response.content if b.type == "text")
        calls = tuple(
            ToolCall(id=b.id, name=b.name, arguments=dict(b.input))
            for b in response.content
            if b.type == "tool_use"
        )
        usage = {
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
        }
        return LLMResponse(content=text, tool_calls=calls, usage=usage)


class OpenAILLM:
    def __init__(self, model: str, client: openai.AsyncOpenAI) -> None:
        self._model = model
        self._client = client

    async def complete(
        self, messages: list[Message], tools: list[ToolSpec] | None = None
    ) -> LLMResponse:
        _reject_tool_turns(messages)
        turns: list[ChatCompletionMessageParam] = []
        for m in messages:
            match m.role:
                case "system":
                    turns.append({"role": "system", "content": m.content})
                case "user":
                    turns.append({"role": "user", "content": m.content})
                case "assistant":
                    turns.append({"role": "assistant", "content": m.content})
        tool_params: list[ChatCompletionFunctionToolParam] = [
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t.get("description", ""),
                    "parameters": t["parameters"],
                },
            }
            for t in tools or []
        ]
        try:
            response = await self._client.chat.completions.create(
                model=self._model, messages=turns, tools=tool_params or openai.omit
            )
        except openai.RateLimitError as e:
            raise LLMUnavailableError(str(e), _retry_after(e.response.headers)) from e
        except openai.APIError as e:
            raise LLMUnavailableError(str(e)) from e
        message = response.choices[0].message
        if message.refusal:
            raise LLMRefusalError(message.refusal)
        calls = tuple(
            # Arguments arrive as a JSON string; always parse, never string-match.
            ToolCall(id=c.id, name=c.function.name, arguments=json.loads(c.function.arguments))
            for c in message.tool_calls or []
            if c.type == "function"
        )
        usage = (
            {
                "input_tokens": response.usage.prompt_tokens,
                "output_tokens": response.usage.completion_tokens,
            }
            if response.usage
            else {}
        )
        return LLMResponse(content=message.content or "", tool_calls=calls, usage=usage)


class OllamaLLM(OpenAILLM):
    """Ollama through its OpenAI-compatible endpoint (`<base>/v1`), reusing the mapping."""

    def __init__(self, model: str, base_url: str) -> None:
        # Ollama ignores the key, but the OpenAI client refuses to start without one.
        client = openai.AsyncOpenAI(api_key="ollama", base_url=f"{base_url.rstrip('/')}/v1")
        super().__init__(model, client)


def create_llm(settings: Settings) -> LLM:
    """The adapter for LLM_PROVIDER, with LLM_MODEL or that provider's default model."""
    model = settings.resolved_llm_model
    match settings.llm_provider:
        case "openai":
            if not settings.openai_api_key:
                raise LLMConfigError("LLM_PROVIDER=openai needs OPENAI_API_KEY")
            return OpenAILLM(model, openai.AsyncOpenAI(api_key=settings.openai_api_key))
        case "anthropic":
            if not settings.anthropic_api_key:
                raise LLMConfigError("LLM_PROVIDER=anthropic needs ANTHROPIC_API_KEY")
            return AnthropicLLM(model, anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key))
        case "ollama":
            return OllamaLLM(model, settings.ollama_base_url)
        case _:
            assert_never(settings.llm_provider)
