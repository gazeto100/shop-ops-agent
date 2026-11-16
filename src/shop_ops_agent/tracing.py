"""Langfuse tracing (#10): one trace per chat turn, with a span for retrieval, each tool
call and each LLM call (model and token usage).

Everything goes through a `Tracer`. Without Langfuse keys it is `NO_TRACER`, whose spans
do nothing: tests and local runs need no account and make no network calls.

Langfuse 5 is built on OpenTelemetry: the open span lives in a context variable, so spans
opened in awaited calls or in `asyncio.to_thread` workers (which copy the context) nest
under the turn's trace without passing a parent around. Spans are exported in the
background; Langfuse flushes the rest at interpreter exit.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, Literal, Protocol

from langfuse import Langfuse, propagate_attributes

from shop_ops_agent.config import Settings
from shop_ops_agent.llm import LLM, LLMResponse, Message, ToolSpec

type SpanType = Literal["span", "retriever", "tool"]

# Our usage keys -> Langfuse's usage_details keys.
_USAGE_KEYS = {"input_tokens": "input", "output_tokens": "output"}


class Observation(Protocol):
    """The part of a Langfuse span we use: record what came out of the step."""

    def update(
        self,
        *,
        output: Any = None,
        metadata: Any = None,
        usage_details: dict[str, int] | None = None,
    ) -> object: ...


class _NoObservation:
    def update(
        self,
        *,
        output: Any = None,
        metadata: Any = None,
        usage_details: dict[str, int] | None = None,
    ) -> None:
        pass


_NO_OBSERVATION = _NoObservation()


class Tracer:
    """Opens Langfuse spans, or nothing when `client` is None. An exception raised inside
    a span propagates unchanged; Langfuse marks the span as an error first."""

    def __init__(self, client: Langfuse | None) -> None:
        self._client = client

    @contextmanager
    def trace(
        self, name: str, *, session_id: str, input: Any, metadata: dict[str, Any]
    ) -> Iterator[Observation]:
        """The root span: one per chat turn. Spans opened inside belong to its trace, and
        traces with the same session id are grouped as one conversation in Langfuse."""
        if self._client is None:
            yield _NO_OBSERVATION
            return
        with (
            self._client.start_as_current_observation(
                name=name, as_type="agent", input=input, metadata=metadata
            ) as root,
            propagate_attributes(session_id=session_id, trace_name=name),
        ):
            yield root

    @contextmanager
    def span(self, name: str, as_type: SpanType = "span", *, input: Any) -> Iterator[Observation]:
        """A step inside the current trace (or a trace of its own outside one)."""
        if self._client is None:
            yield _NO_OBSERVATION
            return
        with self._client.start_as_current_observation(
            name=name, as_type=as_type, input=input
        ) as span:
            yield span

    @contextmanager
    def generation(self, name: str, *, input: Any, model: str) -> Iterator[Observation]:
        """An LLM call: Langfuse shows the model and prices the token usage."""
        if self._client is None:
            yield _NO_OBSERVATION
            return
        with self._client.start_as_current_observation(
            name=name, as_type="generation", input=input, model=model
        ) as generation:
            yield generation


NO_TRACER = Tracer(None)


def create_tracer(settings: Settings) -> Tracer:
    """Langfuse when both LANGFUSE_* keys are set, otherwise the no-op tracer."""
    if not (settings.langfuse_public_key and settings.langfuse_secret_key):
        return NO_TRACER
    return Tracer(
        Langfuse(
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key,
            base_url=settings.langfuse_host,
        )
    )


def _message_view(m: Message) -> dict[str, Any]:
    """What a trace shows of a message. Provider blocks (thinking) are left out."""
    view: dict[str, Any] = {"role": m.role, "content": m.content}
    if m.tool_calls:
        view["tool_calls"] = [{"name": c.name, "arguments": c.arguments} for c in m.tool_calls]
    if m.tool_call_id:
        view["tool_call_id"] = m.tool_call_id
    return view


class TracedLLM:
    """Wraps an `LLM` so every call is a generation span with the model, the messages
    sent, the reply and the token usage. Satisfies the `LLM` protocol itself."""

    def __init__(self, llm: LLM, model: str, tracer: Tracer) -> None:
        self._llm = llm
        self._model = model
        self._tracer = tracer

    async def complete(
        self, messages: list[Message], tools: list[ToolSpec] | None = None
    ) -> LLMResponse:
        view = [_message_view(m) for m in messages]
        with self._tracer.generation("llm", input=view, model=self._model) as span:
            response = await self._llm.complete(messages, tools)
            calls = [{"name": c.name, "arguments": c.arguments} for c in response.tool_calls]
            span.update(
                output={"content": response.content, "tool_calls": calls},
                usage_details={
                    _USAGE_KEYS[k]: v for k, v in response.usage.items() if k in _USAGE_KEYS
                },
            )
        return response
