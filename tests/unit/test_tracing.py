"""Langfuse tracing (#10). A real Langfuse client exports to an in-memory OpenTelemetry
exporter instead of the network, so the tests see exactly the spans Langfuse would get."""

import json
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from langfuse import Langfuse
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode

import shop_ops_agent.api.app as api
import shop_ops_agent.tracing as tracing
from shop_ops_agent.agent.chat import ChatService
from shop_ops_agent.agent.memory import SessionMemory
from shop_ops_agent.agent.prompts import PROMPT_VERSION
from shop_ops_agent.agent.tools import Catalog, build_registry
from shop_ops_agent.api.app import app, get_chat_service
from shop_ops_agent.config import Settings
from shop_ops_agent.llm import LLMResponse, LLMUnavailableError, Message, ToolCall, ToolSpec
from shop_ops_agent.rag.retriever import Hit
from shop_ops_agent.tracing import NO_TRACER, TracedLLM, Tracer, create_tracer
from tests.fakes import FakeLLM, FakeRetriever

SAMPLE = Path(__file__).parents[2] / "data" / "products.sample.csv"
BOILER = Hit(sku="BOIL-80-A", name="Бойлер Tesy 80 л вертикален", score=0.03, text="…")
CHECK_BOILER = ToolCall(id="c1", name="check_stock_and_price", arguments={"sku": "BOIL-80-A"})
ANSWER = "[BOIL-80-A: Бойлер Tesy 80 л вертикален] е 189.00 EUR."


class Spans:
    def __init__(self, client: Langfuse, exporter: InMemorySpanExporter) -> None:
        self.client = client
        self.exporter = exporter

    def finished(self) -> list[ReadableSpan]:
        self.client.flush()
        return list(self.exporter.get_finished_spans())

    def named(self, name: str) -> ReadableSpan:
        [span] = [s for s in self.finished() if s.name == name]
        return span


@pytest.fixture
def spans() -> Iterator[Spans]:
    exporter = InMemorySpanExporter()
    # A fresh key per test: Langfuse keeps one client per public key for the process.
    client = Langfuse(
        public_key=f"pk-test-{uuid.uuid4()}",
        secret_key="sk-test",
        base_url="http://127.0.0.1:9",  # never contacted: spans go to the exporter
        tracer_provider=TracerProvider(),
        span_exporter=exporter,
    )
    yield Spans(client, exporter)
    client.shutdown()


def attr(span: ReadableSpan, key: str) -> Any:
    """A Langfuse attribute: objects arrive JSON-encoded, plain strings as they are."""
    value = (span.attributes or {})[f"langfuse.observation.{key}"]
    try:
        return json.loads(value) if isinstance(value, str) else value
    except json.JSONDecodeError:
        return value


def traced_service(tracer: Tracer, llm: FakeLLM) -> ChatService:
    retriever = FakeRetriever([BOILER])
    registry = build_registry(Catalog(SAMPLE), retriever)
    traced = TracedLLM(llm, "claude-test", tracer)
    return ChatService(retriever, traced, registry, SessionMemory(), tracer=tracer)


def tool_then_answer() -> FakeLLM:
    usage = {"input_tokens": 120, "output_tokens": 15}
    return FakeLLM(
        [
            LLMResponse(content="", tool_calls=(CHECK_BOILER,), usage=usage),
            LLMResponse(content=ANSWER, usage={"input_tokens": 200, "output_tokens": 30}),
        ]
    )


def test_one_chat_request_is_one_trace_with_a_span_per_step(spans: Spans) -> None:
    service = traced_service(Tracer(spans.client), tool_then_answer())
    app.dependency_overrides[get_chat_service] = lambda: service
    try:
        r = TestClient(app).post("/chat", json={"message": "Колко струва бойлерът?"})
    finally:
        app.dependency_overrides.clear()

    assert r.status_code == 200
    finished = spans.finished()
    assert [(s.name, attr(s, "type")) for s in finished] == [
        ("retrieval", "retriever"),
        ("llm", "generation"),
        ("check_stock_and_price", "tool"),
        ("llm", "generation"),
        ("chat", "agent"),
    ]
    root = spans.named("chat")
    assert {s.context.trace_id for s in finished} == {root.context.trace_id}
    assert all(s.parent and s.parent.span_id == root.context.span_id for s in finished[:-1])
    assert root.parent is None
    # Every span carries the session, so Langfuse groups a conversation's turns.
    assert {(s.attributes or {})["session.id"] for s in finished} == {r.json()["session_id"]}


def test_the_app_traces_chat_with_the_configured_model(
    spans: Spans, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = Settings.model_construct(llm_provider="anthropic", llm_model="claude-test")
    monkeypatch.setattr(api, "get_settings", lambda: settings)
    monkeypatch.setattr(api, "create_llm", lambda _: tool_then_answer())
    monkeypatch.setattr(api, "create_tracer", lambda _: Tracer(spans.client))
    monkeypatch.setattr(api, "open_collection", lambda *_: None)
    monkeypatch.setattr(api, "SentenceTransformerEmbedder", lambda _: None)
    monkeypatch.setattr(api, "Retriever", lambda *_: FakeRetriever([BOILER]))
    get_chat_service.cache_clear()
    try:
        r = TestClient(app).post("/chat", json={"message": "Колко струва бойлерът?"})
    finally:
        get_chat_service.cache_clear()

    assert r.status_code == 200
    names = [s.name for s in spans.finished()]
    assert names == ["retrieval", "llm", "check_stock_and_price", "llm", "chat"]
    assert attr(spans.finished()[1], "model.name") == "claude-test"


async def test_the_trace_records_question_answer_and_prompt_version(spans: Spans) -> None:
    service = traced_service(Tracer(spans.client), tool_then_answer())

    result = await service.answer(None, "Колко струва бойлерът?")

    root = spans.named("chat")
    assert attr(root, "input") == "Колко струва бойлерът?"
    assert attr(root, "output") == result.answer == ANSWER
    assert attr(root, "metadata.prompt_version") == PROMPT_VERSION
    assert attr(root, "metadata.audience") == "customer"


async def test_spans_record_retrieval_tool_and_llm_details(spans: Spans) -> None:
    service = traced_service(Tracer(spans.client), tool_then_answer())

    await service.answer(None, "Бойлер?")

    retrieval = spans.named("retrieval")
    assert attr(retrieval, "input") == "Бойлер?"
    assert attr(retrieval, "output") == [{"sku": "BOIL-80-A", "score": 0.03}]

    tool = spans.named("check_stock_and_price")
    assert attr(tool, "input") == {"sku": "BOIL-80-A"}
    assert attr(tool, "output")["price_eur"] == "189.00"

    first, second = (s for s in spans.finished() if s.name == "llm")
    assert attr(first, "model.name") == "claude-test"
    assert attr(first, "usage_details") == {"input": 120, "output": 15}
    assert attr(first, "output")["tool_calls"] == [
        {"name": "check_stock_and_price", "arguments": {"sku": "BOIL-80-A"}}
    ]
    assert attr(second, "usage_details") == {"input": 200, "output": 30}
    assert attr(second, "output")["content"] == ANSWER
    # The second call shows the tool round trip it was given.
    sent = attr(second, "input")
    assert sent[-2]["tool_calls"][0]["name"] == "check_stock_and_price"
    assert sent[-1]["role"] == "tool" and sent[-1]["tool_call_id"] == "c1"


class FailingLLM:
    async def complete(
        self, messages: list[Message], tools: list[ToolSpec] | None = None
    ) -> LLMResponse:
        raise LLMUnavailableError("overloaded")


async def test_a_failed_llm_call_is_an_error_span_and_still_raises(spans: Spans) -> None:
    llm = TracedLLM(FailingLLM(), "claude-test", Tracer(spans.client))

    with pytest.raises(LLMUnavailableError):
        await llm.complete([Message(role="user", content="Здравей")])

    span = spans.named("llm")
    assert span.status.status_code == StatusCode.ERROR
    assert attr(span, "input") == [{"role": "user", "content": "Здравей"}]


# --- No keys, no tracing ------------------------------------------------------------------


def test_without_langfuse_keys_tracing_is_off(monkeypatch: pytest.MonkeyPatch) -> None:
    def no_client(**_: object) -> None:
        raise AssertionError("Langfuse must not be created without keys")

    monkeypatch.setattr(tracing, "Langfuse", no_client)

    assert create_tracer(Settings.model_construct()) is NO_TRACER
    assert create_tracer(Settings.model_construct(langfuse_public_key="pk")) is NO_TRACER


def test_with_both_keys_langfuse_is_configured_from_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created: list[dict[str, object]] = []
    monkeypatch.setattr(tracing, "Langfuse", lambda **kwargs: created.append(kwargs))
    settings = Settings.model_construct(
        langfuse_public_key="pk", langfuse_secret_key="sk", langfuse_host="https://lf.example"
    )

    assert create_tracer(settings) is not NO_TRACER
    assert created == [{"public_key": "pk", "secret_key": "sk", "base_url": "https://lf.example"}]


async def test_the_no_op_tracer_changes_nothing() -> None:
    service = traced_service(NO_TRACER, tool_then_answer())

    result = await service.answer(None, "Колко струва бойлерът?")

    assert result.answer == ANSWER
    with pytest.raises(ValueError, match="boom"), NO_TRACER.span("x", input=None) as span:
        span.update(output="ignored")
        raise ValueError("boom")
