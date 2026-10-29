from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import shop_ops_agent.api.app as api
from shop_ops_agent.agent.chat import ChatService
from shop_ops_agent.agent.memory import SessionMemory
from shop_ops_agent.agent.tools import Catalog, build_registry
from shop_ops_agent.api.app import app, get_chat_service
from shop_ops_agent.config import Settings
from shop_ops_agent.ingest.index import IndexModelMismatchError
from shop_ops_agent.llm import (
    LLM,
    LLMConfigError,
    LLMResponse,
    LLMUnavailableError,
    Message,
    ToolCall,
    ToolSpec,
)
from shop_ops_agent.rag.retriever import Hit
from tests.fakes import FakeLLM, FakeRetriever

SAMPLE = Path(__file__).parents[2] / "data" / "products.sample.csv"
BOILER = Hit(sku="BOIL-80-A", name="Бойлер Tesy 80 л вертикален", score=0.03, text="…")


def chat_service(llm: LLM, memory: SessionMemory | None = None) -> ChatService:
    retriever = FakeRetriever([BOILER])
    registry = build_registry(Catalog(SAMPLE), retriever)
    return ChatService(retriever, llm, registry, memory or SessionMemory())


@pytest.fixture
def client() -> Iterator[TestClient]:
    llm = FakeLLM([LLMResponse(content="Имаме [BOIL-80-A: Бойлер Tesy 80 л вертикален].")])
    service = chat_service(llm)
    app.dependency_overrides[get_chat_service] = lambda: service
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_chat_runs_tools_and_reports_them() -> None:
    check = ToolCall(id="c1", name="check_stock_and_price", arguments={"sku": "BOIL-80-A"})
    llm = FakeLLM(
        [
            LLMResponse(content="", tool_calls=(check,)),
            LLMResponse(content="[BOIL-80-A: Бойлер Tesy 80 л вертикален] е 189.00 EUR."),
        ]
    )
    service = chat_service(llm)
    app.dependency_overrides[get_chat_service] = lambda: service
    try:
        r = TestClient(app).post("/chat", json={"session_id": "s1", "message": "Цена?"})
    finally:
        app.dependency_overrides.clear()

    assert r.status_code == 200
    assert r.json()["answer"] == "[BOIL-80-A: Бойлер Tesy 80 л вертикален] е 189.00 EUR."
    assert r.json()["tool_calls"] == [
        {"name": "check_stock_and_price", "arguments": {"sku": "BOIL-80-A"}}
    ]


def test_returned_session_id_carries_memory_between_requests() -> None:
    llm = FakeLLM([LLMResponse(content="Здравейте!")])
    service = chat_service(llm)
    app.dependency_overrides[get_chat_service] = lambda: service
    try:
        client = TestClient(app)
        session = client.post("/chat", json={"message": "Здравей"}).json()["session_id"]
        again = client.post("/chat", json={"session_id": session, "message": "Как си?"})
        other = client.post("/chat", json={"message": "Здравей"})
    finally:
        app.dependency_overrides.clear()

    assert again.json()["session_id"] == session
    assert other.json()["session_id"] != session
    second, third = llm.calls[1][0], llm.calls[2][0]
    assert Message(role="user", content="Здравей") in second  # the session remembered
    assert len(third) == 2  # a new session starts fresh: system + question


def test_a_guessed_session_id_does_not_reveal_another_conversation() -> None:
    llm = FakeLLM([LLMResponse(content="Добре, Иван.")])
    service = chat_service(llm)
    app.dependency_overrides[get_chat_service] = lambda: service
    try:
        client = TestClient(app)
        client.post("/chat", json={"session_id": "demo", "message": "Аз съм Иван"})
        r = client.post("/chat", json={"session_id": "demo", "message": "Кой съм аз?"})
    finally:
        app.dependency_overrides.clear()

    assert r.json()["session_id"] != "demo"  # client-chosen ids are never adopted
    assert all("Иван" not in m.content for m in llm.calls[1][0][1:-1])


def test_chat_returns_session_answer_citations_and_tool_calls(client: TestClient) -> None:
    r = client.post("/chat", json={"message": "Имате ли бойлер?"})

    assert r.status_code == 200
    body = r.json()
    assert len(body.pop("session_id")) >= 20
    assert body == {
        "answer": "Имаме [BOIL-80-A: Бойлер Tesy 80 л вертикален].",
        "citations": [{"sku": "BOIL-80-A", "name": "Бойлер Tesy 80 л вертикален"}],
        "tool_calls": [],
    }


@pytest.mark.parametrize(
    "body",
    [
        {"session_id": "demo", "message": ""},
        {"session_id": "", "message": "Здравей"},
        {"session_id": "x" * 129, "message": "Здравей"},
        {"session_id": "demo"},
    ],
)
def test_invalid_request_is_rejected(client: TestClient, body: dict[str, str]) -> None:
    assert client.post("/chat", json=body).status_code == 422


# --- Failures -> 503, never a bare 500 -------------------------------------------------

BODY = {"session_id": "demo", "message": "Имате ли бойлер?"}


class UnavailableLLM:
    def __init__(self, retry_after: int | None) -> None:
        self.retry_after = retry_after

    async def complete(
        self, messages: list[Message], tools: list[ToolSpec] | None = None
    ) -> LLMResponse:
        raise LLMUnavailableError("upstream 529", retry_after=self.retry_after)


def _raise(exc: Exception) -> ChatService:
    raise exc


@pytest.mark.parametrize(
    "exc",
    [
        LLMConfigError("LLM_PROVIDER=anthropic needs ANTHROPIC_API_KEY"),
        IndexModelMismatchError(Path("data/chroma"), "other/model", "intfloat/e5"),
    ],
)
def test_misconfiguration_is_a_503_without_leaking_details(exc: Exception) -> None:
    app.dependency_overrides[get_chat_service] = lambda: _raise(exc)
    try:
        r = TestClient(app).post("/chat", json=BODY)
    finally:
        app.dependency_overrides.clear()

    assert r.status_code == 503
    assert r.json() == {"detail": "Chat service is not configured."}
    assert "API_KEY" not in r.text and "chroma" not in r.text


@pytest.mark.parametrize(("retry_after", "header"), [(7, "7"), (None, "30")])
def test_provider_outage_is_a_503_with_retry_after(retry_after: int | None, header: str) -> None:
    service = chat_service(UnavailableLLM(retry_after))
    app.dependency_overrides[get_chat_service] = lambda: service
    try:
        r = TestClient(app).post("/chat", json=BODY)
    finally:
        app.dependency_overrides.clear()

    assert r.status_code == 503
    assert r.headers["retry-after"] == header


def test_bad_config_fails_before_the_embedding_model_loads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    built: list[str] = []
    monkeypatch.setattr(api, "SentenceTransformerEmbedder", lambda model: built.append(model))
    monkeypatch.setattr(
        api, "get_settings", lambda: Settings.model_construct(llm_provider="anthropic")
    )
    get_chat_service.cache_clear()
    try:
        for _ in range(3):  # lru_cache doesn't cache exceptions: every request retries
            assert TestClient(app).post("/chat", json=BODY).status_code == 503
    finally:
        get_chat_service.cache_clear()

    assert built == []
