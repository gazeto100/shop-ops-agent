from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import shop_ops_agent.api.app as api
from shop_ops_agent.agent.chat import ChatService
from shop_ops_agent.api.app import app, get_chat_service
from shop_ops_agent.config import Settings
from shop_ops_agent.ingest.index import IndexModelMismatchError
from shop_ops_agent.llm import (
    LLMConfigError,
    LLMResponse,
    LLMUnavailableError,
    Message,
    ToolSpec,
)
from shop_ops_agent.rag.retriever import Hit
from tests.fakes import FakeLLM, FakeRetriever

BOILER = Hit(sku="BOIL-80-A", name="Бойлер Tesy 80 л вертикален", score=0.03, text="…")


@pytest.fixture
def client() -> Iterator[TestClient]:
    llm = FakeLLM([LLMResponse(content="Имаме [BOIL-80-A: Бойлер Tesy 80 л вертикален].")])
    app.dependency_overrides[get_chat_service] = lambda: ChatService(FakeRetriever([BOILER]), llm)
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_chat_returns_answer_citations_and_tool_calls(client: TestClient) -> None:
    r = client.post("/chat", json={"session_id": "demo", "message": "Имате ли бойлер?"})

    assert r.status_code == 200
    assert r.json() == {
        "answer": "Имаме [BOIL-80-A: Бойлер Tesy 80 л вертикален].",
        "citations": [{"sku": "BOIL-80-A", "name": "Бойлер Tesy 80 л вертикален"}],
        "tool_calls": [],
    }


@pytest.mark.parametrize(
    "body",
    [
        {"session_id": "demo", "message": ""},
        {"session_id": "", "message": "Здравей"},
        {"message": "Здравей"},
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
    service = ChatService(FakeRetriever([BOILER]), UnavailableLLM(retry_after))
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
