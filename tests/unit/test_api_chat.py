from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from shop_ops_agent.agent.chat import ChatService
from shop_ops_agent.api.app import app, get_chat_service
from shop_ops_agent.llm import LLMResponse
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
