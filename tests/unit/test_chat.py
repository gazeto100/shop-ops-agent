from typing import Any

import pytest

from shop_ops_agent.agent.chat import ChatService, cited_hits, detect_language
from shop_ops_agent.agent.prompts import PROMPT_VERSION, REFUSAL_REPLY, SYSTEM_PROMPT
from shop_ops_agent.llm import LLMRefusalError, LLMResponse, Message, ToolCall
from shop_ops_agent.rag.retriever import Hit
from tests.fakes import FakeLLM, FakeRetriever

BOILER = Hit(
    sku="BOIL-80-A",
    name="Бойлер Tesy 80 л вертикален",
    score=0.033,
    text="Бойлер Tesy 80 л вертикален\nКатегория: Бойлери\nГаранция: 60 месеца",
)
KETTLE = Hit(
    sku="KET-17-S",
    name="Електрическа кана 1.7 л инокс",
    score=0.016,
    text="Електрическа кана 1.7 л инокс\nКатегория: Малки уреди\nГаранция: 24 месеца",
)


class RefusingLLM:
    async def complete(
        self, messages: list[Message], tools: list[dict[str, Any]] | None = None
    ) -> LLMResponse:
        raise LLMRefusalError("general_harms")


def service(llm: Any, hits: tuple[Hit, ...] = (BOILER, KETTLE)) -> ChatService:
    return ChatService(FakeRetriever(hits), llm)


async def test_retrieved_context_is_injected_into_the_prompt() -> None:
    llm = FakeLLM()

    await service(llm).answer("Какъв бойлер имате?")

    [(messages, tools)] = llm.calls
    system, user = messages
    assert system == Message(role="system", content=SYSTEM_PROMPT)
    assert user.role == "user"
    for hit in (BOILER, KETTLE):
        assert hit.citation() in user.content
        assert hit.text in user.content
    assert "<question>\nКакъв бойлер имате?\n</question>" in user.content
    assert tools is None  # no tools until #6


async def test_question_is_the_retrieval_query() -> None:
    retriever = FakeRetriever((BOILER,))

    await ChatService(retriever, FakeLLM(), k=3).answer("бойлер 80 литра")

    assert retriever.searches == [("бойлер 80 литра", 3)]


async def test_no_hits_says_so_in_the_prompt() -> None:
    llm = FakeLLM()

    await service(llm, hits=()).answer("Имате ли телевизори?")

    assert "(no matching products)" in llm.calls[0][0][1].content


@pytest.mark.parametrize(
    ("question", "language", "instruction"),
    [
        ("Има ли наличен бойлер?", "bg", "Reply in Bulgarian."),
        ("Do you sell kettles?", "en", "Reply in English."),
        ("Do you sell кани?", "bg", "Reply in Bulgarian."),  # any Cyrillic -> Bulgarian
        ("12000 BTU?", "en", "Reply in English."),
        ("80?", "bg", "Reply in Bulgarian."),  # no letters -> the shop's default
    ],
)
async def test_reply_language_follows_the_user(
    question: str, language: str, instruction: str
) -> None:
    llm = FakeLLM()

    await service(llm).answer(question)

    assert detect_language(question) == language
    assert instruction in llm.calls[0][0][1].content


async def test_citations_are_only_retrieved_products_the_answer_cites() -> None:
    answer = "Препоръчвам [BOIL-80-A: Бойлер Tesy 80 л вертикален] и [FAKE-1: Измислен]."
    llm = FakeLLM([LLMResponse(content=answer)])

    result = await service(llm).answer("Какъв бойлер?")

    assert result.answer == answer
    assert result.citations == (BOILER,)  # KETTLE wasn't cited, FAKE-1 wasn't retrieved


def test_cited_hits_keeps_first_citation_order_without_duplicates() -> None:
    answer = "[KET-17-S: кана], после [BOIL-80-A: бойлер] и пак [KET-17-S: кана]"

    assert cited_hits(answer, [BOILER, KETTLE]) == (KETTLE, BOILER)


async def test_tool_calls_are_passed_through(fake_llm_with_tool_call: FakeLLM) -> None:
    result = await service(fake_llm_with_tool_call).answer("Колко е бойлерът?")

    assert result.tool_calls == (
        ToolCall(id="c1", name="check_stock_and_price", arguments={"sku": "X1"}),
    )


@pytest.mark.parametrize(("question", "language"), [("Как да…?", "bg"), ("How do I…?", "en")])
async def test_refusal_returns_a_polite_reply_in_the_users_language(
    question: str, language: str
) -> None:
    result = await service(RefusingLLM()).answer(question)

    assert result.answer == REFUSAL_REPLY[language]  # type: ignore[index]
    assert result.citations == ()


def test_system_prompt_forbids_inventing_prices_and_stock() -> None:
    # Guards the non-negotiable rule against an accidental prompt edit.
    assert PROMPT_VERSION
    assert "Never state, estimate or guess a price" in SYSTEM_PROMPT
    assert "stock level" in SYSTEM_PROMPT
    assert "even if the user supplies a price" in SYSTEM_PROMPT
