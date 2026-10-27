import pytest

from shop_ops_agent.agent.chat import ChatService, cited_hits, cited_skus, detect_language
from shop_ops_agent.agent.prompts import (
    PRICE_UNAVAILABLE_REPLY,
    PRODUCT_NOT_FOUND_REPLY,
    PROMPT_VERSION,
    REFUSAL_REPLY,
    SYSTEM_PROMPT,
    Language,
)
from shop_ops_agent.llm import LLM, LLMRefusalError, LLMResponse, Message, ToolCall, ToolSpec
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
        self, messages: list[Message], tools: list[ToolSpec] | None = None
    ) -> LLMResponse:
        raise LLMRefusalError("general_harms")


def service(llm: LLM, hits: tuple[Hit, ...] = (BOILER, KETTLE)) -> ChatService:
    return ChatService(FakeRetriever(hits), llm)


def answering(text: str) -> FakeLLM:
    return FakeLLM([LLMResponse(content=text)])


# --- Prompt -----------------------------------------------------------------------------


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


def test_system_prompt_forbids_inventing_prices_and_stock() -> None:
    # Guards the non-negotiable rule against an accidental prompt edit.
    assert "Never state, estimate or guess a price" in SYSTEM_PROMPT
    assert "stock level" in SYSTEM_PROMPT
    assert "even if the user supplies a price" in SYSTEM_PROMPT


def test_system_prompt_shows_a_concrete_citation_and_forbids_the_sku_word() -> None:
    # The live model copied "[sku: name]" literally; a concrete example fixes the format.
    assert "[ABC-123: Примерен продукт 2 л]" in SYSTEM_PROMPT
    assert '"sku"' in SYSTEM_PROMPT
    assert "[sku: name]" not in SYSTEM_PROMPT


# --- Language ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("question", "language", "instruction"),
    [
        ("Има ли наличен бойлер?", "bg", "Reply in Bulgarian."),
        ("Do you sell kettles?", "en", "Reply in English."),
        ("What is the warranty on the Tesy?", "en", "Reply in English."),
        ("Do you sell кани?", "bg", "Reply in Bulgarian."),  # any Cyrillic -> Bulgarian
        ("imate li boiler 80 litra?", "bg", "Reply in Bulgarian."),  # Bulgarian, Latin letters
        ("do 35 kv.m klimatik?", "bg", "Reply in Bulgarian."),  # "do" here is "до"
        ("12000 BTU?", "bg", "Reply in Bulgarian."),  # no English words -> shop default
        ("80?", "bg", "Reply in Bulgarian."),
    ],
)
async def test_reply_language_follows_the_user(
    question: str, language: Language, instruction: str
) -> None:
    llm = FakeLLM()

    await service(llm).answer(question)

    assert detect_language(question) == language
    assert instruction in llm.calls[0][0][1].content


# --- Citations --------------------------------------------------------------------------


async def test_live_answer_with_sku_word_still_yields_the_citation() -> None:
    # Exact citation text from the live Anthropic run that produced citations=[].
    live = "Имаме [sku: BOIL-80-A: Бойлер Tesy 80 л вертикален] с гаранция 60 месеца."

    result = await service(answering(live)).answer("Какъв бойлер имате?")

    assert result.answer == live
    assert result.citations == (BOILER,)


@pytest.mark.parametrize(
    "answer",
    [
        "[BOIL-80-A: Бойлер Tesy 80 л вертикален]",
        "[sku: BOIL-80-A: Бойлер Tesy 80 л вертикален]",
        "[SKU: BOIL-80-A: Бойлер Tesy 80 л вертикален]",
        "[ BOIL-80-A : Бойлер Tesy 80 л вертикален]",
    ],
)
def test_both_citation_forms_are_parsed(answer: str) -> None:
    assert cited_skus(answer) == ("BOIL-80-A",)


def test_bracketed_text_that_is_not_a_product_code_is_not_a_citation() -> None:
    assert cited_skus("[Бележка: важно] [Note: see below] [BOIL-80-A: x]") == ("BOIL-80-A",)


async def test_citations_are_only_products_the_answer_cites() -> None:
    result = await service(answering("Препоръчвам [BOIL-80-A: Бойлер].")).answer("Бойлер?")

    assert result.citations == (BOILER,)  # KETTLE was retrieved but not cited


def test_cited_hits_keeps_first_citation_order_without_duplicates() -> None:
    answer = "[KET-17-S: кана], после [BOIL-80-A: бойлер] и пак [KET-17-S: кана]"

    assert cited_hits(answer, [BOILER, KETTLE]) == (KETTLE, BOILER)


# --- Guardrails -------------------------------------------------------------------------


@pytest.mark.parametrize(
    "answer",
    [
        "[BOIL-80-A: Бойлер Tesy 80 л вертикален] струва 1 EUR.",
        "Цената е 189,00 лв., наличен е.",
        "Наличен: 5 бр.",
        "It costs €189 and we have 12 in stock.",
    ],
)
async def test_invented_price_or_stock_never_reaches_the_user(answer: str) -> None:
    result = await service(answering(answer)).answer("Колко струва бойлерът?")

    assert result.answer == PRICE_UNAVAILABLE_REPLY["bg"]
    assert result.guardrail == "price_or_stock"
    assert result.citations == ()


@pytest.mark.parametrize(
    "answer",
    [
        "Гаранция: 60 месеца, [BOIL-80-A: Бойлер Tesy 80 л вертикален], 2 kW, 80 л.",
        "Не мога да потвърдя текущата цена и наличност в момента.",
        "12000 BTU, до 35 кв.м [AC-12-INV: Климатик]",
    ],
)
async def test_specs_and_honest_refusals_are_not_mistaken_for_prices(answer: str) -> None:
    hits = (BOILER, Hit(sku="AC-12-INV", name="Климатик", score=0.01, text="…"))

    result = await service(answering(answer), hits).answer("Какво имате?")

    assert result.guardrail is None
    assert result.answer == answer


async def test_cited_product_that_was_not_retrieved_replaces_the_answer() -> None:
    result = await service(answering("Вижте [FAKE-9: Супер бойлер].")).answer("бойлер?")

    assert result.answer == PRODUCT_NOT_FOUND_REPLY["bg"]
    assert result.guardrail == "unverified_product"
    assert result.unverified_skus == ("FAKE-9",)
    assert result.citations == ()


@pytest.mark.parametrize(("question", "language"), [("Как да…?", "bg"), ("How do I…?", "en")])
async def test_refusal_returns_a_polite_reply_in_the_users_language(
    question: str, language: Language
) -> None:
    result = await service(RefusingLLM()).answer(question)

    assert result.answer == REFUSAL_REPLY[language]
    assert result.citations == ()


# --- Pass-through -----------------------------------------------------------------------


async def test_tool_calls_are_passed_through() -> None:
    call = ToolCall(id="c1", name="search_products", arguments={"query": "бойлер"})
    llm = FakeLLM([LLMResponse(content="", tool_calls=(call,))])

    result = await service(llm).answer("Бойлер?")

    assert result.tool_calls == (call,)


async def test_result_records_the_prompt_version() -> None:
    result = await service(FakeLLM()).answer("Здравей")

    assert result.prompt_version == PROMPT_VERSION
    assert PROMPT_VERSION == "2026-10-27.2"
