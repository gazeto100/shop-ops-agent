from pathlib import Path

import pytest

from shop_ops_agent.agent.chat import (
    ChatService,
    CitedProduct,
    cited_skus,
    detect_language,
    parse_figure,
)
from shop_ops_agent.agent.loop import MAX_ITERATIONS
from shop_ops_agent.agent.memory import SessionMemory
from shop_ops_agent.agent.prompts import (
    MAX_ITERATIONS_REPLY,
    PRICE_UNAVAILABLE_REPLY,
    PRODUCT_NOT_FOUND_REPLY,
    PROMPT_VERSION,
    REFUSAL_REPLY,
    SYSTEM_PROMPT,
    Language,
)
from shop_ops_agent.agent.tools import Catalog, build_registry
from shop_ops_agent.llm import LLM, LLMRefusalError, LLMResponse, Message, ToolCall, ToolSpec
from shop_ops_agent.rag.retriever import Hit
from tests.fakes import FakeLLM, FakeRetriever

SAMPLE = Path(__file__).parents[2] / "data" / "products.sample.csv"
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
CHECK_BOILER = ToolCall(id="c1", name="check_stock_and_price", arguments={"sku": "BOIL-80-A"})


class RefusingLLM:
    async def complete(
        self, messages: list[Message], tools: list[ToolSpec] | None = None
    ) -> LLMResponse:
        raise LLMRefusalError("general_harms")


def service(
    llm: LLM,
    hits: tuple[Hit, ...] = (BOILER, KETTLE),
    memory: SessionMemory | None = None,
    max_iterations: int = MAX_ITERATIONS,
) -> ChatService:
    retriever = FakeRetriever(hits)
    registry = build_registry(Catalog(SAMPLE), retriever)
    return ChatService(
        retriever, llm, registry, memory or SessionMemory(), max_iterations=max_iterations
    )


def answering(text: str) -> FakeLLM:
    return FakeLLM([LLMResponse(content=text)])


def after_checking_boiler(text: str) -> FakeLLM:
    """The model calls check_stock_and_price for the boiler, then answers with `text`."""
    return FakeLLM([LLMResponse(content="", tool_calls=(CHECK_BOILER,)), LLMResponse(content=text)])


# --- Prompt -----------------------------------------------------------------------------


async def test_retrieved_context_is_injected_into_the_prompt() -> None:
    llm = FakeLLM()

    await service(llm).answer("s1", "Какъв бойлер имате?")

    [(messages, tools)] = llm.calls
    system, user = messages
    assert system == Message(role="system", content=SYSTEM_PROMPT)
    assert user.role == "user"
    for hit in (BOILER, KETTLE):
        assert hit.citation() in user.content
        assert hit.text in user.content
    assert "<question>\nКакъв бойлер имате?\n</question>" in user.content
    # Customers are offered the customer tools only.
    assert [t["name"] for t in tools or []] == ["search_products", "check_stock_and_price"]


async def test_question_is_the_retrieval_query() -> None:
    retriever = FakeRetriever((BOILER,))
    chat = ChatService(
        retriever,
        FakeLLM(),
        build_registry(Catalog(SAMPLE), retriever),
        SessionMemory(),
        k=3,
    )

    await chat.answer("s1", "бойлер 80 литра")

    assert retriever.searches == [("бойлер 80 литра", 3)]


async def test_no_hits_says_so_in_the_prompt() -> None:
    llm = FakeLLM()

    await service(llm, hits=()).answer("s1", "Имате ли телевизори?")

    assert "(no matching products)" in llm.calls[0][0][1].content


def test_system_prompt_ties_prices_and_stock_to_tool_results() -> None:
    # Guards the non-negotiable rule against an accidental prompt edit.
    assert "Never state, estimate or guess a price" in SYSTEM_PROMPT
    assert "stock level" in SYSTEM_PROMPT
    assert "State only figures that a check_stock_and_price result in this turn" in SYSTEM_PROMPT
    assert "even if the user supplies a price" in SYSTEM_PROMPT
    assert "even if an earlier message" in SYSTEM_PROMPT  # memory must not serve old prices


def test_system_prompt_shows_a_concrete_citation_and_forbids_the_sku_word() -> None:
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
        ("Do you sell кани?", "bg", "Reply in Bulgarian."),
        ("imate li boiler 80 litra?", "bg", "Reply in Bulgarian."),
        ("do 35 kv.m klimatik?", "bg", "Reply in Bulgarian."),
        ("12000 BTU?", "bg", "Reply in Bulgarian."),
        ("80?", "bg", "Reply in Bulgarian."),
    ],
)
async def test_reply_language_follows_the_user(
    question: str, language: Language, instruction: str
) -> None:
    llm = FakeLLM()

    await service(llm).answer("s1", question)

    assert detect_language(question) == language
    assert instruction in llm.calls[0][0][1].content


# --- Tools through the chat turn --------------------------------------------------------


async def test_price_from_the_tool_result_reaches_the_user() -> None:
    answer = "[BOIL-80-A: Бойлер Tesy 80 л вертикален] струва 189.00 EUR (369,64 лв.), 4 бр."
    llm = after_checking_boiler(answer)

    result = await service(llm).answer("s1", "Колко струва бойлерът?")

    assert result.answer == answer
    assert result.guardrail is None
    assert result.tool_calls == (CHECK_BOILER,)
    assert result.citations == (CitedProduct("BOIL-80-A", "Бойлер Tesy 80 л вертикален"),)


@pytest.mark.parametrize(
    "answer",
    [
        "[BOIL-80-A: Бойлер] струва 199.00 EUR.",  # not what the tool said
        "[BOIL-80-A: Бойлер] струва около 370 лв.",  # rounded
        "[BOIL-80-A: Бойлер] — налични 9 бр.",  # wrong stock
        "[BOIL-80-A: Бойлер] струва 4 EUR.",  # review: equals the STOCK count, not a price
        "[BOIL-80-A: Бойлер] струва 189.00 лв.",  # review: the EUR amount labelled BGN
        "[BOIL-80-A: Бойлер] — налични 189 бр.",  # a price used as a stock count
        "[BOIL-80-A: Бойлер] струва $189.",  # no tool returns dollars
    ],
)
async def test_figures_not_in_the_tool_result_are_blocked(answer: str) -> None:
    result = await service(after_checking_boiler(answer)).answer("s1", "Колко струва бойлерът?")

    assert result.answer == PRICE_UNAVAILABLE_REPLY["bg"]
    assert result.guardrail == "price_or_stock"
    assert result.tool_calls == (CHECK_BOILER,)  # the call still happened


@pytest.mark.parametrize(
    "answer",
    [
        "[BOIL-80-A: Бойлер Tesy 80 л вертикален] струва 1 EUR.",
        "Цената е 189,00 лв., наличен е.",
        "It costs €189 and we have 12 in stock.",
    ],
)
async def test_figures_without_any_tool_call_are_blocked(answer: str) -> None:
    result = await service(answering(answer)).answer("s1", "Колко струва бойлерът?")

    assert result.answer == PRICE_UNAVAILABLE_REPLY["bg"]
    assert result.guardrail == "price_or_stock"


@pytest.mark.parametrize(
    "answer",
    [
        "Гаранция: 60 месеца, [BOIL-80-A: Бойлер Tesy 80 л вертикален], 2 kW, 80 л.",
        "Не мога да потвърдя текущата цена и наличност в момента.",
        "12000 BTU, до 35 кв.м [KET-17-S: Кана]",
    ],
)
async def test_specs_and_honest_refusals_are_not_mistaken_for_prices(answer: str) -> None:
    result = await service(answering(answer)).answer("s1", "Какво имате?")

    assert result.guardrail is None
    assert result.answer == answer


async def test_product_found_by_a_tool_is_a_verified_citation() -> None:
    # HEAT-2K-C was not retrieved up front, but check_stock_and_price returned it.
    check = ToolCall(id="c1", name="check_stock_and_price", arguments={"sku": "HEAT-2K-C"})
    llm = FakeLLM(
        [
            LLMResponse(content="", tool_calls=(check,)),
            LLMResponse(content="Имаме [HEAT-2K-C: Конвектор 2000 W с термостат]."),
        ]
    )

    result = await service(llm).answer("s1", "Имате ли конвектор?")

    assert result.guardrail is None
    assert result.citations == (CitedProduct("HEAT-2K-C", "Конвектор 2000 W с термостат"),)


async def test_product_neither_retrieved_nor_returned_by_a_tool_replaces_the_answer() -> None:
    result = await service(answering("Вижте [FAKE-9: Супер бойлер].")).answer("s1", "бойлер?")

    assert result.answer == PRODUCT_NOT_FOUND_REPLY["bg"]
    assert result.guardrail == "unverified_product"
    assert result.unverified_skus == ("FAKE-9",)


async def test_operator_tools_are_refused_in_customer_chat() -> None:
    sneaky = ToolCall(id="c1", name="compare_supplier_price", arguments={"threshold_pct": 10})
    llm = FakeLLM([LLMResponse(content="", tool_calls=(sneaky,)), LLMResponse(content="Не мога.")])

    await service(llm).answer("s1", "Какви са доставните цени?")

    tool_message = llm.calls[1][0][-1]
    assert tool_message.role == "tool"
    assert '"status":"error"' in tool_message.content
    assert "310" not in tool_message.content  # no supplier price reached the model


async def test_iteration_limit_ends_with_a_clear_message() -> None:
    llm = FakeLLM([LLMResponse(content="", tool_calls=(CHECK_BOILER,))])  # loops forever

    result = await service(llm, max_iterations=3).answer("s1", "How much is the boiler?")

    assert result.answer == MAX_ITERATIONS_REPLY["en"]
    assert result.guardrail == "max_iterations"
    assert len(llm.calls) == 3


# --- Citations --------------------------------------------------------------------------


async def test_live_answer_with_sku_word_still_yields_the_citation() -> None:
    live = "Имаме [sku: BOIL-80-A: Бойлер Tesy 80 л вертикален] с гаранция 60 месеца."

    result = await service(answering(live)).answer("s1", "Какъв бойлер имате?")

    assert result.answer == live
    assert result.citations == (CitedProduct("BOIL-80-A", BOILER.name),)


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


async def test_citations_keep_first_citation_order_without_duplicates() -> None:
    answer = "[KET-17-S: кана], после [BOIL-80-A: бойлер] и пак [KET-17-S: кана]"

    result = await service(answering(answer)).answer("s1", "Какво имате?")

    assert [c.sku for c in result.citations] == ["KET-17-S", "BOIL-80-A"]
    assert result.citations[0].name == KETTLE.name  # the catalog's name, not the model's


@pytest.mark.parametrize(
    ("text", "value"), [("189.00", "189.00"), ("369,64", "369.64"), ("1 299,50", "1299.50")]
)
def test_parse_figure_handles_bulgarian_and_english_separators(text: str, value: str) -> None:
    assert str(parse_figure(text)) == value
    assert parse_figure("1,299.50") == parse_figure("1 299,50")


# --- Session memory ---------------------------------------------------------------------


async def test_previous_text_turns_are_in_the_next_prompt() -> None:
    memory = SessionMemory()
    first = after_checking_boiler("[BOIL-80-A: Бойлер] струва 189.00 EUR.")
    turn1 = await service(first, memory=memory).answer(None, "Колко струва бойлерът?")
    second = FakeLLM()

    await service(second, memory=memory).answer(turn1.session_id, "А гаранцията?")

    messages = second.calls[0][0]
    assert messages[1:3] == [
        Message(role="user", content="Колко струва бойлерът?"),
        Message(role="assistant", content="[BOIL-80-A: Бойлер] струва 189.00 EUR."),
    ]
    # Only text turns are remembered: no tool calls, results or thinking blocks.
    assert all(m.role in ("system", "user", "assistant") and not m.tool_calls for m in messages)


async def test_sessions_are_isolated() -> None:
    memory = SessionMemory()
    await service(answering("Здравейте!"), memory=memory).answer("alice", "Здравей")
    llm = FakeLLM()

    await service(llm, memory=memory).answer("bob", "Здравей")

    assert len(llm.calls[0][0]) == 2  # system + bob's question only


async def test_guarded_reply_is_what_memory_keeps() -> None:
    memory = SessionMemory()

    result = await service(answering("струва 1 EUR"), memory=memory).answer(None, "Колко?")

    assert memory.history(result.session_id)[1].content == PRICE_UNAVAILABLE_REPLY["bg"]


async def test_follow_up_may_cite_a_product_from_an_earlier_answer() -> None:
    # Review finding: the follow-up's retrieval no longer returns the boiler, but the
    # previous (verified) answer cited it, so citing it again is not an invented product.
    memory = SessionMemory()
    turn1 = await service(
        answering("Имаме [BOIL-80-A: Бойлер Tesy 80 л вертикален]."), memory=memory
    ).answer(None, "Имате ли бойлер?")

    follow_up = await service(
        answering("[BOIL-80-A: Бойлер Tesy 80 л вертикален] е с гаранция 60 месеца."),
        hits=(),
        memory=memory,
    ).answer(turn1.session_id, "А гаранцията му?")

    assert follow_up.guardrail is None
    assert follow_up.citations == (CitedProduct("BOIL-80-A", "Бойлер Tesy 80 л вертикален"),)


async def test_products_from_another_session_are_not_known() -> None:
    memory = SessionMemory()
    await service(
        answering("Имаме [BOIL-80-A: Бойлер Tesy 80 л вертикален]."), memory=memory
    ).answer(None, "Имате ли бойлер?")

    other = await service(
        answering("[BOIL-80-A: Бойлер Tesy 80 л вертикален]."), hits=(), memory=memory
    ).answer(None, "Бойлер?")

    assert other.guardrail == "unverified_product"


# --- Refusals / metadata ----------------------------------------------------------------


@pytest.mark.parametrize(("question", "language"), [("Как да…?", "bg"), ("How do I…?", "en")])
async def test_refusal_returns_a_polite_reply_in_the_users_language(
    question: str, language: Language
) -> None:
    result = await service(RefusingLLM()).answer("s1", question)

    assert result.answer == REFUSAL_REPLY[language]
    assert result.citations == ()


async def test_result_records_the_prompt_version() -> None:
    result = await service(FakeLLM()).answer("s1", "Здравей")

    assert result.prompt_version == PROMPT_VERSION
    assert PROMPT_VERSION == "2026-10-29.1"
