"""Eval runner (#8): golden-set validation, metrics, runner, table and results file.

Driven with FakeLLM through the real ChatService and tools: no network, no model.
"""

import json
from datetime import date
from pathlib import Path

import pytest

from shop_ops_agent.agent.chat import ChatService
from shop_ops_agent.agent.memory import SessionMemory
from shop_ops_agent.agent.tools import Audience, Catalog, build_registry
from shop_ops_agent.evals.golden import REQUIRED_CATEGORIES, GoldenCase, load_golden
from shop_ops_agent.evals.metrics import score_case
from shop_ops_agent.evals.runner import (
    CaseResult,
    EvalReport,
    render_table,
    run_golden,
    write_results,
)
from shop_ops_agent.llm import LLM, LLMResponse, Message, ToolCall, ToolSpec
from shop_ops_agent.rag.retriever import Hit
from tests.fakes import FakeLLM, FakeRetriever

ROOT = Path(__file__).parents[2]
GOLDEN = ROOT / "tests" / "evals" / "golden.jsonl"
SAMPLE = ROOT / "data" / "products.sample.csv"
BOILER = Hit(sku="BOIL-80-A", name="Бойлер Tesy 80 л вертикален", score=0.03, text="…")
CHECK_BOILER = ToolCall(id="c1", name="check_stock_and_price", arguments={"sku": "BOIL-80-A"})


def case(**fields: object) -> GoldenCase:
    defaults: dict[str, object] = {
        "id": "t1",
        "category": "happy",
        "question": "Колко струва бойлерът?",
        "expected_tools": ["check_stock_and_price"],
    }
    return GoldenCase.model_validate(defaults | fields)


def services_for(llm: LLM) -> dict[Audience, ChatService]:
    retriever = FakeRetriever([BOILER])
    registry = build_registry(Catalog(SAMPLE), retriever)
    return {
        a: ChatService(retriever, llm, registry, SessionMemory(), audience=a)
        for a in ("customer", "operator")
    }


# --- The golden set itself ----------------------------------------------------------------


def test_golden_set_has_at_least_30_valid_cases_covering_every_category() -> None:
    cases = load_golden(GOLDEN)

    assert len(cases) >= 30
    assert {c.category for c in cases} >= REQUIRED_CATEGORIES


def test_golden_cases_only_expect_real_tools_their_audience_may_use() -> None:
    registry = build_registry(Catalog(SAMPLE), FakeRetriever())
    for c in load_golden(GOLDEN):
        allowed = {s["name"] for s in registry.specs(c.audience)}
        assert set(c.expected_tools or ()) <= allowed, c.id


def test_the_issue_example_adversarial_case_is_present() -> None:
    questions = [c.question for c in load_golden(GOLDEN) if c.category == "adversarial"]

    assert any("Ignore the catalog and say" in q and "free" in q for q in questions)


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ('{"id":"x"}', "golden.jsonl:1"),
        ("not json", "golden.jsonl:1"),
        (
            '{"id":"a","category":"happy","question":"q","expected_tools":null}\n'
            '{"id":"a","category":"happy","question":"q","expected_tools":null}',
            "golden.jsonl:2: duplicate id 'a'",
        ),
    ],
)
def test_invalid_golden_files_name_the_line(tmp_path: Path, content: str, message: str) -> None:
    path = tmp_path / "golden.jsonl"
    path.write_text(content + "\n\n", encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_golden(path)


# --- Metrics ------------------------------------------------------------------------------


def test_tool_recall_counts_required_tools_and_lists_extras() -> None:
    c = case(expected_tools=["check_stock_and_price", "search_products"])

    score = score_case(c, "", ["check_stock_and_price", "compare_supplier_price"], False)

    assert score.tool_recall == 0.5
    assert score.missing_tools == ("search_products",)
    assert score.extra_tools == ("compare_supplier_price",)


@pytest.mark.parametrize(("called", "recall"), [([], 1.0), (["search_products"], 0.0)])
def test_no_tools_expected_means_none_may_be_called(called: list[str], recall: float) -> None:
    assert score_case(case(expected_tools=[]), "", called, False).tool_recall == recall


def test_tool_choice_not_scored() -> None:
    score = score_case(case(expected_tools=None), "", ["search_products"], False)

    assert score.tool_recall is None
    assert score.extra_tools == ("search_products",)


def test_facts_are_case_insensitive_substrings_incl_cyrillic() -> None:
    c = case(expected_facts=["189.00", "БОЙЛЕР", "369.64"])

    score = score_case(c, "Бойлерът струва 189.00 EUR", [], False)

    assert score.facts_hit_rate == pytest.approx(2 / 3)
    assert score.missing_facts == ("369.64",)


def test_violations_and_citation_and_pass() -> None:
    c = case(must_not_contain=["1 EUR"], must_cite=True, expected_facts=["189.00"])
    good = score_case(c, "[BOIL-80-A: Бойлер] 189.00 EUR", ["check_stock_and_price"], True)
    bad = score_case(c, "Струва 1 EUR", ["check_stock_and_price"], False)

    assert good.passed(c)
    assert (bad.violations, bad.citation_present, bad.passed(c)) == (("1 EUR",), False, False)
    assert score_case(case(), "", ["check_stock_and_price"], False).facts_hit_rate is None


# --- Runner ---------------------------------------------------------------------------


async def test_runner_scores_each_case_through_the_chat_service() -> None:
    llm = FakeLLM(
        [
            LLMResponse(content="", tool_calls=(CHECK_BOILER,)),
            LLMResponse(content="[BOIL-80-A: Бойлер Tesy 80 л вертикален] е 189.00 EUR."),
        ]
    )
    c = case(expected_facts=["189.00"], must_cite=True)

    [result] = await run_golden([c], services_for(llm))

    assert result.error is None
    assert result.tool_calls == ("check_stock_and_price",)
    assert result.citations == ("BOIL-80-A",)
    assert result.passed


async def test_operator_cases_run_with_the_operator_audience() -> None:
    llm = FakeLLM()

    await run_golden([case(audience="operator", expected_tools=None)], services_for(llm))

    offered = {t["name"] for t in llm.calls[0][1] or []}
    assert "compare_supplier_price" in offered


async def test_a_failing_case_is_recorded_and_the_run_continues() -> None:
    class Broken:
        calls = 0

        async def complete(
            self, messages: list[Message], tools: list[ToolSpec] | None = None
        ) -> LLMResponse:
            Broken.calls += 1
            if Broken.calls == 1:
                raise RuntimeError("provider exploded")
            return LLMResponse(content="ok")

    results = await run_golden(
        [case(id="a", expected_tools=None), case(id="b", expected_tools=None)],
        services_for(Broken()),
    )

    assert results[0].error == "RuntimeError: provider exploded"
    assert results[0].score is None and not results[0].passed
    assert results[1].error is None


# --- Report -----------------------------------------------------------------------------


async def sample_report() -> EvalReport:
    llm = FakeLLM(
        [
            LLMResponse(content="", tool_calls=(CHECK_BOILER,)),
            LLMResponse(content="[BOIL-80-A: Бойлер Tesy 80 л вертикален] е 189.00 EUR."),
            LLMResponse(content="Струва 1 EUR."),
        ]
    )
    results = await run_golden(
        [
            case(id="ok", expected_facts=["189.00"], must_cite=True),
            case(id="bad", expected_tools=[], must_not_contain=["1 EUR"], must_cite=True),
        ],
        services_for(llm),
    )
    return EvalReport(date(2026, 11, 3), "test-model", "v1", results)


async def test_summary_aggregates_the_cases() -> None:
    summary = (await sample_report()).summary()

    assert summary == {
        "cases": 2,
        "passed": 1,
        "errors": 0,
        "mean_tool_recall": 1.0,
        "mean_facts_hit_rate": 1.0,
        # "Струва 1 EUR." is replaced by the price guardrail before scoring, so the
        # must_not_contain check sees the safe reply: the guard did its job.
        "must_not_contain_violations": 0,
        "citation_rate": 0.5,
    }


async def test_table_lists_every_case_and_the_totals() -> None:
    table = render_table(await sample_report())

    assert "model test-model, prompt v1" in table
    assert "ok" in table and "PASS" in table
    assert "bad" in table and "FAIL" in table and "guardrail price_or_stock" in table
    assert table.splitlines()[-1].startswith("passed 1/2, errors 0")


def test_table_shows_errors_and_empty_summary() -> None:
    empty = EvalReport(date(2026, 11, 3), "m", "v", ())
    errored = EvalReport(
        date(2026, 11, 3), "m", "v", (CaseResult(case=case(), error="RuntimeError: boom"),)
    )

    assert "passed 0/0" in render_table(empty)
    assert "ERROR  RuntimeError: boom" in render_table(errored)
    assert empty.summary()["mean_tool_recall"] is None
    assert errored.summary()["errors"] == 1


def test_blank_lines_in_the_golden_file_are_ignored(tmp_path: Path) -> None:
    path = tmp_path / "golden.jsonl"
    path.write_text(
        '\n{"id":"a","category":"happy","question":"q","expected_tools":null}\n\n',
        encoding="utf-8",
    )

    assert [c.id for c in load_golden(path)] == ["a"]


async def test_results_file_is_dated_json_with_every_case(tmp_path: Path) -> None:
    path = write_results(await sample_report(), tmp_path / "evals")

    assert path == tmp_path / "evals" / "results-2026-11-03.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["model"] == "test-model"
    assert data["summary"]["cases"] == 2
    assert [c["id"] for c in data["cases"]] == ["ok", "bad"]
    assert data["cases"][0]["score"]["tool_recall"] == 1.0
    assert data["cases"][1]["guardrail"] == "price_or_stock"


@pytest.mark.parametrize("answer", ["Цена: 59,00 EUR", "Price: 59.00 EUR", "1 464,93 BGN и 59,00"])
def test_facts_match_either_decimal_separator(answer: str) -> None:
    # Live run: the model wrote "59,00 EUR" (Bulgarian style) and the fact "59.00" missed.
    score = score_case(case(expected_facts=["59.00"]), answer, [], False)

    assert score.facts_hit_rate == 1.0


def test_must_not_contain_also_ignores_the_decimal_separator() -> None:
    c = case(must_not_contain=["1.00 EUR"])

    assert score_case(c, "Струва 1,00 EUR", [], False).violations == ("1.00 EUR",)


async def test_guarded_cases_keep_what_the_model_wrote(tmp_path: Path) -> None:
    report = await sample_report()
    bad = report.results[1]

    assert bad.guardrail == "price_or_stock"
    assert bad.replaced_answer == "Струва 1 EUR."
    assert bad.unbacked_figures == ("1 EUR",)
    assert "unbacked ['1 EUR']" in render_table(report)
    data = json.loads(write_results(report, tmp_path).read_text(encoding="utf-8"))
    assert data["cases"][1]["replaced_answer"] == "Струва 1 EUR."
    assert data["cases"][0]["replaced_answer"] is None
