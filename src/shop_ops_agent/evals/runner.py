"""Run the golden set through the chat service, score it, print and save a report.

The runner takes ready-made `ChatService`s (one per audience), so unit tests drive it
with FakeLLM and the live eval (tests/evals) with the real provider.
"""

import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from statistics import fmean
from typing import Any

from shop_ops_agent.agent.chat import ChatService
from shop_ops_agent.agent.tools import Audience
from shop_ops_agent.evals.golden import GoldenCase
from shop_ops_agent.evals.metrics import CaseScore, score_case

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class CaseResult:
    case: GoldenCase
    answer: str = ""
    tool_calls: tuple[str, ...] = ()
    citations: tuple[str, ...] = ()
    guardrail: str | None = None
    # What the model wrote before a guardrail replaced it, and the figures it couldn't
    # back: makes a guarded failure diagnosable from the results file alone.
    replaced_answer: str | None = None
    unbacked_figures: tuple[str, ...] = ()
    score: CaseScore | None = None  # None when the case errored
    error: str | None = None

    @property
    def passed(self) -> bool:
        return self.score is not None and self.score.passed(self.case)


@dataclass(frozen=True)
class EvalReport:
    run_date: date
    model: str
    prompt_version: str
    results: tuple[CaseResult, ...]

    def summary(self) -> dict[str, Any]:
        scores = [r.score for r in self.results if r.score is not None]
        recalls = [s.tool_recall for s in scores if s.tool_recall is not None]
        fact_rates = [s.facts_hit_rate for s in scores if s.facts_hit_rate is not None]
        must_cite = [r.score for r in self.results if r.case.must_cite and r.score is not None]
        return {
            "cases": len(self.results),
            "passed": sum(r.passed for r in self.results),
            "errors": sum(r.error is not None for r in self.results),
            "mean_tool_recall": _mean(recalls),
            "mean_facts_hit_rate": _mean(fact_rates),
            "must_not_contain_violations": sum(len(s.violations) for s in scores),
            "citation_rate": _mean([float(s.citation_present) for s in must_cite]),
            "language_ok_rate": _mean([float(s.language_ok) for s in scores]),
        }


def _mean(values: Sequence[float]) -> float | None:
    return round(fmean(values), 3) if values else None


async def run_golden(
    cases: Sequence[GoldenCase], services: Mapping[Audience, ChatService]
) -> tuple[CaseResult, ...]:
    """Each case in a fresh session, one after another (gentle on rate limits)."""
    results = []
    for case in cases:
        try:
            chat = await services[case.audience].answer(None, case.question)
        except Exception as e:  # one failing case must not lose the whole run
            log.exception("eval case %s failed", case.id)
            results.append(CaseResult(case=case, error=f"{type(e).__name__}: {e}"))
            continue
        tools = tuple(c.name for c in chat.tool_calls)
        results.append(
            CaseResult(
                case=case,
                answer=chat.answer,
                tool_calls=tools,
                citations=tuple(c.sku for c in chat.citations),
                guardrail=chat.guardrail,
                replaced_answer=chat.replaced_answer,
                unbacked_figures=chat.unbacked_figures,
                score=score_case(case, chat.answer, tools, bool(chat.citations)),
            )
        )
    return tuple(results)


def _pct(value: float | None) -> str:
    return "  -  " if value is None else f"{value:5.0%}"


def render_table(report: EvalReport) -> str:
    header = (
        f"{'id':<6} {'category':<12} {'tools':>5} {'facts':>5} {'viol':>4} {'cite':>4} "
        f"{'lang':>4}  result"
    )
    lines = [
        f"Golden-set eval {report.run_date} - model {report.model}, prompt {report.prompt_version}",
        header,
        "-" * len(header),
    ]
    for r in report.results:
        if r.score is None:
            lines.append(f"{r.case.id:<6} {r.case.category:<12} ERROR  {r.error}")
            continue
        s = r.score
        cite = ("yes" if s.citation_present else "no") if r.case.must_cite else "-"
        notes = [f"missing tools {list(s.missing_tools)}"] if s.missing_tools else []
        notes += [f"extra {list(s.extra_tools)}"] if s.extra_tools else []
        notes += [f"missing facts {list(s.missing_facts)}"] if s.missing_facts else []
        notes += [f"VIOLATES {list(s.violations)}"] if s.violations else []
        notes += [f"guardrail {r.guardrail}"] if r.guardrail else []
        notes += [f"unbacked {list(r.unbacked_figures)}"] if r.unbacked_figures else []
        notes += [f"not in {r.case.lang}"] if not s.language_ok else []
        lang = "ok" if s.language_ok else "no"
        lines.append(
            f"{r.case.id:<6} {r.case.category:<12} {_pct(s.tool_recall)} "
            f"{_pct(s.facts_hit_rate)} {len(s.violations):>4} {cite:>4} {lang:>4}  "
            f"{'PASS' if r.passed else 'FAIL'}  {'; '.join(notes)}".rstrip()
        )
    summary = report.summary()
    lines += [
        "-" * len(header),
        f"passed {summary['passed']}/{summary['cases']}, errors {summary['errors']}, "
        f"tool recall {_pct(summary['mean_tool_recall']).strip()}, "
        f"facts {_pct(summary['mean_facts_hit_rate']).strip()}, "
        f"violations {summary['must_not_contain_violations']}, "
        f"citations {_pct(summary['citation_rate']).strip()}, "
        f"language {_pct(summary['language_ok_rate']).strip()}",
    ]
    return "\n".join(lines)


def write_results(report: EvalReport, out_dir: Path) -> Path:
    """Write `<out_dir>/results-<YYYY-MM-DD>.json`. Later runs that day get `-2`, `-3`, …
    instead of replacing it, so a before/after comparison (e.g. a prompt change) keeps
    both runs."""
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"results-{report.run_date.isoformat()}"
    path, n = out_dir / f"{stem}.json", 2
    while path.exists():
        path, n = out_dir / f"{stem}-{n}.json", n + 1
    payload = {
        "date": report.run_date.isoformat(),
        "model": report.model,
        "prompt_version": report.prompt_version,
        "summary": report.summary(),
        "cases": [
            {
                **r.case.model_dump(),
                "answer": r.answer,
                "tool_calls": list(r.tool_calls),
                "citations": list(r.citations),
                "guardrail": r.guardrail,
                "replaced_answer": r.replaced_answer,
                "unbacked_figures": list(r.unbacked_figures),
                "passed": r.passed,
                "error": r.error,
                "score": asdict(r.score) if r.score else None,
            }
            for r in report.results
        ],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path
