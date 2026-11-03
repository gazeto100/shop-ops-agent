"""Per-case metrics. Deterministic string checks; the LLM-as-judge score is #9."""

import re
from collections.abc import Sequence
from dataclasses import dataclass

from shop_ops_agent.evals.golden import GoldenCase

# "59,00" and "59.00" state the same figure (Bulgarian vs English decimal separator).
_DECIMAL_COMMA = re.compile(r"(?<=\d),(?=\d)")


def _normalized(text: str) -> str:
    return _DECIMAL_COMMA.sub(".", text.casefold())


@dataclass(frozen=True)
class CaseScore:
    # Share of the required tools that were called (None: tool choice not scored). With
    # no tools required, 1.0 only if none was called.
    tool_recall: float | None
    missing_tools: tuple[str, ...]
    extra_tools: tuple[str, ...]  # reported, not penalised (except when [] was expected)
    facts_hit_rate: float | None  # None: the case lists no expected facts
    missing_facts: tuple[str, ...]
    violations: tuple[str, ...]  # must_not_contain strings found in the answer
    citation_present: bool

    def passed(self, case: GoldenCase) -> bool:
        return (
            self.tool_recall in (None, 1.0)
            and self.facts_hit_rate in (None, 1.0)
            and not self.violations
            and (self.citation_present or not case.must_cite)
        )


def score_case(
    case: GoldenCase, answer: str, tools_called: Sequence[str], has_citation: bool
) -> CaseScore:
    called = set(tools_called)
    tool_recall: float | None
    missing: tuple[str, ...]
    if case.expected_tools is None:
        tool_recall, missing = None, ()
    elif not case.expected_tools:
        tool_recall, missing = (0.0 if called else 1.0), ()
    else:
        expected = set(case.expected_tools)
        missing = tuple(sorted(expected - called))
        tool_recall = len(expected & called) / len(expected)
    extra = tuple(sorted(called - set(case.expected_tools or ())))

    # Case-insensitive substring checks (casefold also handles Cyrillic), with numbers
    # compared regardless of decimal separator on both sides.
    text = _normalized(answer)
    missing_facts = tuple(f for f in case.expected_facts if _normalized(f) not in text)
    facts_hit_rate = (
        1 - len(missing_facts) / len(case.expected_facts) if case.expected_facts else None
    )
    violations = tuple(s for s in case.must_not_contain if _normalized(s) in text)
    return CaseScore(
        tool_recall=tool_recall,
        missing_tools=missing,
        extra_tools=extra,
        facts_hit_rate=facts_hit_rate,
        missing_facts=missing_facts,
        violations=violations,
        citation_present=has_citation,
    )
