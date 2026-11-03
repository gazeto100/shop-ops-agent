"""Per-case metrics. Deterministic string checks; the LLM-as-judge score is #9."""

import re
from collections.abc import Sequence
from dataclasses import dataclass

from shop_ops_agent.agent.prompts import Language
from shop_ops_agent.evals.golden import GoldenCase

# "59,00" and "59.00" state the same figure (Bulgarian vs English decimal separator).
_DECIMAL_COMMA = re.compile(r"(?<=\d),(?=\d)")
_CITATION = re.compile(r"\[[^\[\]]*\]")
_CYRILLIC_WORD = re.compile(r"[Ѐ-ӿ]+")
# Ordinary English words: lowercase, 3+ letters. Leaves out what Bulgarian answers are full
# of too: units and codes (EUR, BGN, BTU, kW, W) and capitalised brands (Tesy, Eldom).
_ENGLISH_WORD = re.compile(r"\b[a-z]{3,}\b")


def _normalized(text: str) -> str:
    return _DECIMAL_COMMA.sub(".", text.casefold())


def _contains(text: str, needle: str) -> bool:
    """Case-insensitive substring match where numbers must match whole numbers: a needle
    that starts or ends with a digit may not continue a longer number in the text, so
    "0 EUR" doesn't match "24.90 EUR" and "36" doesn't match "360". Word edges stay open
    on purpose: Bulgarian inflects ("каталог" must match "каталога") and facts like
    "KET-" are prefixes."""
    n = _normalized(needle)
    lead = r"(?<![\d.,])" if n[:1].isdigit() else ""
    trail = r"(?!\d)" if n[-1:].isdigit() else ""
    return re.search(lead + re.escape(n) + trail, _normalized(text)) is not None


def answer_language(answer: str) -> Language:
    """English if the answer has more ordinary English words than Cyrillic words,
    ignoring [CODE: name] citations (Bulgarian product names would otherwise make every
    English answer look Bulgarian). Ties, including no words at all, count as Bulgarian,
    the shop's default."""
    text = _CITATION.sub(" ", answer)
    english = len(_ENGLISH_WORD.findall(text))
    return "en" if english > len(_CYRILLIC_WORD.findall(text)) else "bg"


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
    language_ok: bool  # answered in the case's `lang`

    def passed(self, case: GoldenCase) -> bool:
        return (
            self.tool_recall in (None, 1.0)
            and self.facts_hit_rate in (None, 1.0)
            and not self.violations
            and (self.citation_present or not case.must_cite)
            and self.language_ok
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

    missing_facts = tuple(f for f in case.expected_facts if not _contains(answer, f))
    facts_hit_rate = (
        1 - len(missing_facts) / len(case.expected_facts) if case.expected_facts else None
    )
    violations = tuple(s for s in case.must_not_contain if _contains(answer, s))
    return CaseScore(
        tool_recall=tool_recall,
        missing_tools=missing,
        extra_tools=extra,
        facts_hit_rate=facts_hit_rate,
        missing_facts=missing_facts,
        violations=violations,
        citation_present=has_citation,
        language_ok=answer_language(answer) == case.lang,
    )
