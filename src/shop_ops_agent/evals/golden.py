"""The golden set: one JSON case per line in tests/evals/golden.jsonl."""

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from shop_ops_agent.agent.prompts import Language
from shop_ops_agent.agent.tools import Audience

type Category = Literal[
    "happy", "not_found", "out_of_stock", "multi_tool", "english", "adversarial", "operator"
]

# The categories issue #8 requires the golden set to cover.
REQUIRED_CATEGORIES: frozenset[Category] = frozenset(
    {"happy", "not_found", "out_of_stock", "multi_tool", "english", "adversarial"}
)


class GoldenCase(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(min_length=1)
    category: Category
    question: str = Field(min_length=1)
    # Tools the answer REQUIRES. [] = no tool should be called at all; None = tool choice
    # is not scored (the catalog excerpt alone may be enough).
    expected_tools: list[str] | None
    expected_facts: list[str] = Field(default_factory=list)  # substrings the answer needs
    must_not_contain: list[str] = Field(default_factory=list)  # hallucination guard
    must_cite: bool = False
    lang: Language = "bg"
    audience: Audience = "customer"


def load_golden(path: Path) -> list[GoldenCase]:
    """Parse and validate the golden set; errors name the line, ids must be unique."""
    cases: list[GoldenCase] = []
    seen: set[str] = set()
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            case = GoldenCase.model_validate(json.loads(line))
        except (json.JSONDecodeError, ValidationError) as e:
            raise ValueError(f"{path.name}:{number}: {e}") from e
        if case.id in seen:
            raise ValueError(f"{path.name}:{number}: duplicate id {case.id!r}")
        seen.add(case.id)
        cases.append(case)
    return cases
