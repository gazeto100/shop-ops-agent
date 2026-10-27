"""Snapshot of what the model is told about the tools (names, descriptions, JSON Schemas).

Tool descriptions are prompt text: changing one changes model behaviour, so it must be a
deliberate, reviewed diff. If this test fails because you meant to change a tool:

    uv run python -m tests.unit.test_tool_specs     # rewrites the snapshot

then bump PROMPT_VERSION in agent/prompts.py and add/adjust a golden-set case.
"""

import json
from pathlib import Path

from shop_ops_agent.agent.tools import Catalog, build_registry
from tests.fakes import FakeRetriever

SNAPSHOT = Path(__file__).with_name("tool_specs.snapshot.json")


def current_specs() -> str:
    # Specs don't read the catalog, so its path is irrelevant here.
    registry = build_registry(Catalog(Path("unused.csv")), FakeRetriever())
    return json.dumps(registry.specs("operator"), ensure_ascii=False, indent=2) + "\n"


def test_tool_specs_match_the_reviewed_snapshot() -> None:
    assert current_specs() == SNAPSHOT.read_text(encoding="utf-8"), (
        "Tool specs changed. If intended, run `uv run python -m tests.unit.test_tool_specs`, "
        "bump PROMPT_VERSION and add a golden-set case."
    )


if __name__ == "__main__":
    SNAPSHOT.write_text(current_specs(), encoding="utf-8", newline="\n")
    print(f"wrote {SNAPSHOT}")
