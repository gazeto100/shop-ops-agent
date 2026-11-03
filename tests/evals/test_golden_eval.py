"""Live golden-set eval (#8): the real LLM provider, embedding model and tools.

Skipped unless RUN_LIVE_EVALS=1. Prints the score table and writes
evals/results-<date>.json. It fails only when a case errors (provider down, bad key);
quality thresholds become a CI gate in #9.
"""

from datetime import date
from pathlib import Path

import pytest

from shop_ops_agent.agent.chat import ChatService
from shop_ops_agent.agent.memory import SessionMemory
from shop_ops_agent.agent.prompts import PROMPT_VERSION
from shop_ops_agent.agent.tools import Audience, Catalog, build_registry
from shop_ops_agent.config import get_settings
from shop_ops_agent.embeddings import SentenceTransformerEmbedder
from shop_ops_agent.evals.golden import load_golden
from shop_ops_agent.evals.runner import EvalReport, render_table, run_golden, write_results
from shop_ops_agent.ingest.index import open_collection, sync_products
from shop_ops_agent.ingest.products import load_products
from shop_ops_agent.llm import create_llm
from shop_ops_agent.rag.retriever import Retriever

ROOT = Path(__file__).parents[2]
GOLDEN = Path(__file__).with_name("golden.jsonl")
SAMPLE = ROOT / "data" / "products.sample.csv"  # the golden set is written against it
RESULTS_DIR = ROOT / "evals"

pytestmark = pytest.mark.skipif(
    not get_settings().run_live_evals, reason="live evals call a real LLM: set RUN_LIVE_EVALS=1"
)


@pytest.fixture(scope="module")
def services(tmp_path_factory: pytest.TempPathFactory) -> dict[Audience, ChatService]:
    settings = get_settings()
    llm = create_llm(settings)  # fails clearly if the provider key is missing
    embedder = SentenceTransformerEmbedder(settings.embedding_model)
    collection = open_collection(
        tmp_path_factory.mktemp("eval") / "chroma", settings.embedding_model
    )
    sync_products(collection, load_products(SAMPLE), embedder)
    retriever = Retriever(collection, embedder)
    registry = build_registry(Catalog(SAMPLE), retriever)
    return {
        audience: ChatService(retriever, llm, registry, SessionMemory(), audience=audience)
        for audience in ("customer", "operator")
    }


async def test_golden_set(
    services: dict[Audience, ChatService], capsys: pytest.CaptureFixture[str]
) -> None:
    results = await run_golden(load_golden(GOLDEN), services)
    report = EvalReport(
        run_date=date.today(),
        model=get_settings().resolved_llm_model,
        prompt_version=PROMPT_VERSION,
        results=results,
    )
    path = write_results(report, RESULTS_DIR)

    with capsys.disabled():  # show the table even though pytest captures output
        print("\n" + render_table(report) + f"\nresults written to {path}\n")

    errors = [f"{r.case.id}: {r.error}" for r in results if r.error]
    assert not errors, "cases errored (infrastructure, not quality):\n" + "\n".join(errors)
