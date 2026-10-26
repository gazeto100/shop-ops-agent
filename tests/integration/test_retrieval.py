"""Retrieval quality with the real embedding model (no LLM).

Downloads the model on first run (~470 MB, then cached). Run with `make test-integration`.
"""

from pathlib import Path

import pytest

from shop_ops_agent.config import Settings
from shop_ops_agent.embeddings import SentenceTransformerEmbedder
from shop_ops_agent.ingest.index import open_collection, sync_products
from shop_ops_agent.ingest.products import load_products
from shop_ops_agent.rag.retriever import Retriever

SAMPLE = Path(__file__).parents[2] / "data" / "products.sample.csv"
KETTLES = {"KET-17-S", "KET-10-P"}


@pytest.fixture(scope="module")
def retriever(tmp_path_factory: pytest.TempPathFactory) -> Retriever:
    # The configured default model, without reading env vars or the dev's .env.
    model = Settings.model_construct().embedding_model
    embedder = SentenceTransformerEmbedder(model)
    collection = open_collection(tmp_path_factory.mktemp("index") / "chroma", model)
    sync_products(collection, load_products(SAMPLE), embedder)
    return Retriever(collection, embedder)


def test_bulgarian_boiler_query_ranks_boil_80_a_first(retriever: Retriever) -> None:
    assert retriever.search("бойлер 80 литра")[0].sku == "BOIL-80-A"


def test_english_kettle_query_ranks_a_kettle_first(retriever: Retriever) -> None:
    hits = retriever.search("kettle")

    assert hits[0].sku in KETTLES
    # Both kettles beat everything else, not just one of them by luck.
    assert {h.sku for h in hits[:2]} == KETTLES
