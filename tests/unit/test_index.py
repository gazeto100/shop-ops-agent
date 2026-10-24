from pathlib import Path

import numpy as np
import pytest

from shop_ops_agent.ingest.chunks import chunk_text
from shop_ops_agent.ingest.index import SyncResult, open_collection, sync_products
from shop_ops_agent.ingest.products import load_products
from tests.fakes import FakeEmbedder

SAMPLE = Path(__file__).parents[2] / "data" / "products.sample.csv"


def test_sync_indexes_one_chunk_per_product(tmp_path: Path) -> None:
    products = load_products(SAMPLE)
    collection = open_collection(tmp_path / "chroma")

    result = sync_products(collection, products, FakeEmbedder())

    assert result == SyncResult(upserted=7, deleted=0)
    stored = collection.get(ids=["BOIL-80-A"], include=["documents", "metadatas", "embeddings"])
    boiler = products[0]
    assert stored["documents"] == [chunk_text(boiler)]
    assert stored["metadatas"] is not None
    assert stored["metadatas"][0]["name"] == boiler.name
    assert stored["embeddings"] is not None
    np.testing.assert_allclose(stored["embeddings"][0], FakeEmbedder.vector(chunk_text(boiler)))


def test_reingest_is_idempotent(tmp_path: Path) -> None:
    products = load_products(SAMPLE)
    collection = open_collection(tmp_path / "chroma")
    sync_products(collection, products, FakeEmbedder())
    before = collection.get()

    result = sync_products(open_collection(tmp_path / "chroma"), products, FakeEmbedder())

    assert result == SyncResult(upserted=7, deleted=0)
    after = collection.get()
    assert sorted(after["ids"]) == sorted(before["ids"])
    assert after["documents"] == before["documents"]


def test_skus_missing_from_feed_are_removed(tmp_path: Path) -> None:
    products = load_products(SAMPLE)
    collection = open_collection(tmp_path / "chroma")
    sync_products(collection, products, FakeEmbedder())

    result = sync_products(collection, products[:5], FakeEmbedder())

    assert result == SyncResult(upserted=5, deleted=2)
    assert sorted(collection.get()["ids"]) == sorted(p.sku for p in products[:5])


# chromadb 1.5 stores embedding_function=None as {"type": "legacy"} and warns on read-back.
@pytest.mark.filterwarnings("ignore:legacy embedding function config:DeprecationWarning")
def test_collection_uses_cosine_distance(tmp_path: Path) -> None:
    hnsw = open_collection(tmp_path / "chroma").configuration.get("hnsw")

    assert hnsw is not None
    assert hnsw.get("space") == "cosine"
