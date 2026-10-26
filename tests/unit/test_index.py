from pathlib import Path

import numpy as np
import pytest

from shop_ops_agent.ingest.chunks import chunk_text
from shop_ops_agent.ingest.index import (
    IndexModelMismatchError,
    SyncResult,
    open_collection,
    sync_products,
)
from shop_ops_agent.ingest.products import load_products
from tests.fakes import FakeEmbedder

SAMPLE = Path(__file__).parents[2] / "data" / "products.sample.csv"
MODEL = "test/fake-model"


def test_sync_indexes_one_chunk_per_product(tmp_path: Path) -> None:
    products = load_products(SAMPLE)
    collection = open_collection(tmp_path / "chroma", MODEL)

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
    collection = open_collection(tmp_path / "chroma", MODEL)
    sync_products(collection, products, FakeEmbedder())
    before = collection.get()

    result = sync_products(open_collection(tmp_path / "chroma", MODEL), products, FakeEmbedder())

    assert result == SyncResult(upserted=7, deleted=0)
    after = collection.get()
    assert sorted(after["ids"]) == sorted(before["ids"])
    assert after["documents"] == before["documents"]


def test_skus_missing_from_feed_are_removed(tmp_path: Path) -> None:
    products = load_products(SAMPLE)
    collection = open_collection(tmp_path / "chroma", MODEL)
    sync_products(collection, products, FakeEmbedder())

    result = sync_products(collection, products[:5], FakeEmbedder())

    assert result == SyncResult(upserted=5, deleted=2)
    assert sorted(collection.get()["ids"]) == sorted(p.sku for p in products[:5])


def test_large_feeds_are_written_in_batches(tmp_path: Path) -> None:
    products = load_products(SAMPLE)
    collection = open_collection(tmp_path / "chroma", MODEL)
    sync_products(collection, products, FakeEmbedder())
    embedder = FakeEmbedder()

    result = sync_products(collection, products[:5], embedder, batch_size=2)

    assert [len(call) for call in embedder.calls] == [2, 2, 1]
    assert result == SyncResult(upserted=5, deleted=2)
    assert collection.count() == 5


def test_index_records_its_embedding_model(tmp_path: Path) -> None:
    collection = open_collection(tmp_path / "chroma", MODEL)

    assert collection.metadata == {"embedding_model": MODEL}


def test_index_built_with_another_model_is_refused(tmp_path: Path) -> None:
    open_collection(tmp_path / "chroma", MODEL)

    with pytest.raises(IndexModelMismatchError, match="built with embedding model test/fake"):
        open_collection(tmp_path / "chroma", "intfloat/multilingual-e5-small")


# chromadb 1.5 stores embedding_function=None as {"type": "legacy"} and warns on read-back.
@pytest.mark.filterwarnings("ignore:legacy embedding function config:DeprecationWarning")
def test_collection_uses_cosine_distance(tmp_path: Path) -> None:
    hnsw = open_collection(tmp_path / "chroma", MODEL).configuration.get("hnsw")

    assert hnsw is not None
    assert hnsw.get("space") == "cosine"
