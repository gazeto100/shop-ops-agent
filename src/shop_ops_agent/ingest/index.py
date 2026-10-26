"""Write products into the ChromaDB collection, keeping it in sync with the feed."""

from dataclasses import dataclass
from itertools import batched
from pathlib import Path

import chromadb
from chromadb.api.models.Collection import Collection
from chromadb.config import Settings

from shop_ops_agent.embeddings import Embedder
from shop_ops_agent.ingest.chunks import chunk_metadata, chunk_text
from shop_ops_agent.ingest.products import Product

COLLECTION = "products"
# Well below Chroma's per-call limit (5461 on the local SQLite backend); also bounds how
# many texts are embedded at once.
BATCH_SIZE = 1000


@dataclass(frozen=True)
class SyncResult:
    upserted: int
    deleted: int


class IndexModelMismatchError(Exception):
    def __init__(self, path: Path, built_with: str | None, expected: str) -> None:
        super().__init__(
            f"index at {path} was built with embedding model {built_with or '<unknown>'}, "
            f"but EMBEDDING_MODEL is {expected}; delete {path} and re-run ingest"
        )


def open_collection(path: Path, embedding_model: str) -> Collection:
    """Open (or create) the product index, refusing one built with a different model.

    Query vectors are only comparable to stored vectors from the same model; a mismatch
    raises instead of silently returning bad neighbours (both e5 sizes are 384-d).
    """
    client = chromadb.PersistentClient(
        path=str(path), settings=Settings(anonymized_telemetry=False)
    )
    # embedding_function=None: we always pass our own vectors (multilingual e5), so Chroma
    # must never fall back to its default English model.
    collection = client.get_or_create_collection(
        COLLECTION,
        configuration={"hnsw": {"space": "cosine"}},
        metadata={"embedding_model": embedding_model},  # only applied on creation
        embedding_function=None,
    )
    built_with = (collection.metadata or {}).get("embedding_model")
    if built_with != embedding_model:
        raise IndexModelMismatchError(path, built_with, embedding_model)
    return collection


def sync_products(
    collection: Collection,
    products: list[Product],
    embedder: Embedder,
    *,
    batch_size: int = BATCH_SIZE,
) -> SyncResult:
    """Upsert every product by sku, then delete skus that are no longer in the feed."""
    existing = set(collection.get(include=[])["ids"])

    for batch in batched(products, batch_size):
        documents = [chunk_text(p) for p in batch]
        collection.upsert(
            ids=[p.sku for p in batch],
            documents=documents,
            metadatas=[chunk_metadata(p) for p in batch],
            embeddings=embedder.embed_documents(documents),
        )
    # Delete only after every batch succeeded, so a failure never leaves the index emptier.
    stale = sorted(existing - {p.sku for p in products})
    for ids in batched(stale, batch_size):
        collection.delete(ids=list(ids))
    return SyncResult(upserted=len(products), deleted=len(stale))
