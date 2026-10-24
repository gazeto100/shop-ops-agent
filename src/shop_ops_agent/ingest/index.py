"""Write products into the ChromaDB collection, keeping it in sync with the feed."""

from dataclasses import dataclass
from pathlib import Path

import chromadb
from chromadb.api.models.Collection import Collection
from chromadb.config import Settings

from shop_ops_agent.embeddings import Embedder
from shop_ops_agent.ingest.chunks import chunk_metadata, chunk_text
from shop_ops_agent.ingest.products import Product

COLLECTION = "products"


@dataclass(frozen=True)
class SyncResult:
    upserted: int
    deleted: int


def open_collection(path: Path) -> Collection:
    client = chromadb.PersistentClient(
        path=str(path), settings=Settings(anonymized_telemetry=False)
    )
    # embedding_function=None: we always pass our own vectors (multilingual e5), so Chroma
    # must never fall back to its default English model.
    return client.get_or_create_collection(
        COLLECTION, configuration={"hnsw": {"space": "cosine"}}, embedding_function=None
    )


def sync_products(
    collection: Collection, products: list[Product], embedder: Embedder
) -> SyncResult:
    """Upsert every product by sku, then delete skus that are no longer in the feed."""
    ids = [p.sku for p in products]
    documents = [chunk_text(p) for p in products]
    existing = set(collection.get(include=[])["ids"])

    collection.upsert(
        ids=ids,
        documents=documents,
        metadatas=[chunk_metadata(p) for p in products],
        embeddings=embedder.embed_documents(documents),
    )
    # Delete only after the upsert succeeded, so a failure never leaves the index emptier.
    stale = sorted(existing - set(ids))
    if stale:
        collection.delete(ids=stale)
    return SyncResult(upserted=len(ids), deleted=len(stale))
