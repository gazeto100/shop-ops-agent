"""Hybrid retrieval: ChromaDB cosine neighbours + BM25, merged by reciprocal-rank fusion.

Vector search catches paraphrases ("водонагревател" ~ "бойлер"); BM25 catches exact
tokens the embedding blurs (model numbers, "12000 BTU", brand names). RRF combines the
two *rankings*, so their incomparable raw scores never need normalising.
"""

import re
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from chromadb.api.models.Collection import Collection
from rank_bm25 import BM25Okapi

from shop_ops_agent.embeddings import Embedder

# 60 is the constant from the original RRF paper (Cormack et al., 2009): large enough
# that one list's top hit can't drown out agreement between lists.
RRF_K = 60
# Minimum depth each list is read before fusing (deeper when k is larger).
CANDIDATES = 20


@dataclass(frozen=True)
class Hit:
    sku: str
    name: str
    score: float  # RRF score: higher is better, only comparable within one search
    text: str

    def citation(self) -> str:
        return f"[{self.sku}: {self.name}]"


class Searcher(Protocol):
    """What callers need from a retriever. Defined here, next to `Hit`, so both the chat
    service and the tools can depend on it without importing each other."""

    def search(self, query: str, k: int = 5) -> list[Hit]: ...


def tokenize(text: str) -> list[str]:
    """Lower-cased word tokens; `\\w` is Unicode-aware, so Cyrillic works too."""
    return re.findall(r"\w+", text.lower())


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[str]], k: int = RRF_K
) -> list[tuple[str, float]]:
    """Fuse ranked id lists: score(id) = sum over lists of 1 / (k + rank), rank from 1."""
    scores: defaultdict[str, float] = defaultdict(float)
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] += 1 / (k + rank)
    # Ties broken by id so results are deterministic.
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))


class Retriever:
    """Searches the product index.

    BM25 statistics are built once from the chunks present at construction; vector search
    and the returned hits always come from the live index. So after a re-ingest, removed
    products never come back and new ones are found by vector search, but BM25 only
    knows them once a new Retriever is built.
    """

    def __init__(self, collection: Collection, embedder: Embedder) -> None:
        self._collection = collection
        self._embedder = embedder
        stored = collection.get(include=["documents"])
        self._bm25_ids = stored["ids"]
        # BM25 over exactly the stored chunks, so both retrievers rank the same corpus.
        corpus = [tokenize(doc) for doc in stored["documents"] or []]
        self._bm25 = BM25Okapi(corpus) if corpus else None

    def search(self, query: str, k: int = 5) -> list[Hit]:
        if k <= 0 or not query.strip():
            return []  # an empty query would just return arbitrary neighbours
        depth = max(CANDIDATES, k)
        fused = reciprocal_rank_fusion(
            [self._vector_ranking(query, depth), self._bm25_ranking(query, depth)]
        )
        if not fused:
            return []  # empty index (Chroma rejects get(ids=[]))
        # Resolve against the live index: ingest may have added or removed products
        # since this Retriever was built.
        live = self._collection.get(
            ids=[doc_id for doc_id, _ in fused], include=["documents", "metadatas"]
        )
        current = {
            doc_id: (str(meta["name"]), doc)
            for doc_id, doc, meta in zip(
                live["ids"], live["documents"] or [], live["metadatas"] or [], strict=True
            )
        }
        hits = [
            Hit(sku=doc_id, name=current[doc_id][0], score=score, text=current[doc_id][1])
            for doc_id, score in fused
            if doc_id in current
        ]
        return hits[:k]

    def _vector_ranking(self, query: str, depth: int) -> list[str]:
        n_results = min(depth, self._collection.count())
        if n_results == 0:
            return []
        result = self._collection.query(
            query_embeddings=self._embedder.embed_query(query).reshape(1, -1),
            n_results=n_results,
            include=[],
        )
        return result["ids"][0]

    def _bm25_ranking(self, query: str, depth: int) -> list[str]:
        if self._bm25 is None:
            return []
        scores = np.asarray(self._bm25.get_scores(tokenize(query)), dtype=np.float64)
        # Only documents sharing at least one query term count as ranked by BM25.
        order = [i for i in np.argsort(-scores, kind="stable") if scores[i] > 0]
        return [self._bm25_ids[i] for i in order[:depth]]
