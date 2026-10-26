"""Hybrid retrieval: ChromaDB cosine neighbours + BM25, merged by reciprocal-rank fusion.

Vector search catches paraphrases ("водонагревател" ~ "бойлер"); BM25 catches exact
tokens the embedding blurs (model numbers, "12000 BTU", brand names). RRF combines the
two *rankings*, so their incomparable raw scores never need normalising.
"""

import re
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from chromadb.api.models.Collection import Collection
from rank_bm25 import BM25Okapi

from shop_ops_agent.embeddings import Embedder

# 60 is the constant from the original RRF paper (Cormack et al., 2009): large enough
# that one list's top hit can't drown out agreement between lists.
RRF_K = 60
# How deep each list is read before fusing; more than any k we ask for.
CANDIDATES = 20


@dataclass(frozen=True)
class Hit:
    sku: str
    name: str
    score: float  # RRF score: higher is better, only comparable within one search
    text: str

    def citation(self) -> str:
        return f"[{self.sku}: {self.name}]"


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
    """Searches the product index. Reads all chunks once at construction, so build a new
    Retriever after re-ingesting."""

    def __init__(self, collection: Collection, embedder: Embedder) -> None:
        self._collection = collection
        self._embedder = embedder
        stored = collection.get(include=["documents", "metadatas"])
        self._ids = stored["ids"]
        self._texts = dict(zip(self._ids, stored["documents"] or [], strict=True))
        self._names = {
            doc_id: str(meta["name"])
            for doc_id, meta in zip(self._ids, stored["metadatas"] or [], strict=True)
        }
        # BM25 over exactly the stored chunks, so both retrievers rank the same corpus.
        corpus = [tokenize(self._texts[doc_id]) for doc_id in self._ids]
        self._bm25 = BM25Okapi(corpus) if corpus else None

    def search(self, query: str, k: int = 5) -> list[Hit]:
        if not self._ids or k <= 0:
            return []
        fused = reciprocal_rank_fusion([self._vector_ranking(query), self._bm25_ranking(query)])
        return [
            Hit(sku=doc_id, name=self._names[doc_id], score=score, text=self._texts[doc_id])
            for doc_id, score in fused[:k]
        ]

    def _vector_ranking(self, query: str) -> list[str]:
        result = self._collection.query(
            query_embeddings=self._embedder.embed_query(query).reshape(1, -1),
            n_results=min(CANDIDATES, len(self._ids)),
            include=[],
        )
        return result["ids"][0]

    def _bm25_ranking(self, query: str) -> list[str]:
        assert self._bm25 is not None  # search() returns early on an empty index
        scores = np.asarray(self._bm25.get_scores(tokenize(query)), dtype=np.float64)
        # Only documents sharing at least one query term count as ranked by BM25.
        order = [i for i in np.argsort(-scores, kind="stable") if scores[i] > 0]
        return [self._ids[i] for i in order[:CANDIDATES]]
