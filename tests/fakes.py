"""Test doubles. Unit tests must never hit a real LLM — use FakeLLM."""

import hashlib
from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

from shop_ops_agent.llm import LLMResponse, Message
from shop_ops_agent.rag.retriever import Hit


class FakeLLM:
    """Scripted LLM: returns queued responses in order. Records every call."""

    def __init__(self, responses: list[LLMResponse] | None = None) -> None:
        self._responses = list(responses or [LLMResponse(content="ok")])
        self.calls: list[tuple[list[Message], list[dict[str, Any]] | None]] = []

    async def complete(
        self, messages: list[Message], tools: list[dict[str, Any]] | None = None
    ) -> LLMResponse:
        # Snapshot: callers (the agent loop) keep appending to the same list.
        self.calls.append((list(messages), tools))
        if len(self._responses) > 1:
            return self._responses.pop(0)
        return self._responses[0]


class FakeEmbedder:
    """Deterministic unit vectors derived from a hash of the text. No model, no network."""

    DIM = 8

    def __init__(self) -> None:
        self.calls: list[list[str]] = []
        self.queries: list[str] = []

    def embed_documents(self, texts: Sequence[str]) -> NDArray[np.float32]:
        self.calls.append(list(texts))
        rows = [self.vector(t) for t in texts]
        return np.asarray(rows, dtype=np.float32).reshape(len(rows), self.DIM)

    def embed_query(self, text: str) -> NDArray[np.float32]:
        self.queries.append(text)
        return self.vector(text)

    @classmethod
    def vector(cls, text: str) -> NDArray[np.float32]:
        digest = hashlib.sha256(text.encode()).digest()[: cls.DIM]
        v = np.frombuffer(digest, dtype=np.uint8).astype(np.float32) + 1
        return v / np.linalg.norm(v)


class FakeRetriever:
    """Returns fixed hits for any query. Records every (query, k)."""

    def __init__(self, hits: Sequence[Hit] = ()) -> None:
        self._hits = list(hits)
        self.searches: list[tuple[str, int]] = []

    def search(self, query: str, k: int = 5) -> list[Hit]:
        self.searches.append((query, k))
        return self._hits[:k]
