"""Text embeddings behind a small Protocol, so tests can swap in a fake."""

from collections.abc import Sequence
from typing import Protocol

import numpy as np
from numpy.typing import NDArray


class Embedder(Protocol):
    def embed_documents(self, texts: Sequence[str]) -> NDArray[np.float32]:
        """One L2-normalised row per text."""
        ...

    def embed_query(self, text: str) -> NDArray[np.float32]:
        """One L2-normalised vector, comparable to `embed_documents` rows."""
        ...


class SentenceTransformerEmbedder:
    # e5 models are trained with these prefixes; leaving them out degrades retrieval.
    PASSAGE_PREFIX = "passage: "
    QUERY_PREFIX = "query: "

    def __init__(self, model_name: str) -> None:
        # Imported here: pulling in torch takes seconds, and most callers (tests, the
        # API before #4) never need it.
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name)

    def embed_documents(self, texts: Sequence[str]) -> NDArray[np.float32]:
        vectors = self._model.encode_document(
            list(texts), prompt=self.PASSAGE_PREFIX, normalize_embeddings=True
        )
        return np.asarray(vectors, dtype=np.float32)

    def embed_query(self, text: str) -> NDArray[np.float32]:
        vectors = self._model.encode_query(
            [text], prompt=self.QUERY_PREFIX, normalize_embeddings=True
        )
        return np.asarray(vectors, dtype=np.float32).reshape(-1)  # (1, d) -> (d,)
