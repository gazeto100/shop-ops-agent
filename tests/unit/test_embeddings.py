"""The real embedder, with sentence-transformers' model replaced by a recording stub."""

import sys
import types
from collections.abc import Sequence
from typing import Any

import numpy as np
import pytest

from shop_ops_agent.embeddings import SentenceTransformerEmbedder


class StubModel:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.kwargs: dict[str, Any] = {}

    def encode_document(self, texts: Sequence[str], **kwargs: Any) -> Any:
        self.kwargs = kwargs
        return np.ones((len(texts), 4), dtype=np.float64)  # float64 on purpose

    def encode_query(self, texts: Sequence[str], **kwargs: Any) -> Any:
        self.kwargs = kwargs
        return np.ones((len(texts), 4), dtype=np.float64)


@pytest.fixture
def embedder(monkeypatch: pytest.MonkeyPatch) -> SentenceTransformerEmbedder:
    # A stand-in module, picked up by the embedder's deferred import: the real package
    # pulls in torch (~15 s on Windows) and would download the model.
    stub = types.ModuleType("sentence_transformers")
    monkeypatch.setattr(stub, "SentenceTransformer", StubModel, raising=False)
    monkeypatch.setitem(sys.modules, "sentence_transformers", stub)
    return SentenceTransformerEmbedder("intfloat/multilingual-e5-small")


def model_of(embedder: SentenceTransformerEmbedder) -> StubModel:
    model = embedder._model
    assert isinstance(model, StubModel)
    return model


def test_documents_get_e5_passage_prefix_and_are_normalised(
    embedder: SentenceTransformerEmbedder,
) -> None:
    embedder.embed_documents(["Бойлер 80 л"])

    kwargs = model_of(embedder).kwargs
    assert kwargs["prompt"] == "passage: "
    assert kwargs["normalize_embeddings"] is True


def test_returns_float32_matrix_one_row_per_text(embedder: SentenceTransformerEmbedder) -> None:
    vectors = embedder.embed_documents(["a", "b", "c"])

    assert vectors.dtype == np.float32
    assert vectors.shape == (3, 4)


def test_queries_get_e5_query_prefix_and_are_normalised(
    embedder: SentenceTransformerEmbedder,
) -> None:
    vector = embedder.embed_query("kettle")

    kwargs = model_of(embedder).kwargs
    assert kwargs["prompt"] == "query: "
    assert kwargs["normalize_embeddings"] is True
    assert vector.dtype == np.float32
    assert vector.shape == (4,)


def test_loads_the_configured_model(embedder: SentenceTransformerEmbedder) -> None:
    assert model_of(embedder).model_name == "intfloat/multilingual-e5-small"
