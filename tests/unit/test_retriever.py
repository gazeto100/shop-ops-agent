"""Fusion logic and wiring, with FakeEmbedder. Ranking *quality* with the real model is
covered in tests/integration/test_retrieval.py."""

from pathlib import Path

import pytest

from shop_ops_agent.ingest.chunks import chunk_text
from shop_ops_agent.ingest.index import open_collection, sync_products
from shop_ops_agent.ingest.products import load_products
from shop_ops_agent.rag.retriever import Hit, Retriever, reciprocal_rank_fusion, tokenize
from tests.fakes import FakeEmbedder

SAMPLE = Path(__file__).parents[2] / "data" / "products.sample.csv"
MODEL = "test/fake-model"


@pytest.fixture
def embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture
def retriever(tmp_path: Path, embedder: FakeEmbedder) -> Retriever:
    collection = open_collection(tmp_path / "chroma", MODEL)
    sync_products(collection, load_products(SAMPLE), embedder)
    return Retriever(collection, embedder)


def test_rrf_scores_are_summed_reciprocal_ranks() -> None:
    fused = dict(reciprocal_rank_fusion([["a", "b"], ["b", "c"]], k=60))

    assert fused == pytest.approx({"a": 1 / 61, "b": 1 / 62 + 1 / 61, "c": 1 / 62})


def test_rrf_rewards_agreement_over_a_single_top_rank() -> None:
    # "b" is never first, but both lists like it: it beats each list's winner.
    ranked = [doc for doc, _ in reciprocal_rank_fusion([["a", "b"], ["c", "b"]])]

    assert ranked[0] == "b"


def test_rrf_breaks_ties_by_id() -> None:
    ranked = [doc for doc, _ in reciprocal_rank_fusion([["b", "a"], ["a", "b"]])]

    assert ranked == ["a", "b"]


def test_tokenize_lowercases_and_handles_cyrillic() -> None:
    assert tokenize("Бойлер Tesy 80 л, A++") == ["бойлер", "tesy", "80", "л", "a"]


def test_exact_token_match_ranks_first_whatever_the_vectors_say(retriever: Retriever) -> None:
    # Fake vectors are arbitrary, but every product is in the vector list, so the only
    # product BM25 also ranks ("Tesy" appears once) must come out on top.
    [first, *_] = retriever.search("Tesy")

    assert first.sku == "BOIL-80-A"


def test_hits_carry_stored_text_name_and_descending_scores(retriever: Retriever) -> None:
    hits = retriever.search("бойлер", k=3)

    assert len(hits) == 3
    assert [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)
    boiler = next(p for p in load_products(SAMPLE) if p.sku == hits[0].sku)
    assert hits[0].text == chunk_text(boiler)
    assert hits[0].name == boiler.name


def test_query_is_embedded_as_a_query(retriever: Retriever, embedder: FakeEmbedder) -> None:
    retriever.search("kettle")

    assert embedder.queries == ["kettle"]


def test_k_caps_results_and_never_exceeds_the_catalog(retriever: Retriever) -> None:
    assert len(retriever.search("бойлер", k=2)) == 2
    assert len(retriever.search("бойлер", k=50)) == 7
    assert retriever.search("бойлер", k=0) == []


def test_empty_index_returns_no_hits(tmp_path: Path, embedder: FakeEmbedder) -> None:
    retriever = Retriever(open_collection(tmp_path / "chroma", MODEL), embedder)

    assert retriever.search("kettle") == []


def test_hit_citation_format() -> None:
    hit = Hit(sku="KET-17-S", name="Електрическа кана 1.7 л инокс", score=0.03, text="…")

    assert hit.citation() == "[KET-17-S: Електрическа кана 1.7 л инокс]"
