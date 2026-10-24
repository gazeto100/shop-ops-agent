from pathlib import Path

import pytest

import shop_ops_agent.ingest.__main__ as cli
from shop_ops_agent.config import Settings
from shop_ops_agent.ingest.index import open_collection
from tests.fakes import FakeEmbedder

SAMPLE = Path(__file__).parents[2] / "data" / "products.sample.csv"


@pytest.fixture
def chroma_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "chroma"
    # model_construct: defaults + overrides only; never reads env vars or the dev's .env.
    settings = Settings.model_construct(chroma_path=str(path))
    monkeypatch.setattr(cli, "get_settings", lambda: settings)
    monkeypatch.setattr(cli, "SentenceTransformerEmbedder", lambda _model: FakeEmbedder())
    return path


def test_ingest_sample_feed(chroma_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main([str(SAMPLE)]) == 0

    assert "Indexed 7 products" in capsys.readouterr().out
    assert open_collection(chroma_path).count() == 7


def test_invalid_feed_exits_1_and_indexes_nothing(
    chroma_path: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    bad = tmp_path / "bad.csv"
    rows = SAMPLE.read_text(encoding="utf-8").splitlines()
    rows[2] = rows[2].replace(",0,EL-50H", ",-3,EL-50H")  # BOIL-50-B stock -3
    bad.write_text("\n".join(rows), encoding="utf-8")

    assert cli.main([str(bad)]) == 1

    err = capsys.readouterr().err
    assert "1 error(s), nothing indexed" in err
    assert "line 3 (sku BOIL-50-B): stock_qty" in err
    assert not chroma_path.exists()
