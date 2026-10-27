"""Tool logic, schemas and registry (#6). No LLM, no model, no network."""

import json
import os
import shutil
from decimal import Decimal
from pathlib import Path

import pytest
from pydantic import BaseModel

from shop_ops_agent.agent.tools import Catalog, ToolError, build_registry
from shop_ops_agent.agent.tools.definitions import (
    CompareSupplierPriceInput,
    ProductNotFound,
    StockAndPrice,
    below_ours_pct,
    compare_supplier_price,
)
from shop_ops_agent.agent.tools.registry import Tool, ToolRegistry
from shop_ops_agent.ingest.products import ProductValidationError
from shop_ops_agent.rag.retriever import Hit
from tests.fakes import FakeRetriever

SAMPLE = Path(__file__).parents[2] / "data" / "products.sample.csv"
HITS = [
    Hit(sku="KET-17-S", name="Електрическа кана 1.7 л инокс", score=0.03, text="Кана…"),
    Hit(sku="KET-10-P", name="Електрическа кана 1.0 л пластмаса", score=0.02, text="Кана…"),
]


@pytest.fixture
def feed(tmp_path: Path) -> Path:
    path = tmp_path / "feed.csv"
    shutil.copy(SAMPLE, path)
    return path


@pytest.fixture
def registry(feed: Path) -> ToolRegistry:
    return build_registry(Catalog(feed), FakeRetriever(HITS))


def bump_mtime(path: Path) -> None:
    """Force a visible mtime change; some filesystems only store seconds."""
    stat = path.stat()
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 2_000_000_000))


# --- Schemas exported to the LLM --------------------------------------------------------


def test_customers_get_the_two_customer_tools(registry: ToolRegistry) -> None:
    names = [s["name"] for s in registry.specs("customer")]

    assert names == ["search_products", "check_stock_and_price"]


def test_operators_also_get_the_supplier_tool(registry: ToolRegistry) -> None:
    names = [s["name"] for s in registry.specs("operator")]

    assert names == ["search_products", "check_stock_and_price", "compare_supplier_price"]


def test_specs_are_strict_json_schemas_with_descriptions(registry: ToolRegistry) -> None:
    for spec in registry.specs("operator"):
        params = spec["parameters"]
        assert spec.get("description")
        assert params["type"] == "object"
        assert params["additionalProperties"] is False
        assert params["required"]
        assert all(p.get("description") for p in params["properties"].values())
        json.dumps(params)  # must be plain JSON for the provider


def test_threshold_is_a_plain_number_in_the_schema(registry: ToolRegistry) -> None:
    [spec] = [s for s in registry.specs("operator") if s["name"] == "compare_supplier_price"]

    assert spec["parameters"]["properties"]["threshold_pct"]["type"] == "number"


# --- search_products --------------------------------------------------------------------


def test_search_returns_codes_names_text_and_no_price(feed: Path) -> None:
    retriever = FakeRetriever(HITS)
    registry = build_registry(Catalog(feed), retriever)

    result = registry.call("search_products", {"query": "кана", "k": 2}, "customer")

    assert retriever.searches == [("кана", 2)]
    assert result.model_dump() == {
        "products": [{"sku": h.sku, "name": h.name, "text": h.text} for h in HITS]
    }
    assert "price" not in result.model_dump_json()


def test_search_k_defaults_to_5(feed: Path) -> None:
    retriever = FakeRetriever(HITS)

    build_registry(Catalog(feed), retriever).call("search_products", {"query": "x"}, "customer")

    assert retriever.searches == [("x", 5)]


# --- check_stock_and_price --------------------------------------------------------------


def test_known_sku_returns_eur_first_price_and_stock(registry: ToolRegistry) -> None:
    result = registry.call("check_stock_and_price", {"sku": "BOIL-80-A"}, "customer")

    assert result == StockAndPrice(
        sku="BOIL-80-A",
        name="Бойлер Tesy 80 л вертикален",
        price_eur=Decimal("189.00"),
        price_bgn=Decimal("369.64"),
        stock_qty=4,
        in_stock=True,
    )
    # Exact money in the JSON the model receives.
    assert '"price_eur":"189.00"' in result.model_dump_json()


def test_sku_lookup_ignores_case_and_spaces(registry: ToolRegistry) -> None:
    result = registry.call("check_stock_and_price", {"sku": "  boil-80-a "}, "customer")

    assert isinstance(result, StockAndPrice)
    assert result.sku == "BOIL-80-A"


def test_out_of_stock_product(registry: ToolRegistry) -> None:
    result = registry.call("check_stock_and_price", {"sku": "KET-10-P"}, "customer")

    assert isinstance(result, StockAndPrice)
    assert (result.stock_qty, result.in_stock) == (0, False)


def test_unknown_sku_is_a_typed_not_found_and_never_raises(registry: ToolRegistry) -> None:
    result = registry.call("check_stock_and_price", {"sku": "ZZZ-999"}, "customer")

    assert result == ProductNotFound(sku="ZZZ-999")
    assert json.loads(result.model_dump_json())["status"] == "not_found"


# --- compare_supplier_price -------------------------------------------------------------


def test_gap_is_relative_to_our_price() -> None:
    # Boiler: ours 369.64, supplier 310.00 -> (369.64 - 310) / 369.64 = 16.13 %
    assert below_ours_pct(Decimal("369.64"), Decimal("310.00")).quantize(Decimal("0.01")) == (
        Decimal("16.13")
    )


def test_supplier_gaps_above_threshold_largest_first(registry: ToolRegistry) -> None:
    result = registry.call("compare_supplier_price", {"threshold_pct": 15}, "operator")

    data = json.loads(result.model_dump_json())
    assert data["threshold_pct"] == "15.0"
    assert [(p["sku"], p["below_ours_pct"]) for p in data["products"]] == [
        ("HEAT-2K-C", "64.5"),
        ("KET-10-P", "24.5"),
        ("KET-17-S", "22.0"),
        ("AC-12-INV", "17.1"),
        ("BOIL-80-A", "16.1"),
        ("BOIL-50-B", "15.9"),
    ]  # AC-18-INV (11.9 %) is below the threshold


def test_threshold_is_strictly_more_than(feed: Path) -> None:
    catalog = Catalog(feed)
    exact = below_ours_pct(Decimal("291.42"), Decimal("245.00"))  # BOIL-50-B, 15.93…%
    just_below = float(exact) - 0.001

    at = compare_supplier_price(catalog, CompareSupplierPriceInput(threshold_pct=float(exact)))
    under = compare_supplier_price(catalog, CompareSupplierPriceInput(threshold_pct=just_below))

    assert "BOIL-50-B" not in {p.sku for p in at.products}
    assert "BOIL-50-B" in {p.sku for p in under.products}


def test_equal_gaps_are_ordered_by_sku(feed: Path) -> None:
    lines = feed.read_text(encoding="utf-8").splitlines()
    # Give both kettles identical prices so their gaps tie exactly.
    lines = [line.replace(",29.14,0,KT-10P,22.00,", ",48.70,0,KT-10P,38.00,") for line in lines]
    feed.write_text("\n".join(lines) + "\n", encoding="utf-8")

    result = compare_supplier_price(Catalog(feed), CompareSupplierPriceInput(threshold_pct=20))

    assert [p.sku for p in result.products][-2:] == ["KET-10-P", "KET-17-S"]


def test_supplier_tool_is_refused_to_customers(registry: ToolRegistry) -> None:
    result = registry.call("compare_supplier_price", {"threshold_pct": 10}, "customer")

    # Same answer as for a tool that doesn't exist: customers can't even probe for it.
    assert result == ToolError(
        tool="compare_supplier_price", message="Unknown tool 'compare_supplier_price'."
    )


# --- Registry never raises --------------------------------------------------------------


def test_unknown_tool(registry: ToolRegistry) -> None:
    result = registry.call("delete_everything", {}, "operator")

    assert result == ToolError(
        tool="delete_everything", message="Unknown tool 'delete_everything'."
    )


@pytest.mark.parametrize(
    ("name", "arguments", "fragment"),
    [
        ("check_stock_and_price", {}, "sku: Field required"),
        ("check_stock_and_price", {"sku": ""}, "sku: String should have at least 1 character"),
        (
            "check_stock_and_price",
            {"sku": "X", "extra": 1},
            "extra: Extra inputs are not permitted",
        ),
        ("search_products", {"query": "x", "k": 50}, "k: Input should be less than or equal to 10"),
        ("compare_supplier_price", {"threshold_pct": -5}, "threshold_pct: Input should be greater"),
    ],
)
def test_invalid_arguments_become_a_tool_error(
    registry: ToolRegistry, name: str, arguments: dict[str, object], fragment: str
) -> None:
    result = registry.call(name, arguments, "operator")

    assert isinstance(result, ToolError)
    assert result.message.startswith("Invalid arguments: ")
    assert fragment in result.message


def test_a_crashing_tool_becomes_a_tool_error(caplog: pytest.LogCaptureFixture) -> None:
    class Args(BaseModel):
        pass

    def boom(_: Args) -> BaseModel:
        raise RuntimeError("database on fire")

    registry = ToolRegistry([Tool(name="boom", description="x", input_model=Args, run=boom)])

    result = registry.call("boom", {}, "customer")

    assert result == ToolError(tool="boom", message="The tool failed; try again later.")
    assert "database on fire" not in result.model_dump_json()  # details only in the log
    assert "tool boom failed" in caplog.text


def test_missing_catalog_file_becomes_a_tool_error(tmp_path: Path) -> None:
    registry = build_registry(Catalog(tmp_path / "missing.csv"), FakeRetriever())

    result = registry.call("check_stock_and_price", {"sku": "BOIL-80-A"}, "customer")

    assert isinstance(result, ToolError)


# --- Catalog freshness ------------------------------------------------------------------


def test_catalog_picks_up_a_changed_feed(feed: Path) -> None:
    catalog = Catalog(feed)
    assert catalog.get("BOIL-80-A").stock_qty == 4  # type: ignore[union-attr]

    feed.write_text(
        feed.read_text(encoding="utf-8").replace(",369.64,4,", ",369.64,9,"), encoding="utf-8"
    )
    bump_mtime(feed)

    assert catalog.get("BOIL-80-A").stock_qty == 9  # type: ignore[union-attr]


def test_unchanged_feed_is_not_reparsed(feed: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    catalog = Catalog(feed)
    catalog.products()
    monkeypatch.setattr(
        "shop_ops_agent.agent.tools.catalog.load_products",
        lambda _: pytest.fail("re-parsed an unchanged file"),
    )

    assert len(catalog.products()) == 7


def test_bad_new_feed_keeps_the_last_good_one(feed: Path, caplog: pytest.LogCaptureFixture) -> None:
    catalog = Catalog(feed)
    assert len(catalog.products()) == 7

    feed.write_text("sku,name\nBROKEN,row\n", encoding="utf-8")
    bump_mtime(feed)

    assert len(catalog.products()) == 7
    assert "rejected, keeping previous version" in caplog.text


def test_bad_first_feed_raises(feed: Path) -> None:
    feed.write_text("sku,name\nBROKEN,row\n", encoding="utf-8")

    with pytest.raises(ProductValidationError):
        Catalog(feed).products()
