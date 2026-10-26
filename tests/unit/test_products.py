from decimal import Decimal
from pathlib import Path

import pytest

from shop_ops_agent.ingest.products import ProductValidationError, load_products

SAMPLE = Path(__file__).parents[2] / "data" / "products.sample.csv"
HEADER = (
    "sku,name,category,price_eur,price_bgn,stock_qty,supplier_sku,supplier_price_bgn,"
    "description,warranty_months"
)
GOOD = "A-1,Кана,Малки уреди,24.90,48.70,12,S-1,38.00,2200 W,24"


def write_feed(tmp_path: Path, *rows: str, header: str = HEADER) -> Path:
    path = tmp_path / "feed.csv"
    path.write_text("\n".join([header, *rows]) + "\n", encoding="utf-8")
    return path


def error_lines(path: Path) -> list[str]:
    with pytest.raises(ProductValidationError) as exc:
        load_products(path)
    return [str(e) for e in exc.value.errors]


def test_sample_feed_loads_every_row_typed() -> None:
    products = load_products(SAMPLE)

    assert len(products) == 7
    boiler = products[0]
    assert boiler.sku == "BOIL-80-A"
    assert boiler.name == "Бойлер Tesy 80 л вертикален"
    assert boiler.price_eur == Decimal("189.00")
    assert boiler.stock_qty == 4
    assert boiler.warranty_months == 60


def test_utf8_bom_and_surrounding_whitespace_are_tolerated(tmp_path: Path) -> None:
    path = tmp_path / "feed.csv"
    path.write_text(
        f"﻿{HEADER}\n A-1 ,Кана ,Малки уреди,24.90,48.70,12,S-1,38.00,x,24\n",
        encoding="utf-8",
    )

    [product] = load_products(path)

    assert product.sku == "A-1"
    assert product.name == "Кана"


@pytest.mark.parametrize(
    ("row", "expected"),
    [
        ("A-1,Кана,Уреди,0,48.70,12,S-1,38.00,x,24", "price_eur: Input should be greater than 0"),
        ("A-1,Кана,Уреди,24.90,48.70,-1,S-1,38.00,x,24", "stock_qty: Input should be greater"),
        ("A-1,Кана,Уреди,24.999,48.70,1,S-1,38.00,x,24", "price_eur: Decimal input should have no"),
        ("A-1,Кана,Уреди,24.90,48.70,two,S-1,38.00,x,24", "stock_qty: Input should be a valid int"),
        (",Кана,Уреди,24.90,48.70,1,S-1,38.00,x,24", "sku: String should have at least 1"),
    ],
)
def test_invalid_field_is_reported_with_line_and_field(
    tmp_path: Path, row: str, expected: str
) -> None:
    [error] = error_lines(write_feed(tmp_path, GOOD.replace("A-1", "OK-1"), row))

    assert error.startswith("line 3")
    assert expected in error


def test_all_bad_rows_are_reported_not_just_the_first(tmp_path: Path) -> None:
    path = write_feed(
        tmp_path,
        "A-1,Кана,Уреди,0,48.70,12,S-1,38.00,x,24",
        GOOD.replace("A-1", "OK-1"),
        "A-3,Кана,Уреди,24.90,48.70,-5,S-1,38.00,x,24",
    )

    errors = error_lines(path)

    assert [e.split(":")[0] for e in errors] == ["line 2 (sku A-1)", "line 4 (sku A-3)"]


def test_duplicate_sku_points_at_first_occurrence(tmp_path: Path) -> None:
    errors = error_lines(write_feed(tmp_path, GOOD, GOOD))

    assert errors == ["line 3 (sku A-1): duplicate sku (first seen on line 2)"]


def test_missing_column_fails_once_for_the_header(tmp_path: Path) -> None:
    header = HEADER.replace(",stock_qty", "")
    errors = error_lines(write_feed(tmp_path, GOOD, header=header))

    assert errors == ["line 1: missing columns: stock_qty"]


@pytest.mark.parametrize(
    ("row", "expected"),
    [
        (GOOD + ",EXTRA", "11 fields, header has 10"),
        ("A-1,Кана,Уреди", "3 fields, header has 10"),
    ],
)
def test_misaligned_row_is_rejected(tmp_path: Path, row: str, expected: str) -> None:
    assert error_lines(write_feed(tmp_path, row)) == [f"line 2 (sku A-1): {expected}"]


def test_empty_feed_is_rejected(tmp_path: Path) -> None:
    assert error_lines(write_feed(tmp_path)) == ["line 1: feed contains no products"]


def test_windows_1251_feed_is_rejected_as_not_utf8(tmp_path: Path) -> None:
    path = tmp_path / "feed.csv"
    path.write_bytes(f"{HEADER}\n{GOOD}\n".encode("cp1251"))

    [error] = error_lines(path)

    assert error.startswith("file: not UTF-8 (invalid byte at offset ")
