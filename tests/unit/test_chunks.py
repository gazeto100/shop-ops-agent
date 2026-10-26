from pathlib import Path

import pytest

from shop_ops_agent.ingest.chunks import chunk_metadata, chunk_text
from shop_ops_agent.ingest.products import Product, load_products

SAMPLE = Path(__file__).parents[2] / "data" / "products.sample.csv"


def boiler() -> Product:
    return next(p for p in load_products(SAMPLE) if p.sku == "BOIL-80-A")


def test_chunk_is_name_category_description_warranty_english_name() -> None:
    assert chunk_text(boiler()) == (
        "Бойлер Tesy 80 л вертикален\n"
        "Категория: Бойлери\n"
        "Емайлиран водосъдържател, 2 kW, термостат, вертикален монтаж\n"
        "Гаранция: 60 месеца\n"
        "Tesy 80 l vertical electric water heater (boiler)"
    )


def test_without_english_name_the_chunk_is_unchanged_from_issue_3() -> None:
    assert chunk_text(boiler().model_copy(update={"name_en": ""})) == (
        "Бойлер Tesy 80 л вертикален\n"
        "Категория: Бойлери\n"
        "Емайлиран водосъдържател, 2 kW, термостат, вертикален монтаж\n"
        "Гаранция: 60 месеца"
    )


@pytest.mark.parametrize(
    ("months", "expected"),
    [(0, "Без гаранция"), (1, "Гаранция: 1 месец"), (24, "Гаранция: 24 месеца")],
)
def test_warranty_line_is_correct_bulgarian(months: int, expected: str) -> None:
    product = boiler().model_copy(update={"warranty_months": months, "name_en": ""})

    assert chunk_text(product).splitlines()[-1] == expected


def test_volatile_fields_stay_out_of_the_index() -> None:
    product = boiler()
    text = chunk_text(product)

    for value in (product.price_eur, product.price_bgn, product.supplier_price_bgn):
        assert str(value) not in text
    assert chunk_metadata(product) == {
        "sku": "BOIL-80-A",
        "name": "Бойлер Tesy 80 л вертикален",
        "category": "Бойлери",
    }
