from pathlib import Path

from shop_ops_agent.ingest.chunks import chunk_metadata, chunk_text
from shop_ops_agent.ingest.products import Product, load_products

SAMPLE = Path(__file__).parents[2] / "data" / "products.sample.csv"


def boiler() -> Product:
    return next(p for p in load_products(SAMPLE) if p.sku == "BOIL-80-A")


def test_chunk_is_name_category_description_warranty() -> None:
    assert chunk_text(boiler()) == (
        "Бойлер Tesy 80 л вертикален\n"
        "Категория: Бойлери\n"
        "Емайлиран водосъдържател, 2 kW, термостат, вертикален монтаж\n"
        "Гаранция: 60 месеца"
    )


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
