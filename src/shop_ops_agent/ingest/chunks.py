"""Product -> the text that gets embedded, plus the metadata stored next to it.

Price and stock are deliberately absent from both: they change daily and are always read
live from the catalog by the tools, so the index can never serve a stale price.
"""

from shop_ops_agent.ingest.products import Product


def chunk_text(product: Product) -> str:
    return (
        f"{product.name}\n"
        f"Категория: {product.category}\n"
        f"{product.description}\n"
        f"Гаранция: {product.warranty_months} месеца"
    )


def chunk_metadata(product: Product) -> dict[str, str]:
    return {"sku": product.sku, "name": product.name, "category": product.category}
