"""Product -> the text that gets embedded, plus the metadata stored next to it.

Price and stock are deliberately absent from both: they change daily and are always read
live from the catalog by the tools, so the index can never serve a stale price.
"""

from shop_ops_agent.ingest.products import Product


def chunk_text(product: Product) -> str:
    text = (
        f"{product.name}\n"
        f"Категория: {product.category}\n"
        f"{product.description}\n"
        f"{warranty_text(product.warranty_months)}"
    )
    # English name last, so a product without one is chunked exactly as before.
    return f"{text}\n{product.name_en}" if product.name_en else text


def warranty_text(months: int) -> str:
    # The model may quote this to customers, so it must be correct Bulgarian.
    match months:
        case 0:
            return "Без гаранция"
        case 1:
            return "Гаранция: 1 месец"
        case _:
            return f"Гаранция: {months} месеца"


def chunk_metadata(product: Product) -> dict[str, str]:
    return {"sku": product.sku, "name": product.name, "category": product.category}
