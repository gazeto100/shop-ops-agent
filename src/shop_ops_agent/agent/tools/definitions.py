"""The three product tools: Pydantic input and output schemas plus the logic.

Inputs forbid extra fields, so their JSON Schema has `additionalProperties: false` and the
model can't smuggle in arguments. Outputs are what the agent loop (#7) passes back to the
model verbatim, as JSON: money stays `Decimal` and serialises as an exact string
("189.00"), never as a rounded float.
"""

from decimal import ROUND_HALF_UP, Decimal
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from shop_ops_agent.agent.tools.catalog import Catalog
from shop_ops_agent.rag.retriever import Searcher


class _Input(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


# --- search_products --------------------------------------------------------------------


class SearchProductsInput(_Input):
    query: str = Field(
        min_length=1, max_length=200, description="What the user is looking for, in their words."
    )
    k: int = Field(default=5, ge=1, le=10, description="How many products to return.")


class ProductMatch(BaseModel):
    sku: str
    name: str
    text: str = Field(description="Catalog description. Contains no price or stock.")


class SearchProductsOutput(BaseModel):
    products: list[ProductMatch]


def search_products(retriever: Searcher, args: SearchProductsInput) -> SearchProductsOutput:
    hits = retriever.search(args.query, args.k)
    return SearchProductsOutput(
        products=[ProductMatch(sku=h.sku, name=h.name, text=h.text) for h in hits]
    )


# --- check_stock_and_price --------------------------------------------------------------


class CheckStockAndPriceInput(_Input):
    sku: str = Field(min_length=1, max_length=64, description="Product code, e.g. BOIL-80-A.")


class StockAndPrice(BaseModel):
    status: Literal["found"] = "found"
    sku: str
    name: str
    price_eur: Decimal = Field(description="Selling price in EUR (primary currency).")
    price_bgn: Decimal = Field(description="Selling price in BGN (secondary).")
    stock_qty: int
    in_stock: bool


class ProductNotFound(BaseModel):
    status: Literal["not_found"] = "not_found"
    sku: str
    message: str = "No product with this code in the catalog."


# `status` ("found" / "not_found") tells the model which of the two shapes it got.
def check_stock_and_price(
    catalog: Catalog, args: CheckStockAndPriceInput
) -> StockAndPrice | ProductNotFound:
    product = catalog.get(args.sku)
    if product is None:
        return ProductNotFound(sku=args.sku)
    return StockAndPrice(
        sku=product.sku,
        name=product.name,
        price_eur=product.price_eur,
        price_bgn=product.price_bgn,
        stock_qty=product.stock_qty,
        in_stock=product.stock_qty > 0,
    )


# --- compare_supplier_price (operator only: supplier prices are internal cost data) -----


class CompareSupplierPriceInput(_Input):
    # float, not Decimal: a Decimal field exports as `number | string`, a muddled schema for
    # the model. It is converted exactly (via str) before any arithmetic.
    threshold_pct: float = Field(
        ge=0,
        le=100,
        description="Return products whose supplier price is more than this many percent "
        "below our selling price.",
    )


class SupplierGap(BaseModel):
    sku: str
    name: str
    price_bgn: Decimal
    supplier_price_bgn: Decimal
    below_ours_pct: Decimal = Field(description="(ours - supplier) / ours * 100, 1 decimal.")


class CompareSupplierPriceOutput(BaseModel):
    threshold_pct: Decimal
    products: list[SupplierGap] = Field(description="Largest gap first.")


def below_ours_pct(price: Decimal, supplier_price: Decimal) -> Decimal:
    """How far the supplier's price is below ours, as a percentage of our price."""
    return (price - supplier_price) / price * 100


def compare_supplier_price(
    catalog: Catalog, args: CompareSupplierPriceInput
) -> CompareSupplierPriceOutput:
    threshold = Decimal(str(args.threshold_pct))  # 15.1 -> Decimal("15.1"), not 15.0999…
    gaps = [(below_ours_pct(p.price_bgn, p.supplier_price_bgn), p) for p in catalog.products()]
    # Strictly "more than", compared before rounding.
    selected = sorted(
        ((pct, p) for pct, p in gaps if pct > threshold),
        key=lambda item: (-item[0], item[1].sku),
    )
    return CompareSupplierPriceOutput(
        threshold_pct=threshold,
        products=[
            SupplierGap(
                sku=p.sku,
                name=p.name,
                price_bgn=p.price_bgn,
                supplier_price_bgn=p.supplier_price_bgn,
                below_ours_pct=pct.quantize(Decimal("0.1"), rounding=ROUND_HALF_UP),
            )
            for pct, p in selected
        ],
    )
