"""The agent's tools (#6). `build_registry` wires them to their data sources."""

from functools import partial

from shop_ops_agent.agent.tools.catalog import Catalog
from shop_ops_agent.agent.tools.definitions import (
    CheckStockAndPriceInput,
    CompareSupplierPriceInput,
    SearchProductsInput,
    check_stock_and_price,
    compare_supplier_price,
    search_products,
)
from shop_ops_agent.agent.tools.registry import Audience, Tool, ToolError, ToolRegistry
from shop_ops_agent.rag.retriever import Searcher

__all__ = ["Audience", "Catalog", "ToolError", "ToolRegistry", "build_registry"]


def build_registry(catalog: Catalog, retriever: Searcher) -> ToolRegistry:
    return ToolRegistry(
        [
            Tool(
                name="search_products",
                description="Find catalog products matching a description. Returns codes, "
                "names and descriptions only - use check_stock_and_price for price/stock.",
                input_model=SearchProductsInput,
                run=partial(search_products, retriever),
            ),
            Tool(
                name="check_stock_and_price",
                description="Current selling price (EUR and BGN) and stock for one product "
                "code. Returns status 'not_found' for an unknown code.",
                input_model=CheckStockAndPriceInput,
                run=partial(check_stock_and_price, catalog),
            ),
            Tool(
                name="compare_supplier_price",
                description="Products whose supplier price is more than threshold_pct "
                "percent below our selling price, largest gap first. Internal data.",
                input_model=CompareSupplierPriceInput,
                run=partial(compare_supplier_price, catalog),
                audience="operator",
            ),
        ]
    )
