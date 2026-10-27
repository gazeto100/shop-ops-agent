"""The live price/stock source behind the tools: the product feed CSV.

Price and stock are deliberately not in the vector index (#3); tools read them here.
The file is re-read whenever its modification time changes, so a new feed is picked up
without a restart. A new feed that fails validation is logged and the last good one is
kept, so one bad upload can't take the tools down.
"""

import logging
from pathlib import Path

from shop_ops_agent.ingest.products import Product, ProductValidationError, load_products

log = logging.getLogger(__name__)


def normalize_sku(sku: str) -> str:
    """Lookup key: tools receive skus from the model, which may change case or spacing."""
    return sku.strip().upper()


class Catalog:
    def __init__(self, path: Path) -> None:
        self._path = path
        self._mtime_ns: int | None = None
        self._by_sku: dict[str, Product] = {}

    def get(self, sku: str) -> Product | None:
        return self._current().get(normalize_sku(sku))

    def products(self) -> list[Product]:
        return list(self._current().values())

    def _current(self) -> dict[str, Product]:
        mtime_ns = self._path.stat().st_mtime_ns
        if mtime_ns != self._mtime_ns:
            try:
                products = load_products(self._path)
            except ProductValidationError as e:
                if self._mtime_ns is None:
                    raise  # nothing good to fall back to
                log.error("catalog %s rejected, keeping previous version: %s", self._path, e)
            else:
                self._by_sku = {normalize_sku(p.sku): p for p in products}
            # Also on rejection: don't re-parse the same bad file on every call.
            self._mtime_ns = mtime_ns
        return self._by_sku
