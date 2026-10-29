"""The live price/stock source behind the tools: the product feed CSV.

Price and stock are deliberately not in the vector index (#3); tools read them here.
The file is re-read whenever its (modification time, size) changes, so a new feed is
picked up without a restart. Once a good feed is loaded, the tools keep serving it if a
new feed fails validation or the file is briefly unreadable (e.g. a non-atomic replace),
so one bad upload can't take the tools down. Replace feeds atomically where possible
(write a temp file, then `os.replace`).
"""

import logging
from pathlib import Path

from shop_ops_agent.ingest.products import (
    Product,
    ProductValidationError,
    load_products,
    normalize_sku,
)

log = logging.getLogger(__name__)


class Catalog:
    def __init__(self, path: Path) -> None:
        self._path = path
        # (st_mtime_ns, st_size) of the file last parsed. Size too, because copies that
        # preserve the modification time (rsync -t, cp -p) would otherwise go unnoticed.
        self._version: tuple[int, int] | None = None
        self._by_sku: dict[str, Product] | None = None  # None until the first good load

    def get(self, sku: str) -> Product | None:
        return self._current().get(normalize_sku(sku))

    def products(self) -> list[Product]:
        return list(self._current().values())

    def _current(self) -> dict[str, Product]:
        try:
            stat = self._path.stat()
        except OSError as e:
            return self._previous_or_raise(e)
        version = (stat.st_mtime_ns, stat.st_size)
        if version == self._version and self._by_sku is not None:
            return self._by_sku
        try:
            products = load_products(self._path)
        except OSError as e:  # vanished between stat() and open(): retry on the next call
            return self._previous_or_raise(e)
        except ProductValidationError as e:
            self._version = version  # don't re-parse the same bad file on every call
            return self._previous_or_raise(e)
        self._by_sku = {normalize_sku(p.sku): p for p in products}
        self._version = version
        return self._by_sku

    def _previous_or_raise(self, error: Exception) -> dict[str, Product]:
        if self._by_sku is None:
            raise error  # nothing good to fall back to
        log.error("catalog %s not updated, serving previous version: %s", self._path, error)
        return self._by_sku
