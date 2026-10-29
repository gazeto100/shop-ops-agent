"""CSV feed -> validated `Product` records.

The whole file is validated before anything is indexed: one bad row fails the run, and
every problem is reported at once so a broken supplier feed can be fixed in one pass.
"""

import csv
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationError


class Product(BaseModel):
    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    sku: str = Field(min_length=1)
    name: str = Field(min_length=1)
    # Optional column: lets English queries match Bulgarian-named products (#4).
    name_en: str = ""
    category: str = Field(min_length=1)
    price_eur: Decimal = Field(gt=0, decimal_places=2)
    price_bgn: Decimal = Field(gt=0, decimal_places=2)
    stock_qty: int = Field(ge=0)
    supplier_sku: str
    supplier_price_bgn: Decimal = Field(gt=0, decimal_places=2)
    description: str
    warranty_months: int = Field(ge=0)


REQUIRED_COLUMNS = frozenset(n for n, f in Product.model_fields.items() if f.is_required())


def normalize_sku(sku: str) -> str:
    """Identity of a product code: case and surrounding spaces don't matter. Used both for
    the feed's duplicate check and for lookups, so the two can never disagree."""
    return sku.strip().upper()


@dataclass(frozen=True)
class RowError:
    line: int | None  # None: the problem is with the file as a whole
    sku: str | None
    message: str

    def __str__(self) -> str:
        where = "file" if self.line is None else f"line {self.line}"
        if self.sku:
            where += f" (sku {self.sku})"
        return f"{where}: {self.message}"


class ProductValidationError(Exception):
    def __init__(self, errors: list[RowError]) -> None:
        self.errors = errors
        super().__init__("\n".join(map(str, errors)))


def load_products(path: Path) -> list[Product]:
    """Parse and validate the feed. Raises `ProductValidationError` listing every bad row.

    OSError (missing/unreadable file) propagates; callers report it.
    """
    try:
        # utf-8-sig: tolerate the BOM that Excel adds when saving CSV as UTF-8.
        with path.open(encoding="utf-8-sig", newline="") as f:
            return _validate(csv.DictReader(f))
    except UnicodeDecodeError as e:
        # Typical for Bulgarian feeds exported from Excel as Windows-1251.
        msg = f"not UTF-8 (invalid byte at offset {e.start}); re-save it as 'CSV UTF-8'"
        raise ProductValidationError([RowError(None, None, msg)]) from e


def _validate(reader: "csv.DictReader[str]") -> list[Product]:
    missing = REQUIRED_COLUMNS - set(reader.fieldnames or [])
    if missing:
        raise ProductValidationError(
            [RowError(1, None, f"missing columns: {', '.join(sorted(missing))}")]
        )

    products: list[Product] = []
    errors: list[RowError] = []
    first_line_of: dict[str, int] = {}
    header_len = len(reader.fieldnames or [])
    for row in reader:
        line = reader.line_num
        sku = (row.get("sku") or "").strip() or None
        # DictReader pads short rows with None and puts surplus fields under key None;
        # either way the columns no longer line up (often an unquoted comma).
        extra = row.pop(None, [])
        n_fields = header_len - list(row.values()).count(None) + len(extra)
        if n_fields != header_len:
            errors.append(RowError(line, sku, f"{n_fields} fields, header has {header_len}"))
            continue
        try:
            product = Product.model_validate(row)
        except ValidationError as e:
            errors.extend(
                RowError(line, sku, f"{'.'.join(map(str, err['loc']))}: {err['msg']}")
                for err in e.errors()
            )
            continue
        key = normalize_sku(product.sku)  # "boil-80-a" duplicates "BOIL-80-A"
        if key in first_line_of:
            msg = f"duplicate sku (first seen on line {first_line_of[key]})"
            errors.append(RowError(line, product.sku, msg))
            continue
        first_line_of[key] = line
        products.append(product)

    if errors:
        raise ProductValidationError(errors)
    if not products:
        # Ingest syncs the index to the feed; an empty feed would wipe the whole catalog.
        raise ProductValidationError([RowError(1, None, "feed contains no products")])
    return products
