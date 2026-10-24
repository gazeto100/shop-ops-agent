"""`python -m shop_ops_agent.ingest <feed.csv>`: validate the feed and sync it into ChromaDB."""

import argparse
import sys
from pathlib import Path

from shop_ops_agent.config import get_settings
from shop_ops_agent.embeddings import SentenceTransformerEmbedder
from shop_ops_agent.ingest.index import open_collection, sync_products
from shop_ops_agent.ingest.products import ProductValidationError, load_products


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m shop_ops_agent.ingest")
    parser.add_argument("csv", type=Path, help="product feed (CSV)")
    args = parser.parse_args(argv)

    # Validate first: a bad feed fails fast, before the embedding model is loaded.
    try:
        products = load_products(args.csv)
    except ProductValidationError as e:
        print(f"{args.csv}: {len(e.errors)} error(s), nothing indexed:", file=sys.stderr)
        for err in e.errors:
            print(f"  {err}", file=sys.stderr)
        return 1

    settings = get_settings()
    collection = open_collection(Path(settings.chroma_path))
    result = sync_products(
        collection, products, SentenceTransformerEmbedder(settings.embedding_model)
    )
    print(
        f"Indexed {result.upserted} products into {settings.chroma_path} "
        f"(removed {result.deleted} stale)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
