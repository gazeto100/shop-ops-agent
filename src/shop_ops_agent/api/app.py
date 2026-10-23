"""FastAPI entrypoint. Issue #1: health only. Issue #5 adds /chat, #3 adds /ingest."""

from fastapi import FastAPI

from shop_ops_agent import __version__

app = FastAPI(title="shop-ops-agent", version=__version__)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "version": __version__}
