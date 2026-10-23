"""FastAPI entrypoint. Issue #1: health only. Issue #5 adds /chat, #3 adds /ingest."""

from fastapi import FastAPI

from shop_ops_agent import __version__
from shop_ops_agent.api.schemas import HealthResponse

app = FastAPI(title="shop-ops-agent", version=__version__)


@app.get("/health")
async def health() -> HealthResponse:
    return HealthResponse(status="ok", version=__version__)
