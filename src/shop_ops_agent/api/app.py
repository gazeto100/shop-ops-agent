"""FastAPI entrypoint: /health and /chat (#5; tools and session memory #7)."""

import logging
from functools import lru_cache
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse

from shop_ops_agent import __version__
from shop_ops_agent.agent.chat import ChatService
from shop_ops_agent.agent.memory import SessionMemory
from shop_ops_agent.agent.tools import Catalog, build_registry
from shop_ops_agent.api.schemas import (
    ChatRequest,
    ChatResponse,
    Citation,
    HealthResponse,
    ToolCallOut,
)
from shop_ops_agent.config import get_settings
from shop_ops_agent.embeddings import SentenceTransformerEmbedder
from shop_ops_agent.ingest.index import IndexModelMismatchError, open_collection
from shop_ops_agent.llm import LLMConfigError, LLMUnavailableError, create_llm
from shop_ops_agent.rag.retriever import Retriever

log = logging.getLogger(__name__)

app = FastAPI(title="shop-ops-agent", version=__version__)

# Used when a rate-limited provider didn't say how long to wait.
DEFAULT_RETRY_AFTER_S = 30


@lru_cache
def get_chat_service() -> ChatService:
    """Built on the first /chat request, not at import: loading the embedding model takes
    seconds and /health must not need it. Tests replace it via dependency_overrides.

    Cheap checks run first (provider config, index model), so a misconfiguration fails
    before the embedding model loads; lru_cache doesn't cache exceptions, so a failing
    build is retried on every request and must stay cheap."""
    settings = get_settings()
    llm = create_llm(settings)
    collection = open_collection(Path(settings.chroma_path), settings.embedding_model)
    embedder = SentenceTransformerEmbedder(settings.embedding_model)
    retriever = Retriever(collection, embedder)
    registry = build_registry(Catalog(Path(settings.catalog_path)), retriever)
    # /chat has no authentication yet, so every caller is a customer: operator-only tools
    # (supplier prices) are never offered or executed here.
    return ChatService(retriever, llm, registry, SessionMemory(), audience="customer")


# Clients get a generic message; the real cause (which may name env vars or paths) is
# logged server-side only.
@app.exception_handler(LLMConfigError)
@app.exception_handler(IndexModelMismatchError)
async def misconfigured(_: Request, exc: Exception) -> JSONResponse:
    log.error("chat service misconfigured: %s", exc)
    return JSONResponse(status_code=503, content={"detail": "Chat service is not configured."})


@app.exception_handler(LLMUnavailableError)
async def llm_unavailable(_: Request, exc: LLMUnavailableError) -> JSONResponse:
    log.warning("LLM provider unavailable: %s", exc)
    retry_after = exc.retry_after if exc.retry_after is not None else DEFAULT_RETRY_AFTER_S
    return JSONResponse(
        status_code=503,
        content={"detail": "The language model is unavailable, please try again later."},
        headers={"Retry-After": str(retry_after)},
    )


@app.get("/health")
async def health() -> HealthResponse:
    return HealthResponse(status="ok", version=__version__)


@app.post("/chat")
async def chat(
    request: ChatRequest, service: Annotated[ChatService, Depends(get_chat_service)]
) -> ChatResponse:
    result = await service.answer(request.session_id, request.message)
    return ChatResponse(
        answer=result.answer,
        citations=[Citation(sku=h.sku, name=h.name) for h in result.citations],
        tool_calls=[ToolCallOut(name=c.name, arguments=c.arguments) for c in result.tool_calls],
    )
