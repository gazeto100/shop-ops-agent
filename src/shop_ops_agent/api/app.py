"""FastAPI entrypoint: /health and /chat (#5)."""

from functools import lru_cache
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI

from shop_ops_agent import __version__
from shop_ops_agent.agent.chat import ChatService
from shop_ops_agent.api.schemas import (
    ChatRequest,
    ChatResponse,
    Citation,
    HealthResponse,
    ToolCallOut,
)
from shop_ops_agent.config import get_settings
from shop_ops_agent.embeddings import SentenceTransformerEmbedder
from shop_ops_agent.ingest.index import open_collection
from shop_ops_agent.llm import create_llm
from shop_ops_agent.rag.retriever import Retriever

app = FastAPI(title="shop-ops-agent", version=__version__)


@lru_cache
def get_chat_service() -> ChatService:
    """Built on the first /chat request, not at import: loading the embedding model takes
    seconds and /health must not need it. Tests replace it via dependency_overrides."""
    settings = get_settings()
    embedder = SentenceTransformerEmbedder(settings.embedding_model)
    collection = open_collection(Path(settings.chroma_path), settings.embedding_model)
    return ChatService(Retriever(collection, embedder), create_llm(settings))


@app.get("/health")
async def health() -> HealthResponse:
    return HealthResponse(status="ok", version=__version__)


@app.post("/chat")
async def chat(
    request: ChatRequest, service: Annotated[ChatService, Depends(get_chat_service)]
) -> ChatResponse:
    result = await service.answer(request.message)
    return ChatResponse(
        answer=result.answer,
        citations=[Citation(sku=h.sku, name=h.name) for h in result.citations],
        tool_calls=[ToolCallOut(name=c.name, arguments=c.arguments) for c in result.tool_calls],
    )
