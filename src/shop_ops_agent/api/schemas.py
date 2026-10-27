"""Request/response models for the HTTP API."""

from typing import Any, Literal

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: Literal["ok"]
    version: str


class ChatRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=128)  # memory arrives in #7
    message: str = Field(min_length=1, max_length=4000)


class Citation(BaseModel):
    sku: str
    name: str


class ToolCallOut(BaseModel):
    name: str
    arguments: dict[str, Any]


class ChatResponse(BaseModel):
    answer: str
    citations: list[Citation]
    tool_calls: list[ToolCallOut]
