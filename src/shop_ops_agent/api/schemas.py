"""Request/response models for the HTTP API."""

from typing import Any, Literal

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: Literal["ok"]
    version: str


class ChatRequest(BaseModel):
    # Omit on the first message; then send back the session_id the response returned.
    # Ids are issued by the server: an unknown one simply starts a new session.
    session_id: str | None = Field(default=None, min_length=1, max_length=128)
    message: str = Field(min_length=1, max_length=4000)


class Citation(BaseModel):
    sku: str
    name: str


class ToolCallOut(BaseModel):
    name: str
    arguments: dict[str, Any]


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    citations: list[Citation]
    tool_calls: list[ToolCallOut]
