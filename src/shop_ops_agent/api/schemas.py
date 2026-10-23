"""Request/response models for the HTTP API."""

from typing import Literal

from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: Literal["ok"]
    version: str
