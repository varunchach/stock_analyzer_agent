"""Pydantic request/response contracts. The API never returns free-form,
unvalidated shapes — every response matches one of these models.
"""
from typing import Any, Literal

from pydantic import BaseModel, Field


class AnalyzeRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500, description="Natural-language stock question")
    thread_id: str | None = Field(default=None, description="Reuse to continue a prior conversation")


class GuardrailEvent(BaseModel):
    stage: str
    result: str
    detail: Any = None


class AnalyzeResponse(BaseModel):
    trace_id: str
    query: str
    answer: str
    tools_used: list[str] = []
    guardrail_events: list[GuardrailEvent] = []
    disclaimer: str
    status: Literal["ok", "blocked", "error"]


class HealthResponse(BaseModel):
    status: str
    llm_model: str
    guard_model: str
