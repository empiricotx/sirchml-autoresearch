from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict


class HealthResponse(BaseModel):
    status: str
    sessions_root: str


class SessionListItem(BaseModel):
    model_config = ConfigDict(extra="allow")

    session_id: str
    status: str | None = None
    objective: str | None = None
    started_at: str | None = None
    completed_at: str | None = None
    total_runs: int | None = None
    latest_run_id: str | None = None
    incumbent_run_id: str | None = None
    final_incumbent_run_id: str | None = None
    best_run_id: str | None = None
    best_primary_metric_value: float | None = None
    keep_count: int | None = None
    discard_count: int | None = None
    crash_count: int | None = None


class SessionListResponse(BaseModel):
    items: list[SessionListItem]


class SessionDetailResponse(BaseModel):
    session_id: str
    context: dict[str, Any] | None = None
    state: dict[str, Any] | None = None
    summary: dict[str, Any] | None = None
    summary_markdown: str | None = None
    runs: list[dict[str, Any]]


class RunDetailResponse(BaseModel):
    session_id: str
    run_id: str
    run_dir: str | None = None
    run_context: dict[str, Any] | None = None
    decision: dict[str, Any] | None = None
    summary: dict[str, Any] | None = None
    analysis_input: dict[str, Any] | None = None
    agent_analysis: dict[str, Any] | None = None
    synopsis_markdown: str | None = None
