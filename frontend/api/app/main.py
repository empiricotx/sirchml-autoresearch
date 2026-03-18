from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.readers import get_run_detail, get_session_detail, get_sessions_root, list_session_runs, list_sessions
from app.schemas import HealthResponse, RunDetailResponse, SessionDetailResponse, SessionListResponse


app = FastAPI(title="sirchml frontend api", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", sessions_root=str(get_sessions_root()))


@app.get("/sessions", response_model=SessionListResponse)
def sessions() -> SessionListResponse:
    return SessionListResponse(items=list_sessions())


@app.get("/sessions/{session_id}", response_model=SessionDetailResponse)
def session_detail(session_id: str) -> SessionDetailResponse:
    detail = get_session_detail(session_id)
    if detail["context"] is None and detail["state"] is None and detail["summary"] is None:
        raise HTTPException(status_code=404, detail=f"Session {session_id!r} was not found.")
    return SessionDetailResponse(**detail)


@app.get("/sessions/{session_id}/runs")
def session_runs(session_id: str) -> list[dict]:
    runs = list_session_runs(session_id)
    if not runs and not (get_sessions_root() / session_id).exists():
        raise HTTPException(status_code=404, detail=f"Session {session_id!r} was not found.")
    return runs


@app.get("/sessions/{session_id}/runs/{run_id}", response_model=RunDetailResponse)
def run_detail(session_id: str, run_id: str) -> RunDetailResponse:
    try:
        return RunDetailResponse(**get_run_detail(session_id, run_id))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
