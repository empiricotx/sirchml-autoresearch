from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any


DEFAULT_SESSIONS_ROOT = Path("/workspace/sessions")


def get_sessions_root() -> Path:
    configured = os.environ.get("SESSIONS_ROOT")
    if not configured:
        return DEFAULT_SESSIONS_ROOT
    return Path(configured).expanduser().resolve()


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _read_markdown(path: Path) -> str | None:
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8")


def _first_non_none(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _session_dir(session_id: str) -> Path:
    return get_sessions_root() / session_id


def _results_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []

    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for raw_row in reader:
            rows.append(
                {
                    key: value if value != "" else None
                    for key, value in raw_row.items()
                }
            )
    return rows


def _resolve_run_dir(session_id: str, run_id: str, state: dict[str, Any]) -> Path | None:
    session_dir = _session_dir(session_id)

    configured_run_dir = state.get("run_dirs", {}).get(run_id)
    if configured_run_dir:
        candidate = Path(configured_run_dir)
        if candidate.exists():
            return candidate

    runs_dir = session_dir / "runs"
    if runs_dir.exists():
        for candidate_dir in runs_dir.glob(f"*__{run_id}"):
            return candidate_dir

    return None


def _coerce_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _primary_metric_name_for_mode(experiment_mode: Any) -> str:
    if experiment_mode == "comparative":
        return "weighted_cv_overall_auc"
    return "weighted_cv_auc"


def list_sessions() -> list[dict[str, Any]]:
    sessions_root = get_sessions_root()
    if not sessions_root.exists():
        return []

    session_items: list[dict[str, Any]] = []
    for session_dir in sessions_root.iterdir():
        if not session_dir.is_dir():
            continue

        session_id = session_dir.name
        context = _read_json(session_dir / "session_context.json") or {}
        state = _read_json(session_dir / "session_state.json") or {}
        summary = _read_json(session_dir / "session_summary.json") or {}
        experiment_mode = context.get("experiment_mode")

        total_runs = _first_non_none(summary.get("total_runs"), len(state.get("ordered_run_ids", [])))

        item = {
            "session_id": session_id,
            "experiment_mode": experiment_mode,
            "primary_metric_name": _primary_metric_name_for_mode(experiment_mode),
            "status": _first_non_none(state.get("status"), summary.get("status")),
            "objective": _first_non_none(state.get("objective"), summary.get("objective")),
            "started_at": _first_non_none(state.get("started_at"), summary.get("started_at")),
            "completed_at": _first_non_none(state.get("completed_at"), summary.get("completed_at")),
            "total_runs": _coerce_int(total_runs),
            "latest_run_id": state.get("latest_run_id"),
            "incumbent_run_id": state.get("incumbent_run_id"),
            "final_incumbent_run_id": _first_non_none(
                state.get("final_incumbent_run_id"),
                summary.get("final_incumbent_run_id"),
            ),
            "best_run_id": summary.get("best_run_id"),
            "best_primary_metric_value": _coerce_float(
                _first_non_none(summary.get("best_primary_metric_value"), state.get("incumbent_primary_metric_value"))
            ),
            "keep_count": _coerce_int(_first_non_none(summary.get("keep_count"), state.get("keep_count"))),
            "discard_count": _coerce_int(_first_non_none(summary.get("discard_count"), state.get("discard_count"))),
            "crash_count": _coerce_int(_first_non_none(summary.get("crash_count"), state.get("crash_count"))),
        }
        session_items.append(item)

    session_items.sort(key=lambda item: item["session_id"], reverse=True)
    return session_items


def get_session_detail(session_id: str) -> dict[str, Any]:
    session_dir = _session_dir(session_id)
    context = _read_json(session_dir / "session_context.json")
    state = _read_json(session_dir / "session_state.json")
    summary = _read_json(session_dir / "session_summary.json")
    summary_markdown = _read_markdown(session_dir / "session_summary.md")
    runs = list_session_runs(session_id)
    return {
        "session_id": session_id,
        "context": context,
        "state": state,
        "summary": summary,
        "summary_markdown": summary_markdown,
        "runs": runs,
    }


def list_session_runs(session_id: str) -> list[dict[str, Any]]:
    session_dir = _session_dir(session_id)
    rows = _results_rows(session_dir / "results.tsv")
    state = _read_json(session_dir / "session_state.json") or {}

    enriched_rows: list[dict[str, Any]] = []
    incumbent_run_id = state.get("incumbent_run_id")
    final_incumbent_run_id = state.get("final_incumbent_run_id")

    for row in rows:
        enriched_rows.append(
            {
                **row,
                "session_run_index": _coerce_int(row.get("session_run_index")),
                "primary_metric_name": row.get("primary_metric_name"),
                "primary_metric_value": _coerce_float(row.get("primary_metric_value")),
                "weighted_cv_auc": _coerce_float(row.get("weighted_cv_auc")),
                "weighted_cv_overall_auc": _coerce_float(row.get("weighted_cv_overall_auc")),
                "weighted_cv_auc_pos_vs_neg": _coerce_float(row.get("weighted_cv_auc_pos_vs_neg")),
                "weighted_cv_rmse_mean": _coerce_float(row.get("weighted_cv_rmse_mean")),
                "cv_rmse_std": _coerce_float(row.get("cv_rmse_std")),
                "weighted_cv_pearson_r": _coerce_float(row.get("weighted_cv_pearson_r")),
                "weighted_cv_spearman_r": _coerce_float(row.get("weighted_cv_spearman_r")),
                "num_params": _coerce_int(row.get("num_params")),
                "train_seconds": _coerce_float(row.get("train_seconds")),
                "decision_baseline_value": _coerce_float(row.get("decision_baseline_value")),
                "decision_delta": _coerce_float(row.get("decision_delta")),
                "is_current_incumbent": row.get("run_id") == incumbent_run_id,
                "is_final_incumbent": row.get("run_id") == final_incumbent_run_id,
            }
        )

    enriched_rows.sort(key=lambda row: row.get("session_run_index") or -1)
    return enriched_rows


def get_run_detail(session_id: str, run_id: str) -> dict[str, Any]:
    session_dir = _session_dir(session_id)
    state = _read_json(session_dir / "session_state.json") or {}
    run_dir = _resolve_run_dir(session_id, run_id, state)
    if run_dir is None:
        raise FileNotFoundError(f"Run {run_id!r} was not found in session {session_id!r}.")

    summary_payload = _read_json(run_dir / "summary.json")
    summary_metrics = (summary_payload or {}).get("summary") or {}
    analysis_input = _read_json(run_dir / "analysis_input.json")
    agent_analysis = _read_json(run_dir / "agent_analysis.json")
    return {
        "session_id": session_id,
        "run_id": run_id,
        "run_dir": str(run_dir),
        "run_context": _read_json(run_dir / "run_context.json"),
        "decision": _read_json(run_dir / "decision.json"),
        "summary": summary_payload,
        "primary_metric_name": summary_metrics.get("primary_metric_name"),
        "primary_metric_value": _coerce_float(summary_metrics.get("primary_metric_value")),
        "analysis_input": analysis_input,
        "agent_analysis": agent_analysis,
        "synopsis_markdown": _read_markdown(run_dir / "synopsis.md"),
    }
