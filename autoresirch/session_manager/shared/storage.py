from __future__ import annotations

import importlib
import json
import subprocess
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from autoresirch.prepare import (
    ARCHITECTURE_CONSTRAINTS,
    DATASET_CONFIG,
    ExperimentMode,
    METRIC_CONFIG,
    SPLIT_CONFIG,
    TRAINING_CONFIG,
    ExperimentSummary,
    load_train_definition,
)


def _session_manager_package():
    try:
        return importlib.import_module("session_manager")
    except ModuleNotFoundError:
        return importlib.import_module("autoresirch.session_manager")
from autoresirch.session_manager.constants import (
    EDITABLE_TRAIN_FILE,
    PROGRAM_FILE,
    REPO_ROOT,
    SESSION_RESULTS_HEADER,
    SESSIONS_DIR,
)
from autoresirch.session_manager.schemas import (
    DecisionRecord,
    RunContext,
    SessionContext,
    SessionState,
)


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    raise TypeError(f"Value is not JSON serializable: {type(value)!r}")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_json_default),
        encoding="utf-8",
    )


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _utc_now_iso() -> str:
    return _utc_now().isoformat()


def _sha256_bytes(payload: bytes) -> str:
    import hashlib

    return hashlib.sha256(payload).hexdigest()


def _sha256_path(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _fingerprint_payload(payload: Any) -> str:
    return _sha256_bytes(json.dumps(payload, sort_keys=True, default=str).encode("utf-8"))


def _sanitize_tsv(value: Any) -> str:
    if value is None:
        return ""
    return str(value).replace("\t", " ").replace("\n", " ").strip()


def _git_output(*args: str) -> str | None:
    result = subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def _collect_git_metadata() -> tuple[str | None, str | None, bool | None]:
    commit = _git_output("rev-parse", "HEAD")
    branch = _git_output("branch", "--show-current")
    dirty_output = _git_output("status", "--short")
    git_is_dirty = None if dirty_output is None else bool(dirty_output)
    return commit, branch, git_is_dirty


def _load_architecture_metadata() -> tuple[str | None, dict[str, Any] | None]:
    try:
        loaded = load_train_definition()
    except Exception:
        return None, None
    spec_payload = asdict(loaded.spec)
    return _fingerprint_payload(spec_payload), spec_payload


def _session_dir(session_id: str) -> Path:
    return _session_manager_package().SESSIONS_DIR / session_id


def _session_runs_dir(session_id: str) -> Path:
    return _session_dir(session_id) / "runs"


def _session_results_path(session_id: str) -> Path:
    return _session_dir(session_id) / "results.tsv"


def _session_context_path(session_id: str) -> Path:
    return _session_dir(session_id) / "session_context.json"


def _session_state_path(session_id: str) -> Path:
    return _session_dir(session_id) / "session_state.json"


def _session_summary_json_path(session_id: str) -> Path:
    return _session_dir(session_id) / "session_summary.json"


def _session_summary_md_path(session_id: str) -> Path:
    return _session_dir(session_id) / "session_summary.md"


def _analysis_input_path(run_dir: Path) -> Path:
    return run_dir / "analysis_input.json"


def _agent_analysis_path(run_dir: Path) -> Path:
    return run_dir / "agent_analysis.json"


def _generate_session_id() -> str:
    return _utc_now().strftime("%Y%m%d-%H%M%S")


def _run_id_for_index(session_id: str, session_run_index: int) -> str:
    return f"{session_id}-r{session_run_index:03d}"


def _config_fingerprints() -> dict[str, str]:
    return {
        "dataset_config_fingerprint": _fingerprint_payload(asdict(DATASET_CONFIG)),
        "split_config_fingerprint": _fingerprint_payload(asdict(SPLIT_CONFIG)),
        "training_config_fingerprint": _fingerprint_payload(asdict(TRAINING_CONFIG)),
        "metric_config_fingerprint": _fingerprint_payload(asdict(METRIC_CONFIG)),
        "constraints_fingerprint": _fingerprint_payload(asdict(ARCHITECTURE_CONSTRAINTS)),
    }


def create_session(
    *,
    session_id: str | None,
    objective: str,
    initiated_by: str,
    experiment_mode: ExperimentMode,
) -> SessionContext:
    resolved_session_id = session_id or _generate_session_id()
    session_dir = _session_dir(resolved_session_id)
    if session_dir.exists():
        raise ValueError(f"Session {resolved_session_id!r} already exists.")

    _session_manager_package().SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    _session_runs_dir(resolved_session_id).mkdir(parents=True, exist_ok=False)
    _session_results_path(resolved_session_id).write_text(SESSION_RESULTS_HEADER, encoding="utf-8")

    config_fingerprints = _config_fingerprints()
    base_run_id = _run_id_for_index(resolved_session_id, 0)
    context = SessionContext(
        session_id=resolved_session_id,
        base_run_id=base_run_id,
        started_at=_utc_now_iso(),
        objective=objective,
        initiated_by=initiated_by,
        experiment_mode=experiment_mode,
        program_md_sha256=_sha256_path(_session_manager_package().PROGRAM_FILE),
        **config_fingerprints,
    )
    state = SessionState(
        session_id=resolved_session_id,
        status="active",
        started_at=context.started_at,
        objective=objective,
        initiated_by=initiated_by,
        base_run_id=base_run_id,
    )
    save_session_context(context)
    save_session_state(state)
    return context


def load_session_context(session_id: str) -> SessionContext:
    return SessionContext(**_read_json(_session_context_path(session_id)))


def load_session_state(session_id: str) -> SessionState:
    return SessionState(**_read_json(_session_state_path(session_id)))


def save_session_context(context: SessionContext) -> None:
    _write_json(_session_context_path(context.session_id), asdict(context))


def save_session_state(state: SessionState) -> None:
    _write_json(_session_state_path(state.session_id), asdict(state))


def allocate_run_dir(
    context: SessionContext,
    state: SessionState,
    *,
    run_role: str,
) -> tuple[str, int, Path]:
    session_run_index = len(state.ordered_run_ids)
    if run_role == "base":
        if session_run_index != 0:
            raise ValueError("A session can only have one base run, and it must be first.")
        run_id = context.base_run_id
    else:
        run_id = _run_id_for_index(context.session_id, session_run_index)
    run_dir = _session_runs_dir(context.session_id) / f"{session_run_index:03d}_{run_role}__{run_id}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_id, session_run_index, run_dir


def write_train_snapshot(run_dir: Path) -> Path:
    snapshot_path = run_dir / "train_snapshot.py"
    snapshot_path.write_text(
        _session_manager_package().EDITABLE_TRAIN_FILE.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    return snapshot_path


def write_run_context(run_context: RunContext) -> None:
    _write_json(Path(run_context.run_dir) / "run_context.json", asdict(run_context))


def write_decision_record(run_dir: Path, record: DecisionRecord) -> None:
    _write_json(run_dir / "decision.json", asdict(record))


def append_session_results_row(
    session_id: str,
    run_context: RunContext,
    decision: DecisionRecord,
    summary: ExperimentSummary | None,
) -> None:
    row = "\t".join(
        [
            _sanitize_tsv(session_id),
            _sanitize_tsv(run_context.session_run_index),
            _sanitize_tsv(run_context.run_id),
            _sanitize_tsv(run_context.run_role),
            _sanitize_tsv(run_context.parent_run_id),
            _sanitize_tsv(run_context.compared_against_run_id),
            _sanitize_tsv(run_context.git_commit),
            _sanitize_tsv(None if summary is None else summary.experiment_mode),
            _sanitize_tsv(None if summary is None else summary.primary_metric_name),
            _sanitize_tsv(None if summary is None else summary.primary_metric_value),
            _sanitize_tsv(None if summary is None else summary.weighted_cv_rmse_mean),
            _sanitize_tsv(None if summary is None else summary.cv_rmse_std),
            _sanitize_tsv(None if summary is None else summary.weighted_cv_auc_mean),
            _sanitize_tsv(None if summary is None else summary.weighted_cv_overall_auc),
            _sanitize_tsv(None if summary is None else summary.weighted_cv_auc_pos_vs_neg),
            _sanitize_tsv(None if summary is None else summary.weighted_cv_pearson_r_mean),
            _sanitize_tsv(None if summary is None else summary.weighted_cv_spearman_r_mean),
            _sanitize_tsv(decision.decision_status),
            _sanitize_tsv(None if summary is None else summary.num_params),
            _sanitize_tsv(None if summary is None else summary.train_seconds),
            _sanitize_tsv(decision.decision_baseline_value),
            _sanitize_tsv(decision.decision_delta),
            _sanitize_tsv(run_context.hypothesis),
            _sanitize_tsv(run_context.mutation_summary),
            _sanitize_tsv(run_context.description),
            _sanitize_tsv(run_context.run_dir),
        ]
    )
    with _session_results_path(session_id).open("a", encoding="utf-8") as handle:
        handle.write(f"{row}\n")


def _load_run_summary(run_dir: Path) -> dict[str, Any]:
    return _read_json(run_dir / "summary.json")
