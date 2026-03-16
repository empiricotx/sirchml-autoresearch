from __future__ import annotations

import argparse
import json
import math
import subprocess
import traceback
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from prepare import (
    ARCHITECTURE_CONSTRAINTS,
    DATASET_CONFIG,
    METRIC_CONFIG,
    SPLIT_CONFIG,
    TRAINING_CONFIG,
    ExperimentSummary,
    load_train_definition,
    run_experiment,
)


REPO_ROOT = Path(__file__).resolve().parent
SESSIONS_DIR = REPO_ROOT / "sessions"
EDITABLE_TRAIN_FILE = REPO_ROOT / "train.py"
PROGRAM_FILE = REPO_ROOT / "program.md"
RUN_LOG = REPO_ROOT / "run.log"

SESSION_RESULTS_HEADER = (
    "session_id\tsession_run_index\trun_id\trun_role\tparent_run_id\tcompared_against_run_id\t"
    "commit\tweighted_cv_rmse_mean\tcv_rmse_std\tweighted_cv_auc\tweighted_cv_pearson_r\t"
    "weighted_cv_spearman_r\tstatus\tnum_params\ttrain_seconds\tdecision_baseline_value\t"
    "decision_delta\thypothesis\tmutation_summary\tdescription\trun_dir\n"
)


@dataclass(frozen=True)
class SessionContext:
    session_id: str
    base_run_id: str
    started_at: str
    objective: str
    initiated_by: str
    dataset_config_fingerprint: str
    split_config_fingerprint: str
    training_config_fingerprint: str
    metric_config_fingerprint: str
    constraints_fingerprint: str
    program_md_sha256: str


@dataclass
class SessionState:
    session_id: str
    status: str
    started_at: str
    objective: str
    initiated_by: str
    base_run_id: str | None = None
    base_architecture_fingerprint: str | None = None
    base_primary_metric_value: float | None = None
    completed_at: str | None = None
    duration_seconds: float | None = None
    end_reason: str | None = None
    ordered_run_ids: list[str] = field(default_factory=list)
    latest_run_id: str | None = None
    incumbent_run_id: str | None = None
    incumbent_primary_metric_value: float | None = None
    final_incumbent_run_id: str | None = None
    incumbent_progression: list[dict[str, Any]] = field(default_factory=list)
    keep_count: int = 0
    discard_count: int = 0
    crash_count: int = 0
    rerun_count: int = 0
    run_dirs: dict[str, str] = field(default_factory=dict)
    run_roles: dict[str, str] = field(default_factory=dict)
    run_statuses: dict[str, str] = field(default_factory=dict)
    run_primary_metric_values: dict[str, float | None] = field(default_factory=dict)


@dataclass(frozen=True)
class RunIntent:
    session_id: str
    run_role: str
    parent_run_id: str | None
    compared_against_run_id: str | None
    hypothesis: str
    mutation_summary: str
    description: str


@dataclass
class RunContext:
    session_id: str
    run_id: str
    session_run_index: int
    run_role: str
    is_base_run: bool
    parent_run_id: str | None
    compared_against_run_id: str | None
    best_known_run_id_at_start: str | None
    hypothesis: str
    mutation_summary: str
    description: str
    started_at: str
    completed_at: str | None
    duration_seconds: float | None
    run_dir: str
    git_commit: str | None
    git_branch: str | None
    git_is_dirty: bool | None
    train_py_sha256: str
    architecture_fingerprint: str | None
    program_md_sha256: str
    architecture_spec: dict[str, Any] | None = None
    summary_path: str | None = None
    failure_path: str | None = None
    hypothesis_result: str | None = None


@dataclass(frozen=True)
class DecisionRecord:
    session_id: str
    run_id: str
    decision_status: str
    decision_metric_name: str
    decision_metric_value: float | None
    decision_baseline_run_id: str | None
    decision_baseline_value: float | None
    decision_delta: float | None
    decision_epsilon: float | None
    decision_reason: str
    incumbent_before_run_id: str | None
    incumbent_after_run_id: str | None
    compared_against_run_id: str | None
    hypothesis_result: str | None


@dataclass(frozen=True)
class SessionSummaryRecord:
    session_id: str
    status: str
    started_at: str
    completed_at: str | None
    duration_seconds: float | None
    objective: str
    initiated_by: str
    end_reason: str | None
    base_run_id: str | None
    best_run_id: str | None
    final_incumbent_run_id: str | None
    total_runs: int
    keep_count: int
    discard_count: int
    crash_count: int
    rerun_count: int
    best_primary_metric_value: float | None
    delta_from_base: float | None
    best_secondary_metric_snapshot: dict[str, float | None]
    most_promising_near_misses: list[dict[str, Any]]
    search_space_coverage: dict[str, Any]
    hypotheses_tested: list[str]
    hypotheses_supported: list[str]
    hypotheses_unsupported: list[str]
    patterns_that_helped: list[str]
    patterns_that_hurt: list[str]
    instability_patterns: list[str]
    unresolved_questions: list[str]
    recommended_next_starting_run_id: str | None
    recommended_next_mutations: list[str]


@dataclass(frozen=True)
class InterpretationMetricSpec:
    summary_attr: str
    display_name: str
    direction: str
    flat_threshold: float


@dataclass(frozen=True)
class MetricDeltaView:
    metric_name: str
    current_value: float | int | None
    delta_vs_compared: float | None
    delta_vs_base: float | None
    compared_label: str
    base_label: str


INTERPRETATION_NEAR_MISS_AUC_DELTA = 0.005
INTERPRETATION_AUC_SPAN_DELTA = 0.08
INTERPRETATION_RMSE_SPAN_DELTA = 0.05

INTERPRETATION_METRIC_SPECS: dict[str, InterpretationMetricSpec] = {
    "weighted_cv_auc": InterpretationMetricSpec(
        summary_attr="primary_metric_value",
        display_name="Weighted CV AUC",
        direction="higher",
        flat_threshold=max(METRIC_CONFIG.improvement_epsilon * 10.0, 0.001),
    ),
    "weighted_cv_rmse_mean": InterpretationMetricSpec(
        summary_attr="weighted_cv_rmse_mean",
        display_name="Weighted CV RMSE",
        direction="lower",
        flat_threshold=0.002,
    ),
    "weighted_cv_pearson_r_mean": InterpretationMetricSpec(
        summary_attr="weighted_cv_pearson_r_mean",
        display_name="Weighted CV Pearson",
        direction="higher",
        flat_threshold=0.01,
    ),
    "weighted_cv_spearman_r_mean": InterpretationMetricSpec(
        summary_attr="weighted_cv_spearman_r_mean",
        display_name="Weighted CV Spearman",
        direction="higher",
        flat_threshold=0.01,
    ),
    "cv_rmse_std": InterpretationMetricSpec(
        summary_attr="cv_rmse_std",
        display_name="CV RMSE std",
        direction="lower",
        flat_threshold=0.01,
    ),
    "num_params": InterpretationMetricSpec(
        summary_attr="num_params",
        display_name="Parameter count",
        direction="lower",
        flat_threshold=1.0,
    ),
    "train_seconds": InterpretationMetricSpec(
        summary_attr="train_seconds",
        display_name="Train seconds",
        direction="lower",
        flat_threshold=1.0,
    ),
}

INTERPRETATION_METRIC_ORDER = (
    "weighted_cv_auc",
    "weighted_cv_rmse_mean",
    "weighted_cv_pearson_r_mean",
    "weighted_cv_spearman_r_mean",
    "cv_rmse_std",
    "num_params",
    "train_seconds",
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
    return SESSIONS_DIR / session_id


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
) -> SessionContext:
    resolved_session_id = session_id or _generate_session_id()
    session_dir = _session_dir(resolved_session_id)
    if session_dir.exists():
        raise ValueError(f"Session {resolved_session_id!r} already exists.")

    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
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
        program_md_sha256=_sha256_path(PROGRAM_FILE),
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
    snapshot_path.write_text(EDITABLE_TRAIN_FILE.read_text(encoding="utf-8"), encoding="utf-8")
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
            _sanitize_tsv(None if summary is None else summary.weighted_cv_rmse_mean),
            _sanitize_tsv(None if summary is None else summary.cv_rmse_std),
            _sanitize_tsv(None if summary is None else summary.primary_metric_value),
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


def _metric_delta(current: float | None, baseline: float | None) -> float | None:
    if not _is_defined_number(current) or not _is_defined_number(baseline):
        return None
    return current - baseline


def _is_defined_number(value: float | int | None) -> bool:
    if value is None:
        return False
    if isinstance(value, float) and math.isnan(value):
        return False
    return True


def _format_metric(value: float | int | None) -> str:
    if not _is_defined_number(value):
        return "n/a"
    return f"{float(value):.6f}"


def _format_metric_delta(value: float | None) -> str:
    if not _is_defined_number(value):
        return "n/a"
    return f"{float(value):+.6f}"


def _format_metric_value(metric_name: str, value: float | int | None) -> str:
    if not _is_defined_number(value):
        return "n/a"
    if metric_name == "num_params":
        return str(int(value))
    if metric_name == "train_seconds":
        return f"{float(value):.1f}"
    return f"{float(value):.6f}"


def _format_metric_delta_value(metric_name: str, value: float | None) -> str:
    if not _is_defined_number(value):
        return "n/a"
    if metric_name == "num_params":
        return f"{int(value):+d}"
    if metric_name == "train_seconds":
        return f"{float(value):+.1f}"
    return f"{float(value):+.6f}"


def _run_summary_payload_for_run_id(
    state: SessionState,
    run_id: str | None,
) -> dict[str, Any] | None:
    if run_id is None:
        return None
    run_dir_value = state.run_dirs.get(run_id)
    if run_dir_value is None:
        return None
    summary_path = Path(run_dir_value) / "summary.json"
    if not summary_path.exists():
        return None
    return _read_json(summary_path)


def _compared_summary_payload(
    state: SessionState,
    run_context: RunContext,
) -> dict[str, Any] | None:
    compared_run_id = run_context.compared_against_run_id or run_context.best_known_run_id_at_start
    return _run_summary_payload_for_run_id(state, compared_run_id)


def _session_base_summary_payload(state: SessionState) -> dict[str, Any] | None:
    return _run_summary_payload_for_run_id(state, state.base_run_id)


def _summary_metric_value(
    summary: ExperimentSummary | None,
    metric_name: str,
) -> float | int | None:
    if summary is None:
        return None
    metric_spec = INTERPRETATION_METRIC_SPECS[metric_name]
    return getattr(summary, metric_spec.summary_attr)


def _classify_metric_direction(metric_name: str, delta: float | None) -> str:
    if not _is_defined_number(delta):
        return "undefined"
    metric_spec = INTERPRETATION_METRIC_SPECS[metric_name]
    if float(delta) > metric_spec.flat_threshold:
        return "better" if metric_spec.direction == "higher" else "worse"
    if float(delta) < -metric_spec.flat_threshold:
        return "worse" if metric_spec.direction == "higher" else "better"
    return "flat"


def _build_metric_delta_bundle(
    current_summary: ExperimentSummary,
    compared_summary: ExperimentSummary | None,
    base_summary: ExperimentSummary | None,
) -> dict[str, MetricDeltaView]:
    metric_bundle: dict[str, MetricDeltaView] = {}
    for metric_name in INTERPRETATION_METRIC_ORDER:
        current_value = _summary_metric_value(current_summary, metric_name)
        compared_value = _summary_metric_value(compared_summary, metric_name)
        base_value = _summary_metric_value(base_summary, metric_name)
        delta_vs_compared = _metric_delta(
            None if current_value is None else float(current_value),
            None if compared_value is None else float(compared_value),
        )
        delta_vs_base = _metric_delta(
            None if current_value is None else float(current_value),
            None if base_value is None else float(base_value),
        )
        metric_bundle[metric_name] = MetricDeltaView(
            metric_name=metric_name,
            current_value=current_value,
            delta_vs_compared=delta_vs_compared,
            delta_vs_base=delta_vs_base,
            compared_label=_classify_metric_direction(metric_name, delta_vs_compared),
            base_label=_classify_metric_direction(metric_name, delta_vs_base),
        )
    return metric_bundle


def _format_metric_movement_line(metric_movement: MetricDeltaView) -> str:
    metric_spec = INTERPRETATION_METRIC_SPECS[metric_movement.metric_name]
    return (
        f"- {metric_spec.display_name}: "
        f"`{_format_metric_value(metric_movement.metric_name, metric_movement.current_value)}`"
        f" | vs compared "
        f"`{_format_metric_delta_value(metric_movement.metric_name, metric_movement.delta_vs_compared)}`"
        f" (`{metric_movement.compared_label}`)"
        f" | vs base "
        f"`{_format_metric_delta_value(metric_movement.metric_name, metric_movement.delta_vs_base)}`"
        f" (`{metric_movement.base_label}`)"
    )


def _diagnostic_metric_count(
    diagnostics: dict[str, Any] | None,
    metric_name: str,
) -> int:
    if diagnostics is None:
        return 0
    nan_metric_counts = diagnostics.get("nan_metric_counts")
    if not isinstance(nan_metric_counts, dict):
        return 0
    value = nan_metric_counts.get(metric_name, 0)
    return int(value) if isinstance(value, (int, float)) else 0


def _fold_metric_span(
    diagnostics: dict[str, Any] | None,
    *,
    best_key: str,
    worst_key: str,
    metric_name: str,
) -> float | None:
    if diagnostics is None:
        return None
    best_fold = diagnostics.get(best_key)
    worst_fold = diagnostics.get(worst_key)
    if not isinstance(best_fold, dict) or not isinstance(worst_fold, dict):
        return None
    best_value = best_fold.get(metric_name)
    worst_value = worst_fold.get(metric_name)
    if not _is_defined_number(best_value) or not _is_defined_number(worst_value):
        return None
    return abs(float(best_value) - float(worst_value))


def _summary_secondary_metrics(summary: ExperimentSummary) -> dict[str, float | None]:
    return {
        "weighted_cv_rmse_mean": summary.weighted_cv_rmse_mean,
        "weighted_cv_mae_mean": summary.weighted_cv_mae_mean,
        "weighted_cv_r2_mean": summary.weighted_cv_r2_mean,
        "weighted_cv_pearson_r_mean": summary.weighted_cv_pearson_r_mean,
        "weighted_cv_spearman_r_mean": summary.weighted_cv_spearman_r_mean,
        "pooled_cv_rmse": summary.pooled_cv_rmse,
    }


def _is_near_miss_discard(
    decision: DecisionRecord,
    metric_bundle: dict[str, MetricDeltaView],
) -> bool:
    if decision.decision_status != "discard":
        return False
    auc_delta = metric_bundle["weighted_cv_auc"].delta_vs_compared
    if not _is_defined_number(auc_delta):
        return False
    if float(auc_delta) >= 0:
        return False
    if abs(float(auc_delta)) > INTERPRETATION_NEAR_MISS_AUC_DELTA:
        return False
    secondary_better_count = sum(
        metric_bundle[metric_name].compared_label == "better"
        for metric_name in (
            "weighted_cv_rmse_mean",
            "weighted_cv_pearson_r_mean",
            "weighted_cv_spearman_r_mean",
        )
    )
    return secondary_better_count >= 2


def _classify_hypothesis_result(
    decision: DecisionRecord,
    *,
    current_summary: ExperimentSummary,
    compared_summary: ExperimentSummary | None,
) -> str:
    if decision.decision_status == "crash":
        return "inconclusive"
    if compared_summary is None:
        return "inconclusive"
    if decision.decision_status == "keep":
        return "supported"

    metric_bundle = _build_metric_delta_bundle(current_summary, compared_summary, None)
    auc_label = metric_bundle["weighted_cv_auc"].compared_label
    rmse_label = metric_bundle["weighted_cv_rmse_mean"].compared_label
    pearson_label = metric_bundle["weighted_cv_pearson_r_mean"].compared_label
    spearman_label = metric_bundle["weighted_cv_spearman_r_mean"].compared_label

    if _is_near_miss_discard(decision, metric_bundle):
        return "partially_supported"
    if auc_label in {"worse", "flat"} and rmse_label == "better":
        return "partially_supported"
    if auc_label in {"worse", "flat"} and (
        pearson_label == "better" or spearman_label == "better"
    ):
        return "partially_supported"
    return "unsupported"


def _enrich_decision_record(
    decision: DecisionRecord,
    *,
    current_summary: ExperimentSummary,
    compared_summary: ExperimentSummary | None,
) -> DecisionRecord:
    hypothesis_result = _classify_hypothesis_result(
        decision,
        current_summary=current_summary,
        compared_summary=compared_summary,
    )
    if hypothesis_result == decision.hypothesis_result:
        return decision
    return DecisionRecord(
        session_id=decision.session_id,
        run_id=decision.run_id,
        decision_status=decision.decision_status,
        decision_metric_name=decision.decision_metric_name,
        decision_metric_value=decision.decision_metric_value,
        decision_baseline_run_id=decision.decision_baseline_run_id,
        decision_baseline_value=decision.decision_baseline_value,
        decision_delta=decision.decision_delta,
        decision_epsilon=decision.decision_epsilon,
        decision_reason=decision.decision_reason,
        incumbent_before_run_id=decision.incumbent_before_run_id,
        incumbent_after_run_id=decision.incumbent_after_run_id,
        compared_against_run_id=decision.compared_against_run_id,
        hypothesis_result=hypothesis_result,
    )


def _build_robustness_interpretation(
    *,
    metric_bundle: dict[str, MetricDeltaView],
    current_diagnostics: dict[str, Any] | None,
    compared_diagnostics: dict[str, Any] | None,
) -> tuple[list[str], bool]:
    current_auc_nan = _diagnostic_metric_count(current_diagnostics, "auc")
    current_pearson_nan = _diagnostic_metric_count(current_diagnostics, "pearson_r")
    current_spearman_nan = _diagnostic_metric_count(current_diagnostics, "spearman_r")
    compared_auc_nan = _diagnostic_metric_count(compared_diagnostics, "auc")
    compared_pearson_nan = _diagnostic_metric_count(compared_diagnostics, "pearson_r")
    compared_spearman_nan = _diagnostic_metric_count(compared_diagnostics, "spearman_r")

    coverage_concern = (
        current_auc_nan > compared_auc_nan
        or current_pearson_nan > compared_pearson_nan
        or current_spearman_nan > compared_spearman_nan
    )
    auc_span = _fold_metric_span(
        current_diagnostics,
        best_key="best_auc_fold",
        worst_key="worst_auc_fold",
        metric_name="auc",
    )
    compared_auc_span = _fold_metric_span(
        compared_diagnostics,
        best_key="best_auc_fold",
        worst_key="worst_auc_fold",
        metric_name="auc",
    )
    rmse_span = _fold_metric_span(
        current_diagnostics,
        best_key="best_rmse_fold",
        worst_key="worst_rmse_fold",
        metric_name="rmse",
    )
    compared_rmse_span = _fold_metric_span(
        compared_diagnostics,
        best_key="best_rmse_fold",
        worst_key="worst_rmse_fold",
        metric_name="rmse",
    )
    auc_span_worse = (
        _is_defined_number(auc_span)
        and _is_defined_number(compared_auc_span)
        and float(auc_span) > float(compared_auc_span) + INTERPRETATION_AUC_SPAN_DELTA
    )
    rmse_span_worse = (
        _is_defined_number(rmse_span)
        and _is_defined_number(compared_rmse_span)
        and float(rmse_span) > float(compared_rmse_span) + INTERPRETATION_RMSE_SPAN_DELTA
    )
    robustness_concern = (
        metric_bundle["cv_rmse_std"].compared_label == "worse"
        or coverage_concern
        or auc_span_worse
        or rmse_span_worse
    )

    bullets: list[str] = []
    if coverage_concern:
        bullets.append(
            "- Robustness interpretation: this is a `robustness concern`; undefined AUC or correlation counts increased, so metric coverage became less informative."
        )
    if (
        metric_bundle["cv_rmse_std"].compared_label == "worse"
        or auc_span_worse
        or rmse_span_worse
    ):
        bullets.append(
            "- Robustness interpretation: this is a `robustness concern`; fold variability worsened relative to the compared run."
        )
    if not bullets:
        if compared_diagnostics is None:
            bullets.append(
                "- Robustness interpretation: this run establishes the initial fold-robustness reference for the session."
            )
        else:
            bullets.append(
                "- Robustness interpretation: no new robustness concern stood out; fold spread and undefined metric counts were broadly stable."
            )
    return bullets, robustness_concern


def _build_next_run_implication(
    *,
    decision: DecisionRecord,
    metric_bundle: dict[str, MetricDeltaView],
    robustness_concern: bool,
) -> str:
    if decision.decision_baseline_run_id is None:
        return "use this run as the session reference point and perturb only one architectural axis next."

    auc_label = metric_bundle["weighted_cv_auc"].compared_label
    rmse_label = metric_bundle["weighted_cv_rmse_mean"].compared_label
    pearson_label = metric_bundle["weighted_cv_pearson_r_mean"].compared_label
    spearman_label = metric_bundle["weighted_cv_spearman_r_mean"].compared_label
    mixed_signal = auc_label in {"worse", "flat"} and (
        rmse_label == "better" or pearson_label == "better" or spearman_label == "better"
    )

    if decision.decision_status == "keep":
        if rmse_label == "worse" or pearson_label == "worse" or spearman_label == "worse":
            return (
                "keep the incumbent, then probe nearby variants that try to recover the weakened secondary behavior without giving back AUC."
            )
        return "keep the incumbent and continue with one local architectural mutation at a time."

    if mixed_signal:
        return (
            "try a smaller or more localized mutation in the same direction rather than repeating the full change."
        )
    if robustness_concern:
        return "revert to the incumbent and prefer a smaller or simpler nearby mutation before revisiting this direction."
    return "revert to the incumbent and switch to a different architectural axis next."


def _build_interpretation_bullets(
    *,
    decision: DecisionRecord,
    metric_bundle: dict[str, MetricDeltaView],
    current_diagnostics: dict[str, Any] | None,
    compared_diagnostics: dict[str, Any] | None,
) -> list[str]:
    auc_movement = metric_bundle["weighted_cv_auc"]
    rmse_movement = metric_bundle["weighted_cv_rmse_mean"]
    pearson_movement = metric_bundle["weighted_cv_pearson_r_mean"]
    spearman_movement = metric_bundle["weighted_cv_spearman_r_mean"]

    if decision.decision_status == "keep":
        decision_line = (
            f"- Decision interpretation: kept because `{decision.decision_metric_name}` moved by "
            f"`{_format_metric_delta(decision.decision_delta)}` against the compared run, so the primary-metric rule accepted it."
        )
    elif decision.decision_status == "discard":
        decision_line = (
            f"- Decision interpretation: discarded because `{decision.decision_metric_name}` moved by "
            f"`{_format_metric_delta(decision.decision_delta)}` against the compared run, so the primary-metric rule did not keep it."
        )
    else:
        decision_line = "- Decision interpretation: this run crashed before a valid summary was produced."

    lines = [decision_line]
    near_miss = _is_near_miss_discard(decision, metric_bundle)
    if decision.decision_baseline_run_id is None:
        lines[0] = "- Decision interpretation: this base run established the first incumbent and the initial session reference point."
        lines.append(
            "- Metric tradeoff: no compared run exists yet, so this synopsis establishes the baseline for later tradeoff interpretation."
        )
    elif decision.decision_status == "keep":
        if rmse_movement.compared_label == "worse":
            lines.append(
                "- Metric tradeoff: this was a `brittle gain`; `weighted_cv_auc` improved while `weighted_cv_rmse_mean` worsened."
            )
            lines.append(
                "- Metric tradeoff: this looks like a `threshold-separation gain` rather than a broad regression improvement."
            )
        elif (
            pearson_movement.compared_label == "worse"
            or spearman_movement.compared_label == "worse"
        ):
            lines.append(
                "- Metric tradeoff: this was a `brittle gain`; the primary metric improved, but at least one correlation metric weakened."
            )
        elif sum(
            movement.compared_label == "better"
            for movement in (rmse_movement, pearson_movement, spearman_movement)
        ) >= 2:
            lines.append(
                "- Metric tradeoff: this was a `broader-based gain`; `weighted_cv_auc` improved alongside multiple secondary metrics."
            )
        else:
            lines.append(
                "- Metric tradeoff: this was a `clear improvement` on the primary metric with limited secondary-metric disagreement."
            )
    elif near_miss:
        lines.append(
            "- Metric tradeoff: this was a `near miss` and a `mixed signal`; `weighted_cv_auc` declined only slightly while multiple secondary metrics improved."
        )
        if (
            pearson_movement.compared_label == "better"
            or spearman_movement.compared_label == "better"
        ):
            lines.append(
                "- Metric tradeoff: this looks more like a `ranking gain` than a `threshold-separation gain`."
            )
        if rmse_movement.compared_label == "better":
            lines.append(
                "- Metric tradeoff: there was also a `calibration gain`; `weighted_cv_rmse_mean` improved despite the discard."
            )
    elif (
        auc_movement.compared_label == "worse"
        and pearson_movement.compared_label == "better"
        and spearman_movement.compared_label == "better"
    ):
        lines.append(
            "- Metric tradeoff: this was a `mixed signal`; `weighted_cv_pearson_r_mean` and `weighted_cv_spearman_r_mean` improved while `weighted_cv_auc` worsened."
        )
        lines.append(
            "- Metric tradeoff: the mutation may have improved global ordering structure without improving threshold-based separation."
        )
    elif auc_movement.compared_label in {"worse", "flat"} and rmse_movement.compared_label == "better":
        lines.append(
            "- Metric tradeoff: this looks like a `calibration gain`; `weighted_cv_rmse_mean` improved without a primary-metric gain."
        )
        if auc_movement.compared_label == "worse":
            lines.append(
                "- Metric tradeoff: the run fit the continuous target better but hurt `weighted_cv_auc`."
            )
    elif auc_movement.compared_label == "worse":
        lines.append(
            "- Metric tradeoff: this was a `clear regression`; the primary metric declined without offsetting secondary improvements."
        )
    else:
        lines.append(
            "- Metric tradeoff: metric movements were limited and did not produce a strong new interpretation signal."
        )

    if (
        auc_movement.base_label == "better"
        and auc_movement.compared_label in {"worse", "flat"}
    ):
        lines.append(
            "- Session context: despite losing to the compared run, this candidate still outperformed the session base on `weighted_cv_auc`."
        )

    robustness_lines, robustness_concern = _build_robustness_interpretation(
        metric_bundle=metric_bundle,
        current_diagnostics=current_diagnostics,
        compared_diagnostics=compared_diagnostics,
    )
    lines.extend(robustness_lines)
    lines.append(f"- Hypothesis assessment: `{decision.hypothesis_result or 'inconclusive'}`")
    lines.append(
        "- Next-run implication: "
        + _build_next_run_implication(
            decision=decision,
            metric_bundle=metric_bundle,
            robustness_concern=robustness_concern,
        )
    )
    return lines


def _suggest_next_mutations(
    architecture_payload: dict[str, Any] | None,
    *,
    decision_status: str,
) -> list[str]:
    if architecture_payload is None:
        return [
            "Return to the incumbent and perturb one architecture variable at a time.",
            "Prefer a small local change over a multi-axis mutation.",
        ]

    hidden_dims = architecture_payload.get("hidden_dims", [])
    dropout = architecture_payload.get("dropout")
    family = architecture_payload.get("family")
    return [
        f"Probe a nearby width schedule around {hidden_dims} without changing more than one width.",
        f"Test a small dropout adjustment around {dropout} while keeping family={family}.",
        "If the result was not kept, revert to the incumbent and change only one architectural axis next.",
    ] if decision_status != "keep" else [
        f"Explore one neighboring width variant around {hidden_dims}.",
        f"Test a nearby dropout value around {dropout}.",
        "Try one simplification step and keep it only if the primary metric still improves.",
    ]


def write_run_synopsis(
    *,
    run_dir: Path,
    run_context: RunContext,
    decision: DecisionRecord,
    summary: ExperimentSummary | None,
    summary_payload: dict[str, Any] | None,
    compared_summary_payload: dict[str, Any] | None,
    session_base_summary_payload: dict[str, Any] | None,
) -> None:
    architecture_payload = None if summary_payload is None else summary_payload.get("architecture")
    diagnostics = None if summary_payload is None else summary_payload.get("diagnostics", {})
    compared_diagnostics = (
        None if compared_summary_payload is None else compared_summary_payload.get("diagnostics", {})
    )
    compared_summary = (
        None if compared_summary_payload is None else _summary_from_payload(compared_summary_payload)
    )
    base_summary = (
        None if session_base_summary_payload is None else _summary_from_payload(session_base_summary_payload)
    )
    lines = [
        "---",
        f"session_id: {run_context.session_id}",
        f"run_id: {run_context.run_id}",
        f"session_run_index: {run_context.session_run_index}",
        f"run_role: {run_context.run_role}",
        f"parent_run_id: {run_context.parent_run_id or ''}",
        f"decision_status: {decision.decision_status}",
        f"decision_metric_name: {decision.decision_metric_name}",
        f"decision_metric_value: {'' if decision.decision_metric_value is None else decision.decision_metric_value}",
        f"decision_baseline_value: {'' if decision.decision_baseline_value is None else decision.decision_baseline_value}",
        f"decision_delta: {'' if decision.decision_delta is None else decision.decision_delta}",
        f"hypothesis: {run_context.hypothesis}",
        f"mutation_summary: {run_context.mutation_summary}",
        "---",
        "",
        "# Run Synopsis",
        "",
        "## Run Header",
        (
            f"- Status: `{decision.decision_status}`"
            f" on `{decision.decision_metric_name}` = `{_format_metric(decision.decision_metric_value)}`"
            f" (`{_format_metric_delta(decision.decision_delta)}` vs compared run)"
        ),
        f"- Run role: `{run_context.run_role}`",
        f"- Compared against: `{run_context.compared_against_run_id or 'none'}`",
        "",
        "## Architecture Synopsis",
        f"- Architecture: `{architecture_payload}`" if architecture_payload is not None else "- Architecture: unavailable",
        f"- Hypothesis: {run_context.hypothesis or 'No hypothesis recorded.'}",
        f"- Mutation summary: {run_context.mutation_summary or 'No mutation summary recorded.'}",
        "",
        "## Performance Overview",
    ]
    if summary is None:
        lines.extend(
            [
                "- This run crashed before a summary was written.",
                "",
                "## Next-Run Guidance",
                "- Restore the incumbent with `session_manager.py sync-incumbent` before the next mutation.",
            ]
        )
    else:
        metric_bundle = _build_metric_delta_bundle(
            summary,
            compared_summary,
            base_summary,
        )
        for metric_name in INTERPRETATION_METRIC_ORDER:
            lines.append(_format_metric_movement_line(metric_bundle[metric_name]))
        lines.extend(
            [
                "",
                "## Robustness Notes",
                f"- Best AUC fold: `{diagnostics.get('best_auc_fold')}`",
                f"- Worst AUC fold: `{diagnostics.get('worst_auc_fold')}`",
                f"- Best RMSE fold: `{diagnostics.get('best_rmse_fold')}`",
                f"- Worst RMSE fold: `{diagnostics.get('worst_rmse_fold')}`",
                f"- Undefined metric counts: `{diagnostics.get('nan_metric_counts')}`",
                "",
                "## Interpretation",
            ]
        )
        lines.extend(
            _build_interpretation_bullets(
                decision=decision,
                metric_bundle=metric_bundle,
                current_diagnostics=diagnostics,
                compared_diagnostics=compared_diagnostics,
            )
        )
        lines.extend(["", "## Next-Run Guidance"])
        for suggestion in _suggest_next_mutations(
            architecture_payload,
            decision_status=decision.decision_status,
        ):
            lines.append(f"- {suggestion}")

    (run_dir / "synopsis.md").write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


def _run_record_from_state(state: SessionState, run_id: str) -> tuple[RunContext, dict[str, Any] | None, DecisionRecord | None]:
    run_dir = Path(state.run_dirs[run_id])
    run_context = RunContext(**_read_json(run_dir / "run_context.json"))
    summary_payload = _load_run_summary(run_dir) if (run_dir / "summary.json").exists() else None
    decision_path = run_dir / "decision.json"
    decision = DecisionRecord(**_read_json(decision_path)) if decision_path.exists() else None
    return run_context, summary_payload, decision


def _architecture_key(payload: dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True)


def _recommended_next_mutations(best_architecture: dict[str, Any] | None) -> list[str]:
    return _suggest_next_mutations(best_architecture, decision_status="keep")


def write_session_summary(
    session_id: str,
    *,
    status: str,
    end_reason: str,
) -> SessionSummaryRecord:
    context = load_session_context(session_id)
    state = load_session_state(session_id)
    run_ids = list(state.ordered_run_ids)
    run_records = [
        _run_record_from_state(state, run_id)
        for run_id in run_ids
    ]
    successful_runs = [
        (run_context, summary_payload, decision)
        for run_context, summary_payload, decision in run_records
        if summary_payload is not None and decision is not None
    ]
    best_run_payload = None
    if state.incumbent_run_id is not None and state.incumbent_run_id in state.run_dirs:
        best_run_payload = _load_run_summary(Path(state.run_dirs[state.incumbent_run_id]))

    architectures = [summary_payload["architecture"] for _, summary_payload, _ in successful_runs]
    unique_architectures = {_architecture_key(payload) for payload in architectures}
    discarded_runs = [
        (run_context, summary_payload, decision)
        for run_context, summary_payload, decision in successful_runs
        if decision is not None and decision.decision_status == "discard"
    ]
    near_misses = sorted(
        [
            {
                "run_id": run_context.run_id,
                "primary_metric_value": summary_payload["summary"]["primary_metric_value"],
                "mutation_summary": run_context.mutation_summary,
                "description": run_context.description,
            }
            for run_context, summary_payload, _ in discarded_runs
        ],
        key=lambda item: item["primary_metric_value"],
        reverse=True,
    )[:3]
    best_secondary_snapshot = {}
    if best_run_payload is not None:
        best_summary = best_run_payload["summary"]
        best_secondary_snapshot = {
            "weighted_cv_rmse_mean": best_summary.get("weighted_cv_rmse_mean"),
            "weighted_cv_mae_mean": best_summary.get("weighted_cv_mae_mean"),
            "weighted_cv_r2_mean": best_summary.get("weighted_cv_r2_mean"),
            "weighted_cv_pearson_r_mean": best_summary.get("weighted_cv_pearson_r_mean"),
            "weighted_cv_spearman_r_mean": best_summary.get("weighted_cv_spearman_r_mean"),
            "pooled_cv_rmse": best_summary.get("pooled_cv_rmse"),
        }

    hypotheses_tested = [
        run_context.hypothesis
        for run_context, _, _ in run_records
        if run_context.hypothesis
    ]
    hypotheses_supported = [
        run_context.hypothesis
        for run_context, _, decision in run_records
        if decision is not None and decision.hypothesis_result == "supported" and run_context.hypothesis
    ]
    hypotheses_unsupported = [
        run_context.hypothesis
        for run_context, _, decision in run_records
        if decision is not None and decision.hypothesis_result == "unsupported" and run_context.hypothesis
    ]
    patterns_that_helped = list(
        dict.fromkeys(
            run_context.mutation_summary
            for run_context, _, decision in run_records
            if decision is not None and decision.decision_status == "keep" and run_context.mutation_summary
        )
    )[:5]
    patterns_that_hurt = list(
        dict.fromkeys(
            run_context.mutation_summary
            for run_context, _, decision in run_records
            if decision is not None and decision.decision_status in {"discard", "crash"} and run_context.mutation_summary
        )
    )[:5]
    instability_patterns = [
        f"{run_context.run_id}: undefined metrics {summary_payload.get('diagnostics', {}).get('nan_metric_counts')}"
        for run_context, summary_payload, _ in successful_runs
        if summary_payload.get("diagnostics", {}).get("nan_metric_counts", {}).get("auc", 0) > 0
    ][:5]
    unresolved_questions = [
        "Which nearby width or dropout change around the incumbent should be explored next?"
    ]

    if state.completed_at is None:
        completed_at = _utc_now_iso()
    else:
        completed_at = state.completed_at
    duration_seconds = (
        (_utc_now() - datetime.fromisoformat(context.started_at)).total_seconds()
        if state.duration_seconds is None
        else state.duration_seconds
    )
    base_metric = state.base_primary_metric_value
    best_metric = state.incumbent_primary_metric_value
    summary_record = SessionSummaryRecord(
        session_id=session_id,
        status=status,
        started_at=context.started_at,
        completed_at=completed_at,
        duration_seconds=duration_seconds,
        objective=context.objective,
        initiated_by=context.initiated_by,
        end_reason=end_reason,
        base_run_id=state.base_run_id,
        best_run_id=state.incumbent_run_id,
        final_incumbent_run_id=state.incumbent_run_id,
        total_runs=len(run_ids),
        keep_count=state.keep_count,
        discard_count=state.discard_count,
        crash_count=state.crash_count,
        rerun_count=state.rerun_count,
        best_primary_metric_value=best_metric,
        delta_from_base=None if best_metric is None or base_metric is None else best_metric - base_metric,
        best_secondary_metric_snapshot=best_secondary_snapshot,
        most_promising_near_misses=near_misses,
        search_space_coverage={
            "families_tried": sorted({payload.get("family") for payload in architectures}),
            "activations_tried": sorted({payload.get("activation") for payload in architectures}),
            "normalizations_tried": sorted({payload.get("normalization") for payload in architectures}),
            "dropout_values_tried": sorted({payload.get("dropout") for payload in architectures}),
            "depth_values_tried": sorted({len(payload.get("hidden_dims", [])) for payload in architectures}),
            "max_width_values_tried": sorted(
                {max(payload.get("hidden_dims", [0])) for payload in architectures}
            ),
            "param_count_min": (
                None if not successful_runs else min(item[1]["summary"]["num_params"] for item in successful_runs)
            ),
            "param_count_max": (
                None if not successful_runs else max(item[1]["summary"]["num_params"] for item in successful_runs)
            ),
            "unique_architecture_count": len(unique_architectures),
            "repeated_architecture_count": len(architectures) - len(unique_architectures),
        },
        hypotheses_tested=hypotheses_tested,
        hypotheses_supported=hypotheses_supported,
        hypotheses_unsupported=hypotheses_unsupported,
        patterns_that_helped=patterns_that_helped,
        patterns_that_hurt=patterns_that_hurt,
        instability_patterns=instability_patterns,
        unresolved_questions=unresolved_questions,
        recommended_next_starting_run_id=state.incumbent_run_id,
        recommended_next_mutations=_recommended_next_mutations(
            None if best_run_payload is None else best_run_payload.get("architecture")
        ),
    )
    _write_json(_session_summary_json_path(session_id), asdict(summary_record))

    lines = [
        "---",
        f"session_id: {summary_record.session_id}",
        f"status: {summary_record.status}",
        f"base_run_id: {summary_record.base_run_id or ''}",
        f"best_run_id: {summary_record.best_run_id or ''}",
        f"final_incumbent_run_id: {summary_record.final_incumbent_run_id or ''}",
        f"total_runs: {summary_record.total_runs}",
        f"keep_count: {summary_record.keep_count}",
        f"discard_count: {summary_record.discard_count}",
        f"crash_count: {summary_record.crash_count}",
        f"best_primary_metric_value: {'' if summary_record.best_primary_metric_value is None else summary_record.best_primary_metric_value}",
        f"delta_from_base: {'' if summary_record.delta_from_base is None else summary_record.delta_from_base}",
        f"end_reason: {summary_record.end_reason or ''}",
        "---",
        "",
        "# Session Summary",
        "",
        "## Session Header",
        f"- Status: `{summary_record.status}`",
        f"- End reason: {summary_record.end_reason or 'n/a'}",
        f"- Total runs: `{summary_record.total_runs}`",
        "",
        "## Starting Point",
        f"- Base run: `{summary_record.base_run_id or 'n/a'}`",
        f"- Base primary metric: `{_format_metric(state.base_primary_metric_value)}`",
        "",
        "## Champion Progression",
    ]
    if not state.incumbent_progression:
        lines.append("- No incumbent progression recorded.")
    else:
        for entry in state.incumbent_progression:
            lines.append(
                f"- After `{entry['after_run_id']}`, incumbent became `{entry['new_incumbent_run_id']}` "
                f"at `{_format_metric(entry['primary_metric_value'])}` "
                f"({ _format_metric_delta(entry['delta_vs_previous_incumbent']) } vs previous incumbent)"
            )
    lines.extend(
        [
            "",
            "## Search Space Explored",
            f"- Coverage: `{summary_record.search_space_coverage}`",
            "",
            "## Session-Wide Findings",
            f"- Patterns that helped: `{summary_record.patterns_that_helped}`",
            f"- Patterns that hurt: `{summary_record.patterns_that_hurt}`",
            f"- Instability patterns: `{summary_record.instability_patterns}`",
            "",
            "## Final Recommendation",
            f"- Continue from run `{summary_record.recommended_next_starting_run_id or 'n/a'}`",
        ]
    )
    for mutation in summary_record.recommended_next_mutations:
        lines.append(f"- {mutation}")
    _session_summary_md_path(session_id).write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
    return summary_record


def sync_train_to_incumbent(session_id: str) -> Path:
    state = load_session_state(session_id)
    if state.incumbent_run_id is None:
        raise ValueError("Session has no incumbent run to sync.")
    incumbent_run_dir = Path(state.run_dirs[state.incumbent_run_id])
    snapshot_path = incumbent_run_dir / "train_snapshot.py"
    if not snapshot_path.exists():
        raise FileNotFoundError(f"Incumbent snapshot not found at {snapshot_path}.")
    EDITABLE_TRAIN_FILE.write_text(snapshot_path.read_text(encoding="utf-8"), encoding="utf-8")
    return snapshot_path


def finalize_session(
    session_id: str,
    *,
    status: str,
    end_reason: str,
) -> SessionSummaryRecord:
    state = load_session_state(session_id)
    state.status = status
    state.completed_at = _utc_now_iso()
    started_at = datetime.fromisoformat(state.started_at)
    state.duration_seconds = (_utc_now() - started_at).total_seconds()
    state.end_reason = end_reason
    state.final_incumbent_run_id = state.incumbent_run_id
    save_session_state(state)
    return write_session_summary(session_id, status=status, end_reason=end_reason)


def _build_run_context(
    context: SessionContext,
    state: SessionState,
    intent: RunIntent,
    *,
    run_id: str,
    session_run_index: int,
    run_dir: Path,
) -> RunContext:
    git_commit, git_branch, git_is_dirty = _collect_git_metadata()
    architecture_fingerprint, architecture_spec = _load_architecture_metadata()
    resolved_parent_run_id = intent.parent_run_id or state.incumbent_run_id
    resolved_compared_against_run_id = intent.compared_against_run_id or state.incumbent_run_id
    return RunContext(
        session_id=context.session_id,
        run_id=run_id,
        session_run_index=session_run_index,
        run_role=intent.run_role,
        is_base_run=intent.run_role == "base",
        parent_run_id=resolved_parent_run_id,
        compared_against_run_id=resolved_compared_against_run_id,
        best_known_run_id_at_start=state.incumbent_run_id,
        hypothesis=intent.hypothesis,
        mutation_summary=intent.mutation_summary,
        description=intent.description,
        started_at=_utc_now_iso(),
        completed_at=None,
        duration_seconds=None,
        run_dir=str(run_dir),
        git_commit=git_commit,
        git_branch=git_branch,
        git_is_dirty=git_is_dirty,
        train_py_sha256=_sha256_path(EDITABLE_TRAIN_FILE),
        architecture_fingerprint=architecture_fingerprint,
        program_md_sha256=_sha256_path(PROGRAM_FILE),
        architecture_spec=architecture_spec,
    )


def _summary_from_payload(payload: dict[str, Any]) -> ExperimentSummary:
    return ExperimentSummary(**payload["summary"])


def _success_decision_record(
    *,
    run_context: RunContext,
    summary: ExperimentSummary,
    state: SessionState,
) -> DecisionRecord:
    incumbent_before_run_id = state.incumbent_run_id
    baseline_value = state.incumbent_primary_metric_value
    if incumbent_before_run_id is None:
        status = "keep"
        decision_reason = "Base run establishes the first incumbent for the session."
        decision_delta = None
        hypothesis_result = "inconclusive"
        incumbent_after_run_id = run_context.run_id
    else:
        if baseline_value is None:
            raise RuntimeError("Incumbent metric is undefined for a non-base run.")
        decision_delta = summary.primary_metric_value - baseline_value
        if decision_delta > summary.improvement_epsilon:
            status = "keep"
            decision_reason = (
                f"Primary metric improved by {decision_delta:.6f}, above epsilon "
                f"{summary.improvement_epsilon:.6f}."
            )
            incumbent_after_run_id = run_context.run_id
            hypothesis_result = "supported"
        else:
            status = "discard"
            decision_reason = (
                f"Primary metric delta {decision_delta:.6f} did not exceed epsilon "
                f"{summary.improvement_epsilon:.6f}."
            )
            incumbent_after_run_id = incumbent_before_run_id
            hypothesis_result = "unsupported"

    return DecisionRecord(
        session_id=run_context.session_id,
        run_id=run_context.run_id,
        decision_status=status,
        decision_metric_name=summary.primary_metric_name,
        decision_metric_value=summary.primary_metric_value,
        decision_baseline_run_id=incumbent_before_run_id,
        decision_baseline_value=baseline_value,
        decision_delta=decision_delta,
        decision_epsilon=summary.improvement_epsilon,
        decision_reason=decision_reason,
        incumbent_before_run_id=incumbent_before_run_id,
        incumbent_after_run_id=incumbent_after_run_id,
        compared_against_run_id=run_context.compared_against_run_id,
        hypothesis_result=hypothesis_result,
    )


def _crash_decision_record(run_context: RunContext, state: SessionState) -> DecisionRecord:
    return DecisionRecord(
        session_id=run_context.session_id,
        run_id=run_context.run_id,
        decision_status="crash",
        decision_metric_name=METRIC_CONFIG.primary_metric_name,
        decision_metric_value=None,
        decision_baseline_run_id=state.incumbent_run_id,
        decision_baseline_value=state.incumbent_primary_metric_value,
        decision_delta=None,
        decision_epsilon=METRIC_CONFIG.improvement_epsilon,
        decision_reason="Run crashed before a valid experiment summary was produced.",
        incumbent_before_run_id=state.incumbent_run_id,
        incumbent_after_run_id=state.incumbent_run_id,
        compared_against_run_id=run_context.compared_against_run_id,
        hypothesis_result="inconclusive",
    )


def _record_successful_run(
    *,
    context: SessionContext,
    state: SessionState,
    run_context: RunContext,
    decision: DecisionRecord,
    summary: ExperimentSummary,
    summary_payload: dict[str, Any],
    compared_summary_payload: dict[str, Any] | None,
    session_base_summary_payload: dict[str, Any] | None,
) -> None:
    state.ordered_run_ids.append(run_context.run_id)
    state.latest_run_id = run_context.run_id
    state.run_dirs[run_context.run_id] = run_context.run_dir
    state.run_roles[run_context.run_id] = run_context.run_role
    state.run_statuses[run_context.run_id] = decision.decision_status
    state.run_primary_metric_values[run_context.run_id] = summary.primary_metric_value
    if run_context.run_role == "rerun":
        state.rerun_count += 1

    if decision.decision_status == "keep":
        previous_incumbent_value = state.incumbent_primary_metric_value
        state.keep_count += 1
        state.incumbent_run_id = run_context.run_id
        state.incumbent_primary_metric_value = summary.primary_metric_value
        state.final_incumbent_run_id = run_context.run_id
        state.incumbent_progression.append(
            {
                "after_run_id": run_context.run_id,
                "new_incumbent_run_id": run_context.run_id,
                "primary_metric_value": summary.primary_metric_value,
                "delta_vs_previous_incumbent": (
                    None
                    if previous_incumbent_value is None
                    else summary.primary_metric_value - previous_incumbent_value
                ),
            }
        )
        if run_context.is_base_run:
            state.base_run_id = run_context.run_id
            state.base_architecture_fingerprint = run_context.architecture_fingerprint
            state.base_primary_metric_value = summary.primary_metric_value
    else:
        state.discard_count += 1

    save_session_state(state)
    write_decision_record(Path(run_context.run_dir), decision)
    append_session_results_row(context.session_id, run_context, decision, summary)
    write_run_synopsis(
        run_dir=Path(run_context.run_dir),
        run_context=run_context,
        decision=decision,
        summary=summary,
        summary_payload=summary_payload,
        compared_summary_payload=compared_summary_payload,
        session_base_summary_payload=session_base_summary_payload,
    )


def _record_crashed_run(
    *,
    context: SessionContext,
    state: SessionState,
    run_context: RunContext,
    decision: DecisionRecord,
    failure_path: Path,
) -> None:
    state.ordered_run_ids.append(run_context.run_id)
    state.latest_run_id = run_context.run_id
    state.crash_count += 1
    state.run_dirs[run_context.run_id] = run_context.run_dir
    state.run_roles[run_context.run_id] = run_context.run_role
    state.run_statuses[run_context.run_id] = "crash"
    state.run_primary_metric_values[run_context.run_id] = None
    if run_context.run_role == "rerun":
        state.rerun_count += 1
    save_session_state(state)
    write_decision_record(Path(run_context.run_dir), decision)
    append_session_results_row(context.session_id, run_context, decision, summary=None)
    write_run_synopsis(
        run_dir=Path(run_context.run_dir),
        run_context=run_context,
        decision=decision,
        summary=None,
        summary_payload=None,
        compared_summary_payload=None,
        session_base_summary_payload=None,
    )
    RUN_LOG.write_text(failure_path.read_text(encoding="utf-8"), encoding="utf-8")


def run_session_experiment(intent: RunIntent) -> DecisionRecord:
    context = load_session_context(intent.session_id)
    state = load_session_state(intent.session_id)
    run_id, session_run_index, run_dir = allocate_run_dir(
        context,
        state,
        run_role=intent.run_role,
    )
    if intent.run_role != "base" and state.incumbent_run_id is None:
        raise ValueError("Run the base experiment before candidate runs.")

    run_context = _build_run_context(
        context,
        state,
        intent,
        run_id=run_id,
        session_run_index=session_run_index,
        run_dir=run_dir,
    )
    write_run_context(run_context)
    write_train_snapshot(run_dir)

    started_at = datetime.fromisoformat(run_context.started_at)
    try:
        summary = run_experiment(run_dir=run_dir)
    except Exception as exc:
        completed_at = _utc_now_iso()
        run_context.completed_at = completed_at
        run_context.duration_seconds = (_utc_now() - started_at).total_seconds()
        run_context.hypothesis_result = "inconclusive"
        failure_payload = {
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "timestamp": completed_at,
            "traceback": traceback.format_exc(),
        }
        failure_path = run_dir / "failure.json"
        _write_json(failure_path, failure_payload)
        run_context.failure_path = str(failure_path)
        write_run_context(run_context)
        decision = _crash_decision_record(run_context, state)
        _record_crashed_run(
            context=context,
            state=state,
            run_context=run_context,
            decision=decision,
            failure_path=failure_path,
        )
        return decision

    completed_at = _utc_now_iso()
    run_context.completed_at = completed_at
    run_context.duration_seconds = (_utc_now() - started_at).total_seconds()
    run_context.summary_path = str(run_dir / "summary.json")

    summary_payload = _load_run_summary(run_dir)
    compared_summary_payload = _compared_summary_payload(state, run_context)
    session_base_summary_payload = _session_base_summary_payload(state)
    decision = _success_decision_record(
        run_context=run_context,
        summary=summary,
        state=state,
    )
    decision = _enrich_decision_record(
        decision,
        current_summary=summary,
        compared_summary=(
            None if compared_summary_payload is None else _summary_from_payload(compared_summary_payload)
        ),
    )
    run_context.hypothesis_result = decision.hypothesis_result
    write_run_context(run_context)
    _record_successful_run(
        context=context,
        state=state,
        run_context=run_context,
        decision=decision,
        summary=summary,
        summary_payload=summary_payload,
        compared_summary_payload=compared_summary_payload,
        session_base_summary_payload=session_base_summary_payload,
    )
    RUN_LOG.write_text(
        json.dumps(
            {
                "session_id": context.session_id,
                "run_id": run_context.run_id,
                "decision_status": decision.decision_status,
                "primary_metric_value": summary.primary_metric_value,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return decision


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Manage autonomous search sessions and run artifacts.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    start_parser = subparsers.add_parser("start", help="Create a new search session.")
    start_parser.add_argument("--session-id", default=None)
    start_parser.add_argument("--objective", default=f"Maximize {METRIC_CONFIG.primary_metric_name}")
    start_parser.add_argument("--initiated-by", default="agent")

    run_parser = subparsers.add_parser("run", help="Execute one run inside an existing session.")
    run_parser.add_argument("--session-id", required=True)
    run_parser.add_argument(
        "--run-role",
        required=True,
        choices=("base", "candidate", "rerun", "recovery"),
    )
    run_parser.add_argument("--parent-run-id", default=None)
    run_parser.add_argument("--compared-against-run-id", default=None)
    run_parser.add_argument("--hypothesis", default="")
    run_parser.add_argument("--mutation-summary", default="")
    run_parser.add_argument("--description", default="")

    sync_parser = subparsers.add_parser("sync-incumbent", help="Restore train.py from the incumbent run.")
    sync_parser.add_argument("--session-id", required=True)

    status_parser = subparsers.add_parser("status", help="Show current session status.")
    status_parser.add_argument("--session-id", required=True)

    finalize_parser = subparsers.add_parser("finalize", help="Finalize a session and write its summary.")
    finalize_parser.add_argument("--session-id", required=True)
    finalize_parser.add_argument("--status", default="completed", choices=("completed", "aborted"))
    finalize_parser.add_argument("--end-reason", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "start":
        context = create_session(
            session_id=args.session_id,
            objective=args.objective,
            initiated_by=args.initiated_by,
        )
        print(context.session_id)
        return 0

    if args.command == "run":
        decision = run_session_experiment(
            RunIntent(
                session_id=args.session_id,
                run_role=args.run_role,
                parent_run_id=args.parent_run_id,
                compared_against_run_id=args.compared_against_run_id,
                hypothesis=args.hypothesis,
                mutation_summary=args.mutation_summary,
                description=args.description,
            )
        )
        print(json.dumps(asdict(decision), indent=2, sort_keys=True))
        return 0

    if args.command == "sync-incumbent":
        snapshot_path = sync_train_to_incumbent(args.session_id)
        print(snapshot_path)
        return 0

    if args.command == "status":
        state = load_session_state(args.session_id)
        payload = {
            "session_id": state.session_id,
            "status": state.status,
            "latest_run_id": state.latest_run_id,
            "incumbent_run_id": state.incumbent_run_id,
            "incumbent_primary_metric_value": state.incumbent_primary_metric_value,
            "keep_count": state.keep_count,
            "discard_count": state.discard_count,
            "crash_count": state.crash_count,
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    if args.command == "finalize":
        summary = finalize_session(
            args.session_id,
            status=args.status,
            end_reason=args.end_reason,
        )
        print(json.dumps(asdict(summary), indent=2, sort_keys=True))
        return 0

    raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
