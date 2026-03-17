from __future__ import annotations

import json
import traceback
from datetime import datetime
from pathlib import Path

from autoresirch.prepare import METRIC_CONFIG, run_experiment
from autoresirch.session_manager.analysis import (
    _build_analysis_input_record,
    _build_failure_analysis_input_record,
    _compared_summary_payload,
    _enrich_decision_record,
    _session_base_summary_payload,
    _summary_from_payload,
    _write_analysis_input,
    write_run_synopsis,
    write_session_summary,
)
from autoresirch.session_manager.constants import EDITABLE_TRAIN_FILE, PROGRAM_FILE, RUN_LOG
from autoresirch.session_manager.schemas import (
    DecisionRecord,
    RunContext,
    RunIntent,
    SessionContext,
    SessionState,
    SessionSummaryRecord,
)
from autoresirch.session_manager.storage import (
    _collect_git_metadata,
    _load_architecture_metadata,
    _load_run_summary,
    _sha256_path,
    _utc_now,
    _utc_now_iso,
    _write_json,
    allocate_run_dir,
    append_session_results_row,
    create_session,
    load_session_context,
    load_session_state,
    save_session_state,
    write_decision_record,
    write_run_context,
    write_train_snapshot,
)


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


def _success_decision_record(
    *,
    run_context: RunContext,
    summary,
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
    summary,
    summary_payload: dict,
    compared_summary_payload: dict | None,
    session_base_summary_payload: dict | None,
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
    run_dir = Path(run_context.run_dir)
    write_decision_record(run_dir, decision)
    _write_analysis_input(
        run_dir,
        _build_analysis_input_record(
            state=state,
            run_context=run_context,
            decision=decision,
            summary=summary,
            summary_payload=summary_payload,
            compared_summary_payload=compared_summary_payload,
            session_base_summary_payload=session_base_summary_payload,
        ),
    )
    append_session_results_row(context.session_id, run_context, decision, summary)
    write_run_synopsis(
        run_dir=run_dir,
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
    failure_payload: dict,
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
    run_dir = Path(run_context.run_dir)
    write_decision_record(run_dir, decision)
    _write_analysis_input(
        run_dir,
        _build_failure_analysis_input_record(
            state=state,
            run_context=run_context,
            decision=decision,
            failure_payload=failure_payload,
        ),
    )
    append_session_results_row(context.session_id, run_context, decision, summary=None)
    write_run_synopsis(
        run_dir=run_dir,
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
            failure_payload=failure_payload,
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
