from __future__ import annotations

import json
import math
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

from autoresirch.prepare import ExperimentSummary
from autoresirch.session_manager.constants import (
    AGENT_ANALYSIS_FAILURE_MIN_WORDS,
    AGENT_ANALYSIS_MAX_FACTORS,
    AGENT_ANALYSIS_MAX_FACTOR_WORDS,
    AGENT_ANALYSIS_MAX_NEXT_STEP_WORDS,
    AGENT_ANALYSIS_MAX_WORDS,
    AGENT_ANALYSIS_MIN_WORDS,
    ANALYSIS_SCHEMA_VERSION,
    INTERPRETATION_AUC_SPAN_DELTA,
    INTERPRETATION_NEAR_MISS_AUC_DELTA,
    INTERPRETATION_RMSE_SPAN_DELTA,
)
from autoresirch.session_manager.shared.metrics import (
    analysis_input_metric_order_for_mode,
    interpretation_metric_order_for_mode,
    interpretation_metric_specs_for_mode,
    primary_metric_name_for_mode,
)
from autoresirch.session_manager.schemas import (
    AgentAnalysisRecord,
    AnalysisInputRecord,
    DecisionRecord,
    MetricDeltaView,
    RunContext,
    SessionState,
    SessionSummaryRecord,
)
from autoresirch.session_manager.storage import (
    _agent_analysis_path,
    _analysis_input_path,
    _load_run_summary,
    _read_json,
    _session_summary_json_path,
    _session_summary_md_path,
    _sha256_path,
    _utc_now,
    _utc_now_iso,
    _write_json,
    load_session_context,
    load_session_state,
)


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


def _summary_from_payload(payload: dict[str, Any]) -> ExperimentSummary:
    return ExperimentSummary(**payload["summary"])


def _summary_metric_value(
    summary: ExperimentSummary | None,
    metric_name: str,
) -> float | int | None:
    if summary is None:
        return None
    metric_spec = interpretation_metric_specs_for_mode(summary.experiment_mode)[metric_name]
    return getattr(summary, metric_spec.summary_attr)


def _classify_metric_direction(
    metric_name: str,
    delta: float | None,
    *,
    experiment_mode: str,
) -> str:
    if not _is_defined_number(delta):
        return "undefined"
    metric_spec = interpretation_metric_specs_for_mode(experiment_mode)[metric_name]
    if float(delta) > metric_spec.flat_threshold:
        return "better" if metric_spec.direction == "higher" else "worse"
    if float(delta) < -metric_spec.flat_threshold:
        return "worse" if metric_spec.direction == "higher" else "better"
    return "flat"


def _build_metric_delta_bundle(
    current_summary: ExperimentSummary,
    compared_summary: ExperimentSummary | None,
    base_summary: ExperimentSummary | None,
    *,
    metric_names: Sequence[str] | None = None,
) -> dict[str, MetricDeltaView]:
    resolved_metric_names = metric_names or interpretation_metric_order_for_mode(
        current_summary.experiment_mode
    )
    metric_bundle: dict[str, MetricDeltaView] = {}
    for metric_name in resolved_metric_names:
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
            compared_label=_classify_metric_direction(
                metric_name,
                delta_vs_compared,
                experiment_mode=current_summary.experiment_mode,
            ),
            base_label=_classify_metric_direction(
                metric_name,
                delta_vs_base,
                experiment_mode=current_summary.experiment_mode,
            ),
        )
    return metric_bundle


def _format_metric_movement_line(
    metric_movement: MetricDeltaView,
    *,
    experiment_mode: str,
) -> str:
    metric_spec = interpretation_metric_specs_for_mode(experiment_mode)[metric_movement.metric_name]
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


def _metric_view_payload(
    metric_bundle: dict[str, MetricDeltaView],
    *,
    experiment_mode: str,
    compared_summary: ExperimentSummary | None,
    base_summary: ExperimentSummary | None,
) -> dict[str, dict[str, Any]]:
    payload: dict[str, dict[str, Any]] = {}
    for metric_name, metric_view in metric_bundle.items():
        metric_spec = interpretation_metric_specs_for_mode(experiment_mode)[metric_name]
        payload[metric_name] = {
            "display_name": metric_spec.display_name,
            "direction": metric_spec.direction,
            "current_value": metric_view.current_value,
            "compared_value": _summary_metric_value(compared_summary, metric_name),
            "base_value": _summary_metric_value(base_summary, metric_name),
            "delta_vs_compared": metric_view.delta_vs_compared,
            "delta_vs_base": metric_view.delta_vs_base,
            "compared_label": metric_view.compared_label,
            "base_label": metric_view.base_label,
        }
    return payload


def _analysis_diagnostics_payload(
    diagnostics: dict[str, Any] | None,
) -> dict[str, Any]:
    if diagnostics is None:
        return {}
    return {
        "fold_count": diagnostics.get("fold_count"),
        "nan_metric_counts": diagnostics.get("nan_metric_counts"),
        "undefined_metric_counts": diagnostics.get("undefined_metric_counts"),
        "best_auc_fold": diagnostics.get("best_auc_fold"),
        "worst_auc_fold": diagnostics.get("worst_auc_fold"),
        "best_rmse_fold": diagnostics.get("best_rmse_fold"),
        "worst_rmse_fold": diagnostics.get("worst_rmse_fold"),
        "class_support": diagnostics.get("class_support"),
    }


def _analysis_constraints_payload(*, analysis_mode: str) -> dict[str, Any]:
    return {
        "analysis_mode": analysis_mode,
        "do_not_override_decision_rule": True,
        "require_concrete_metric_references": analysis_mode == "metric_comparison",
        "freeform_analysis_min_words": (
            AGENT_ANALYSIS_MIN_WORDS
            if analysis_mode == "metric_comparison"
            else AGENT_ANALYSIS_FAILURE_MIN_WORDS
        ),
        "freeform_analysis_max_words": AGENT_ANALYSIS_MAX_WORDS,
        "max_likely_helped_items": AGENT_ANALYSIS_MAX_FACTORS,
        "max_likely_hurt_items": AGENT_ANALYSIS_MAX_FACTORS,
        "max_factor_words": AGENT_ANALYSIS_MAX_FACTOR_WORDS,
        "next_step_reasoning_max_words": AGENT_ANALYSIS_MAX_NEXT_STEP_WORDS,
        "max_follow_up_ideas": 2,
    }


def _word_count(text: str) -> int:
    return len([word for word in text.strip().split() if word])


def _normalize_analysis_list(items: Sequence[str]) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    for item in items:
        cleaned = " ".join(item.strip().split())
        if not cleaned or cleaned in seen:
            continue
        normalized.append(cleaned)
        seen.add(cleaned)
    return normalized


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
    secondary_metrics = {
        "weighted_cv_rmse_mean": summary.weighted_cv_rmse_mean,
        "weighted_cv_mae_mean": summary.weighted_cv_mae_mean,
        "weighted_cv_r2_mean": summary.weighted_cv_r2_mean,
        "weighted_cv_pearson_r_mean": summary.weighted_cv_pearson_r_mean,
        "weighted_cv_spearman_r_mean": summary.weighted_cv_spearman_r_mean,
        "pooled_cv_rmse": summary.pooled_cv_rmse,
    }
    if summary.experiment_mode == "comparative":
        secondary_metrics.update(
            {
                "weighted_cv_auc_pos_vs_neg": summary.weighted_cv_auc_pos_vs_neg,
                "weighted_cv_auc_class_neg1": summary.weighted_cv_auc_class_neg1,
                "weighted_cv_auc_class_0": summary.weighted_cv_auc_class_0,
                "weighted_cv_auc_class_pos1": summary.weighted_cv_auc_class_pos1,
            }
        )
    return secondary_metrics


def _is_near_miss_discard(
    decision: DecisionRecord,
    metric_bundle: dict[str, MetricDeltaView],
    *,
    experiment_mode: str,
) -> bool:
    if decision.decision_status != "discard":
        return False
    auc_delta = metric_bundle[primary_metric_name_for_mode(experiment_mode)].delta_vs_compared
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
    auc_label = metric_bundle[primary_metric_name_for_mode(current_summary.experiment_mode)].compared_label
    rmse_label = metric_bundle["weighted_cv_rmse_mean"].compared_label
    pearson_label = metric_bundle["weighted_cv_pearson_r_mean"].compared_label
    spearman_label = metric_bundle["weighted_cv_spearman_r_mean"].compared_label

    if _is_near_miss_discard(
        decision,
        metric_bundle,
        experiment_mode=current_summary.experiment_mode,
    ):
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
    experiment_mode: str,
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
        metric_name="overall_auc" if experiment_mode == "comparative" else "auc",
    )
    compared_auc_span = _fold_metric_span(
        compared_diagnostics,
        best_key="best_auc_fold",
        worst_key="worst_auc_fold",
        metric_name="overall_auc" if experiment_mode == "comparative" else "auc",
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
    experiment_mode: str,
) -> str:
    if decision.decision_baseline_run_id is None:
        return "use this run as the session reference point and perturb only one architectural axis next."

    auc_label = metric_bundle[primary_metric_name_for_mode(experiment_mode)].compared_label
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
    experiment_mode: str,
) -> list[str]:
    primary_metric_name = primary_metric_name_for_mode(experiment_mode)
    auc_movement = metric_bundle[primary_metric_name]
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
    near_miss = _is_near_miss_discard(
        decision,
        metric_bundle,
        experiment_mode=experiment_mode,
    )
    if decision.decision_baseline_run_id is None:
        lines[0] = "- Decision interpretation: this base run established the first incumbent and the initial session reference point."
        lines.append(
            "- Metric tradeoff: no compared run exists yet, so this synopsis establishes the baseline for later tradeoff interpretation."
        )
    elif decision.decision_status == "keep":
        if rmse_movement.compared_label == "worse":
            lines.append(
                f"- Metric tradeoff: this was a `brittle gain`; `{primary_metric_name}` improved while `weighted_cv_rmse_mean` worsened."
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
                f"- Metric tradeoff: this was a `broader-based gain`; `{primary_metric_name}` improved alongside multiple secondary metrics."
            )
        else:
            lines.append(
                "- Metric tradeoff: this was a `clear improvement` on the primary metric with limited secondary-metric disagreement."
            )
    elif near_miss:
        lines.append(
            f"- Metric tradeoff: this was a `near miss` and a `mixed signal`; `{primary_metric_name}` declined only slightly while multiple secondary metrics improved."
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
            f"- Metric tradeoff: this was a `mixed signal`; `weighted_cv_pearson_r_mean` and `weighted_cv_spearman_r_mean` improved while `{primary_metric_name}` worsened."
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
                f"- Metric tradeoff: the run fit the continuous target better but hurt `{primary_metric_name}`."
            )
    elif auc_movement.compared_label == "worse":
        lines.append(
            "- Metric tradeoff: this was a `clear regression`; the primary metric declined without offsetting secondary improvements."
        )
    else:
        lines.append(
            "- Metric tradeoff: metric movements were limited and did not produce a strong new interpretation signal."
        )

    if auc_movement.base_label == "better" and auc_movement.compared_label in {"worse", "flat"}:
        lines.append(
            f"- Session context: despite losing to the compared run, this candidate still outperformed the session base on `{primary_metric_name}`."
        )
    undefined_metric_counts = current_diagnostics.get("undefined_metric_counts", {}) if current_diagnostics else {}
    if experiment_mode == "comparative" and any(int(value) > 0 for value in undefined_metric_counts.values()):
        lines.append(
            "- Comparative class support: some AUC metrics were undefined because at least one held-out gene fold did not contain the required class support."
        )

    robustness_lines, robustness_concern = _build_robustness_interpretation(
        metric_bundle=metric_bundle,
        current_diagnostics=current_diagnostics,
        compared_diagnostics=compared_diagnostics,
        experiment_mode=experiment_mode,
    )
    lines.extend(robustness_lines)
    lines.append(f"- Hypothesis assessment: `{decision.hypothesis_result or 'inconclusive'}`")
    lines.append(
        "- Next-run implication: "
        + _build_next_run_implication(
            decision=decision,
            metric_bundle=metric_bundle,
            robustness_concern=robustness_concern,
            experiment_mode=experiment_mode,
        )
    )
    return lines


def _build_analysis_input_record(
    *,
    state: SessionState,
    run_context: RunContext,
    decision: DecisionRecord,
    summary: ExperimentSummary,
    summary_payload: dict[str, Any],
    compared_summary_payload: dict[str, Any] | None,
    session_base_summary_payload: dict[str, Any] | None,
) -> AnalysisInputRecord:
    compared_summary = (
        None if compared_summary_payload is None else _summary_from_payload(compared_summary_payload)
    )
    base_summary = (
        None
        if session_base_summary_payload is None
        else _summary_from_payload(session_base_summary_payload)
    )
    diagnostics = summary_payload.get("diagnostics", {})
    compared_diagnostics = (
        None if compared_summary_payload is None else compared_summary_payload.get("diagnostics", {})
    )
    interpretation_metric_bundle = _build_metric_delta_bundle(
        summary,
        compared_summary,
        base_summary,
    )
    analysis_metric_bundle = _build_metric_delta_bundle(
        summary,
        compared_summary,
        base_summary,
        metric_names=analysis_input_metric_order_for_mode(summary.experiment_mode),
    )
    return AnalysisInputRecord(
        schema_version=ANALYSIS_SCHEMA_VERSION,
        analysis_mode="metric_comparison",
        session_id=run_context.session_id,
        run_id=run_context.run_id,
        session_run_index=run_context.session_run_index,
        run_role=run_context.run_role,
        decision_status=decision.decision_status,
        compared_against_run_id=run_context.compared_against_run_id,
        base_run_id=state.base_run_id,
        best_known_run_id_at_start=run_context.best_known_run_id_at_start,
        hypothesis=run_context.hypothesis,
        mutation_summary=run_context.mutation_summary,
        description=run_context.description,
        architecture=summary_payload.get("architecture"),
        decision=asdict(decision),
        metrics=_metric_view_payload(
            analysis_metric_bundle,
            experiment_mode=summary.experiment_mode,
            compared_summary=compared_summary,
            base_summary=base_summary,
        ),
        diagnostics=_analysis_diagnostics_payload(diagnostics),
        rule_based_interpretation=_build_interpretation_bullets(
            decision=decision,
            metric_bundle=interpretation_metric_bundle,
            current_diagnostics=diagnostics,
            compared_diagnostics=compared_diagnostics,
            experiment_mode=summary.experiment_mode,
        ),
        failure=None,
        analysis_constraints=_analysis_constraints_payload(analysis_mode="metric_comparison"),
    )


def _build_failure_analysis_input_record(
    *,
    state: SessionState,
    run_context: RunContext,
    decision: DecisionRecord,
    failure_payload: dict[str, Any],
) -> AnalysisInputRecord:
    return AnalysisInputRecord(
        schema_version=ANALYSIS_SCHEMA_VERSION,
        analysis_mode="failure_review",
        session_id=run_context.session_id,
        run_id=run_context.run_id,
        session_run_index=run_context.session_run_index,
        run_role=run_context.run_role,
        decision_status=decision.decision_status,
        compared_against_run_id=run_context.compared_against_run_id,
        base_run_id=state.base_run_id,
        best_known_run_id_at_start=run_context.best_known_run_id_at_start,
        hypothesis=run_context.hypothesis,
        mutation_summary=run_context.mutation_summary,
        description=run_context.description,
        architecture=run_context.architecture_spec,
        decision=asdict(decision),
        metrics={},
        diagnostics={},
        rule_based_interpretation=[],
        failure={
            "error_type": failure_payload.get("error_type"),
            "error_message": failure_payload.get("error_message"),
            "timestamp": failure_payload.get("timestamp"),
        },
        analysis_constraints=_analysis_constraints_payload(analysis_mode="failure_review"),
    )


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
    if decision_status != "keep":
        return [
            f"Probe a nearby width schedule around {hidden_dims} without changing more than one width.",
            f"Test a small dropout adjustment around {dropout} while keeping family={family}.",
            "If the result was not kept, revert to the incumbent and change only one architectural axis next.",
        ]
    return [
        f"Explore one neighboring width variant around {hidden_dims}.",
        f"Test a nearby dropout value around {dropout}.",
        "Try one simplification step and keep it only if the primary metric still improves.",
    ]


def _load_agent_analysis_payload(run_dir: Path) -> dict[str, Any] | None:
    agent_analysis_path = _agent_analysis_path(run_dir)
    if not agent_analysis_path.exists():
        return None
    return _read_json(agent_analysis_path)


def _format_agent_analysis_lines(agent_analysis_payload: dict[str, Any] | None) -> list[str]:
    if agent_analysis_payload is None:
        return [
            "- No agent analysis recorded yet. Run `python -m autoresirch.session_manager.cli analyze-run` before syncing the incumbent."
        ]

    likely_helped = agent_analysis_payload.get("likely_helped") or []
    likely_hurt = agent_analysis_payload.get("likely_hurt") or []
    helped_text = "; ".join(str(item) for item in likely_helped) if likely_helped else "n/a"
    hurt_text = "; ".join(str(item) for item in likely_hurt) if likely_hurt else "n/a"
    return [
        f"- Summary label: `{agent_analysis_payload.get('summary_label', 'n/a')}`",
        f"- Confidence: `{float(agent_analysis_payload.get('confidence', 0.0)):.2f}`",
        f"- Analysis: {agent_analysis_payload.get('freeform_analysis', '')}",
        f"- Likely helped: {helped_text}",
        f"- Likely hurt: {hurt_text}",
        f"- Next-step reasoning: {agent_analysis_payload.get('next_step_reasoning', '')}",
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
    agent_analysis_payload = _load_agent_analysis_payload(run_dir)
    compared_summary = (
        None if compared_summary_payload is None else _summary_from_payload(compared_summary_payload)
    )
    base_summary = (
        None
        if session_base_summary_payload is None
        else _summary_from_payload(session_base_summary_payload)
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
        (
            f"- Architecture: `{architecture_payload}`"
            if architecture_payload is not None
            else "- Architecture: unavailable"
        ),
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
                "## Interpretation",
                "",
                "### Agent Analysis",
            ]
        )
        lines.extend(_format_agent_analysis_lines(agent_analysis_payload))
        lines.extend(
            [
                "",
                "## Next-Run Guidance",
                "- Restore the incumbent with `python -m autoresirch.session_manager.cli sync-incumbent` before the next mutation.",
            ]
        )
    else:
        metric_bundle = _build_metric_delta_bundle(
            summary,
            compared_summary,
            base_summary,
        )
        for metric_name in interpretation_metric_order_for_mode(summary.experiment_mode):
            lines.append(
                _format_metric_movement_line(
                    metric_bundle[metric_name],
                    experiment_mode=summary.experiment_mode,
                )
            )
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
                "",
                "### Rule-Based Interpretation",
            ]
        )
        lines.extend(
            _build_interpretation_bullets(
                decision=decision,
                metric_bundle=metric_bundle,
                current_diagnostics=diagnostics,
                compared_diagnostics=compared_diagnostics,
                experiment_mode=summary.experiment_mode,
            )
        )
        lines.extend(["", "### Agent Analysis"])
        lines.extend(_format_agent_analysis_lines(agent_analysis_payload))
        lines.extend(["", "## Next-Run Guidance"])
        for suggestion in _suggest_next_mutations(
            architecture_payload,
            decision_status=decision.decision_status,
        ):
            lines.append(f"- {suggestion}")

    (run_dir / "synopsis.md").write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


def _run_record_from_state(
    state: SessionState,
    run_id: str,
) -> tuple[RunContext, dict[str, Any] | None, DecisionRecord | None]:
    run_dir = Path(state.run_dirs[run_id])
    run_context = RunContext(**_read_json(run_dir / "run_context.json"))
    summary_payload = _load_run_summary(run_dir) if (run_dir / "summary.json").exists() else None
    decision_path = run_dir / "decision.json"
    decision = DecisionRecord(**_read_json(decision_path)) if decision_path.exists() else None
    return run_context, summary_payload, decision


def _write_analysis_input(run_dir: Path, record: AnalysisInputRecord) -> None:
    _write_json(_analysis_input_path(run_dir), asdict(record))


def _regenerate_run_synopsis(session_id: str, run_id: str) -> None:
    state = load_session_state(session_id)
    run_context, summary_payload, decision = _run_record_from_state(state, run_id)
    if decision is None:
        raise FileNotFoundError(f"Decision artifact not found for run {run_id}.")
    run_dir = Path(run_context.run_dir)
    write_run_synopsis(
        run_dir=run_dir,
        run_context=run_context,
        decision=decision,
        summary=None if summary_payload is None else _summary_from_payload(summary_payload),
        summary_payload=summary_payload,
        compared_summary_payload=_compared_summary_payload(state, run_context),
        session_base_summary_payload=_session_base_summary_payload(state),
    )


def _validate_agent_analysis_fields(
    *,
    analysis_input: AnalysisInputRecord,
    summary_label: str,
    freeform_analysis: str,
    likely_helped: Sequence[str],
    likely_hurt: Sequence[str],
    confidence: float,
    next_step_reasoning: str,
) -> tuple[str, str, list[str], list[str], str]:
    cleaned_summary_label = " ".join(summary_label.strip().split())
    if not cleaned_summary_label:
        raise ValueError("summary_label must not be empty.")
    if len(cleaned_summary_label) > 80:
        raise ValueError("summary_label must be 80 characters or fewer.")

    cleaned_freeform_analysis = " ".join(freeform_analysis.strip().split())
    min_words = (
        AGENT_ANALYSIS_MIN_WORDS
        if analysis_input.analysis_mode == "metric_comparison"
        else AGENT_ANALYSIS_FAILURE_MIN_WORDS
    )
    freeform_word_count = _word_count(cleaned_freeform_analysis)
    if freeform_word_count < min_words or freeform_word_count > AGENT_ANALYSIS_MAX_WORDS:
        raise ValueError(
            "freeform_analysis must stay within the configured word bounds for the analysis mode."
        )

    cleaned_likely_helped = _normalize_analysis_list(likely_helped)
    cleaned_likely_hurt = _normalize_analysis_list(likely_hurt)
    if len(cleaned_likely_helped) > AGENT_ANALYSIS_MAX_FACTORS:
        raise ValueError("Too many likely_helped items were provided.")
    if len(cleaned_likely_hurt) > AGENT_ANALYSIS_MAX_FACTORS:
        raise ValueError("Too many likely_hurt items were provided.")
    for item in [*cleaned_likely_helped, *cleaned_likely_hurt]:
        if _word_count(item) > AGENT_ANALYSIS_MAX_FACTOR_WORDS:
            raise ValueError("Each likely_helped/likely_hurt item must stay concise.")

    if not 0.0 <= confidence <= 1.0:
        raise ValueError("confidence must be between 0.0 and 1.0.")

    cleaned_next_step_reasoning = " ".join(next_step_reasoning.strip().split())
    if not cleaned_next_step_reasoning:
        raise ValueError("next_step_reasoning must not be empty.")
    if _word_count(cleaned_next_step_reasoning) > AGENT_ANALYSIS_MAX_NEXT_STEP_WORDS:
        raise ValueError("next_step_reasoning must stay concise.")

    return (
        cleaned_summary_label,
        cleaned_freeform_analysis,
        cleaned_likely_helped,
        cleaned_likely_hurt,
        cleaned_next_step_reasoning,
    )


def record_agent_analysis(
    *,
    session_id: str,
    run_id: str,
    summary_label: str,
    freeform_analysis: str,
    likely_helped: Sequence[str],
    likely_hurt: Sequence[str],
    confidence: float,
    next_step_reasoning: str,
) -> AgentAnalysisRecord:
    state = load_session_state(session_id)
    run_dir_value = state.run_dirs.get(run_id)
    if run_dir_value is None:
        raise ValueError(f"Run {run_id!r} was not found in session {session_id!r}.")

    run_dir = Path(run_dir_value)
    analysis_input_path = _analysis_input_path(run_dir)
    if not analysis_input_path.exists():
        raise FileNotFoundError(f"Analysis input not found at {analysis_input_path}.")

    analysis_input = AnalysisInputRecord(**_read_json(analysis_input_path))
    (
        cleaned_summary_label,
        cleaned_freeform_analysis,
        cleaned_likely_helped,
        cleaned_likely_hurt,
        cleaned_next_step_reasoning,
    ) = _validate_agent_analysis_fields(
        analysis_input=analysis_input,
        summary_label=summary_label,
        freeform_analysis=freeform_analysis,
        likely_helped=likely_helped,
        likely_hurt=likely_hurt,
        confidence=confidence,
        next_step_reasoning=next_step_reasoning,
    )

    agent_analysis = AgentAnalysisRecord(
        schema_version=ANALYSIS_SCHEMA_VERSION,
        session_id=session_id,
        run_id=run_id,
        analysis_mode=analysis_input.analysis_mode,
        created_at=_utc_now_iso(),
        analysis_input_sha256=_sha256_path(analysis_input_path),
        decision_status=analysis_input.decision_status,
        summary_label=cleaned_summary_label,
        freeform_analysis=cleaned_freeform_analysis,
        likely_helped=cleaned_likely_helped,
        likely_hurt=cleaned_likely_hurt,
        confidence=confidence,
        next_step_reasoning=cleaned_next_step_reasoning,
    )
    _write_json(_agent_analysis_path(run_dir), asdict(agent_analysis))
    _regenerate_run_synopsis(session_id, run_id)
    return agent_analysis


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
    run_records = [_run_record_from_state(state, run_id) for run_id in run_ids]
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
    best_secondary_snapshot: dict[str, float | None] = {}
    if best_run_payload is not None:
        best_secondary_snapshot = _summary_secondary_metrics(_summary_from_payload(best_run_payload))

    hypotheses_tested = [
        run_context.hypothesis for run_context, _, _ in run_records if run_context.hypothesis
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
            if decision is not None
            and decision.decision_status == "keep"
            and run_context.mutation_summary
        )
    )[:5]
    patterns_that_hurt = list(
        dict.fromkeys(
            run_context.mutation_summary
            for run_context, _, decision in run_records
            if decision is not None
            and decision.decision_status in {"discard", "crash"}
            and run_context.mutation_summary
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

    completed_at = state.completed_at or _utc_now_iso()
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
                {
                    max(hidden_dims)
                    for payload in architectures
                    for hidden_dims in [payload.get("hidden_dims", [])]
                    if hidden_dims
                }
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
                f"({_format_metric_delta(entry['delta_vs_previous_incumbent'])} vs previous incumbent)"
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
