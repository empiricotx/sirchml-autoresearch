from __future__ import annotations

from autoresirch.prepare import ExperimentSummary
from autoresirch.session_manager.comparative.metrics import (
    COMPARATIVE_ANALYSIS_INPUT_METRIC_ORDER,
    COMPARATIVE_INTERPRETATION_METRIC_ORDER,
    COMPARATIVE_INTERPRETATION_METRIC_SPECS,
)
from autoresirch.session_manager.standard.metrics import (
    STANDARD_ANALYSIS_INPUT_METRIC_ORDER,
    STANDARD_INTERPRETATION_METRIC_ORDER,
    STANDARD_INTERPRETATION_METRIC_SPECS,
)


SESSION_RESULTS_HEADER = (
    "session_id\tsession_run_index\trun_id\trun_role\tparent_run_id\tcompared_against_run_id\t"
    "commit\texperiment_mode\tprimary_metric_name\tprimary_metric_value\tweighted_cv_rmse_mean\t"
    "cv_rmse_std\tweighted_cv_auc\tweighted_cv_overall_auc\tweighted_cv_auc_pos_vs_neg\t"
    "weighted_cv_pearson_r\tweighted_cv_spearman_r\tstatus\tnum_params\ttrain_seconds\t"
    "decision_baseline_value\tdecision_delta\thypothesis\tmutation_summary\tdescription\trun_dir\n"
)



def interpretation_metric_specs_for_mode(
    experiment_mode: str,
) -> dict[str, InterpretationMetricSpec]:
    if experiment_mode == "comparative":
        return COMPARATIVE_INTERPRETATION_METRIC_SPECS
    return STANDARD_INTERPRETATION_METRIC_SPECS


def interpretation_metric_order_for_mode(experiment_mode: str) -> tuple[str, ...]:
    if experiment_mode == "comparative":
        return COMPARATIVE_INTERPRETATION_METRIC_ORDER
    return STANDARD_INTERPRETATION_METRIC_ORDER


def analysis_input_metric_order_for_mode(experiment_mode: str) -> tuple[str, ...]:
    if experiment_mode == "comparative":
        return COMPARATIVE_ANALYSIS_INPUT_METRIC_ORDER
    return STANDARD_ANALYSIS_INPUT_METRIC_ORDER


def primary_metric_name_for_mode(experiment_mode: str) -> str:
    if experiment_mode == "comparative":
        return "weighted_cv_overall_auc"
    return "weighted_cv_auc"


def primary_metric_label_for_summary(summary: ExperimentSummary) -> str:
    return primary_metric_name_for_mode(summary.experiment_mode)
