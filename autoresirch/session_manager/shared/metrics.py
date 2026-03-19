from __future__ import annotations

from autoresirch.prepare import METRIC_CONFIG, ExperimentSummary
from autoresirch.session_manager.schemas import InterpretationMetricSpec


SESSION_RESULTS_HEADER = (
    "session_id\tsession_run_index\trun_id\trun_role\tparent_run_id\tcompared_against_run_id\t"
    "commit\texperiment_mode\tprimary_metric_name\tprimary_metric_value\tweighted_cv_rmse_mean\t"
    "cv_rmse_std\tweighted_cv_auc\tweighted_cv_overall_auc\tweighted_cv_auc_pos_vs_neg\t"
    "weighted_cv_pearson_r\tweighted_cv_spearman_r\tstatus\tnum_params\ttrain_seconds\t"
    "decision_baseline_value\tdecision_delta\thypothesis\tmutation_summary\tdescription\trun_dir\n"
)

STANDARD_INTERPRETATION_METRIC_SPECS: dict[str, InterpretationMetricSpec] = {
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
    "weighted_cv_mae_mean": InterpretationMetricSpec(
        summary_attr="weighted_cv_mae_mean",
        display_name="Weighted CV MAE",
        direction="lower",
        flat_threshold=0.002,
    ),
    "weighted_cv_r2_mean": InterpretationMetricSpec(
        summary_attr="weighted_cv_r2_mean",
        display_name="Weighted CV R2",
        direction="higher",
        flat_threshold=0.01,
    ),
    "pooled_cv_rmse": InterpretationMetricSpec(
        summary_attr="pooled_cv_rmse",
        display_name="Pooled CV RMSE",
        direction="lower",
        flat_threshold=0.002,
    ),
}

COMPARATIVE_INTERPRETATION_METRIC_SPECS: dict[str, InterpretationMetricSpec] = {
    "weighted_cv_overall_auc": InterpretationMetricSpec(
        summary_attr="primary_metric_value",
        display_name="Weighted CV Overall AUC",
        direction="higher",
        flat_threshold=max(METRIC_CONFIG.improvement_epsilon * 10.0, 0.001),
    ),
    "weighted_cv_auc_pos_vs_neg": InterpretationMetricSpec(
        summary_attr="weighted_cv_auc_pos_vs_neg",
        display_name="Weighted CV Pos-vs-Neg AUC",
        direction="higher",
        flat_threshold=0.01,
    ),
    "weighted_cv_auc_class_neg1": InterpretationMetricSpec(
        summary_attr="weighted_cv_auc_class_neg1",
        display_name="Weighted CV Class -1 AUC",
        direction="higher",
        flat_threshold=0.01,
    ),
    "weighted_cv_auc_class_0": InterpretationMetricSpec(
        summary_attr="weighted_cv_auc_class_0",
        display_name="Weighted CV Class 0 AUC",
        direction="higher",
        flat_threshold=0.01,
    ),
    "weighted_cv_auc_class_pos1": InterpretationMetricSpec(
        summary_attr="weighted_cv_auc_class_pos1",
        display_name="Weighted CV Class +1 AUC",
        direction="higher",
        flat_threshold=0.01,
    ),
    "weighted_cv_rmse_mean": InterpretationMetricSpec(
        summary_attr="weighted_cv_rmse_mean",
        display_name="Weighted CV RMSE",
        direction="lower",
        flat_threshold=0.01,
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
    "weighted_cv_mae_mean": InterpretationMetricSpec(
        summary_attr="weighted_cv_mae_mean",
        display_name="Weighted CV MAE",
        direction="lower",
        flat_threshold=0.01,
    ),
    "weighted_cv_r2_mean": InterpretationMetricSpec(
        summary_attr="weighted_cv_r2_mean",
        display_name="Weighted CV R2",
        direction="higher",
        flat_threshold=0.01,
    ),
    "pooled_cv_rmse": InterpretationMetricSpec(
        summary_attr="pooled_cv_rmse",
        display_name="Pooled CV RMSE",
        direction="lower",
        flat_threshold=0.01,
    ),
}

STANDARD_INTERPRETATION_METRIC_ORDER = (
    "weighted_cv_auc",
    "weighted_cv_rmse_mean",
    "weighted_cv_pearson_r_mean",
    "weighted_cv_spearman_r_mean",
    "cv_rmse_std",
    "num_params",
    "train_seconds",
)

COMPARATIVE_INTERPRETATION_METRIC_ORDER = (
    "weighted_cv_overall_auc",
    "weighted_cv_auc_pos_vs_neg",
    "weighted_cv_rmse_mean",
    "weighted_cv_pearson_r_mean",
    "weighted_cv_spearman_r_mean",
    "cv_rmse_std",
    "num_params",
    "train_seconds",
)

STANDARD_ANALYSIS_INPUT_METRIC_ORDER = (
    *STANDARD_INTERPRETATION_METRIC_ORDER,
    "weighted_cv_mae_mean",
    "weighted_cv_r2_mean",
    "pooled_cv_rmse",
)

COMPARATIVE_ANALYSIS_INPUT_METRIC_ORDER = (
    *COMPARATIVE_INTERPRETATION_METRIC_ORDER,
    "weighted_cv_auc_class_neg1",
    "weighted_cv_auc_class_0",
    "weighted_cv_auc_class_pos1",
    "weighted_cv_mae_mean",
    "weighted_cv_r2_mean",
    "pooled_cv_rmse",
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
