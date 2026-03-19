from __future__ import annotations

from autoresirch.prepare import METRIC_CONFIG
from autoresirch.session_manager.schemas import InterpretationMetricSpec


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

STANDARD_INTERPRETATION_METRIC_ORDER = (
    "weighted_cv_auc",
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
