from __future__ import annotations

from pathlib import Path

from autoresirch.prepare import METRIC_CONFIG
from autoresirch.session_manager.schemas import InterpretationMetricSpec


REPO_ROOT = Path(__file__).resolve().parents[2]
SESSIONS_DIR = REPO_ROOT / "sessions"
EDITABLE_TRAIN_FILE = REPO_ROOT / "autoresirch" / "train.py"
PROGRAM_FILE = REPO_ROOT / "autoresirch" / "program.md"
RUN_LOG = REPO_ROOT / "run.log"

SESSION_RESULTS_HEADER = (
    "session_id\tsession_run_index\trun_id\trun_role\tparent_run_id\tcompared_against_run_id\t"
    "commit\tweighted_cv_rmse_mean\tcv_rmse_std\tweighted_cv_auc\tweighted_cv_pearson_r\t"
    "weighted_cv_spearman_r\tstatus\tnum_params\ttrain_seconds\tdecision_baseline_value\t"
    "decision_delta\thypothesis\tmutation_summary\tdescription\trun_dir\n"
)

INTERPRETATION_NEAR_MISS_AUC_DELTA = 0.005
INTERPRETATION_AUC_SPAN_DELTA = 0.08
INTERPRETATION_RMSE_SPAN_DELTA = 0.05
ANALYSIS_SCHEMA_VERSION = 1
AGENT_ANALYSIS_MAX_WORDS = 180
AGENT_ANALYSIS_MIN_WORDS = 15
AGENT_ANALYSIS_FAILURE_MIN_WORDS = 5
AGENT_ANALYSIS_MAX_FACTORS = 3
AGENT_ANALYSIS_MAX_FACTOR_WORDS = 16
AGENT_ANALYSIS_MAX_NEXT_STEP_WORDS = 80

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

INTERPRETATION_METRIC_ORDER = (
    "weighted_cv_auc",
    "weighted_cv_rmse_mean",
    "weighted_cv_pearson_r_mean",
    "weighted_cv_spearman_r_mean",
    "cv_rmse_std",
    "num_params",
    "train_seconds",
)

ANALYSIS_INPUT_METRIC_ORDER = (
    *INTERPRETATION_METRIC_ORDER,
    "weighted_cv_mae_mean",
    "weighted_cv_r2_mean",
    "pooled_cv_rmse",
)
