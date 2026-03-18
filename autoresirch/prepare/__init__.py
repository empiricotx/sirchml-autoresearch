from .data import build_rnafm_embedding_tensor
from .architecture_loading import (
    _extract_loaded_architecture,
    load_train_definition,
    validate_architecture_spec,
    validate_train_source,
)
from .dataset_preparation import (
    build_prepared_dataset_from_frame,
    build_sequence_feature_frame,
    choose_cv_genes,
    choose_test_genes,
    infer_feature_columns,
    normalize_gene_label,
    normalize_sequence,
    prepare_dataset,
    print_dataset_summary,
    read_raw_dataframe,
)
from .fold_preprocessor import FoldPreprocessor, TargetScaler, build_cv_folds
from .orchestration import print_experiment_summary, run_experiment, save_run_summary
from .schemas import (
    ALLOWED_TRAIN_IMPORTS,
    ARCHITECTURE_CONSTRAINTS,
    CACHE_DIR,
    DATASET_CONFIG,
    DATA_DIR,
    EDITABLE_TRAIN_FILE,
    FORBIDDEN_ATTRIBUTE_NAMES,
    FORBIDDEN_CALL_NAMES,
    METRIC_CONFIG,
    RESULTS_HEADER,
    RESULTS_TSV,
    REPO_ROOT,
    RUNS_DIR,
    SPLIT_CONFIG,
    TRAINING_CONFIG,
    ArchitectureConstraints,
    ArchitectureContext,
    ArchitectureSpec,
    DatasetConfig,
    ExperimentSummary,
    FoldDiagnostics,
    FoldResult,
    FoldSpec,
    LoadedArchitecture,
    MetricConfig,
    ModelBuilder,
    PreparedDataset,
    RegressionMetrics,
    SplitConfig,
    TrainingConfig,
)
from .training_harness import (
    _is_defined,
    _pick_fold_by_metric,
    _train_epoch,
    aggregate_fold_results,
    build_run_diagnostics,
    create_dataloader,
    instantiate_model,
    predict_regression,
    train_final_holdout,
    train_fold,
    validate_budget,
)
from .utils import (
    _config_fingerprint,
    _json_default,
    _make_run_dir,
    _normalize_model_output,
    _state_dict_to_cpu,
    binary_effective_labels,
    build_fold_diagnostics,
    count_parameters,
    ensure_results_tsv,
    ensure_runtime_dirs,
    evaluate_predictions,
    mae,
    pearson_r_score,
    r2_score,
    rmse,
    roc_auc_score_binary,
    scale_regression_predictions,
    set_random_seed,
    spearman_r_score,
)


def parse_args():
    from .cli import parse_args as cli_parse_args

    return cli_parse_args()


def main():
    from .cli import main as cli_main

    return cli_main()

__all__ = [
    # Schemas
    "DatasetConfig",
    "SplitConfig",
    "TrainingConfig",
    "MetricConfig",
    "ArchitectureConstraints",
    "ArchitectureSpec",
    "ArchitectureContext",
    "ModelBuilder",
    "PreparedDataset",
    "FoldSpec",
    "RegressionMetrics",
    "FoldDiagnostics",
    "FoldResult",
    "ExperimentSummary",
    "LoadedArchitecture",

    # Utils
    "_json_default",
    "_config_fingerprint",
    "ensure_runtime_dirs",
    "ensure_results_tsv",
    "set_random_seed",
    "count_parameters",
    "rmse",
    "mae",
    "r2_score",
    "pearson_r_score",
    "spearman_r_score",
    "scale_regression_predictions",
    "binary_effective_labels",
    "build_fold_diagnostics",
    "roc_auc_score_binary",
    "evaluate_predictions",
    "_state_dict_to_cpu",
    "_normalize_model_output",
    "_make_run_dir",
    "build_rnafm_embedding_tensor",

    # Dataset Preparation
    "read_raw_dataframe",
    "normalize_sequence",
    "build_sequence_feature_frame",
    "infer_feature_columns",
    "normalize_gene_label",
    "choose_test_genes",
    "choose_cv_genes",
    "build_prepared_dataset_from_frame",
    "prepare_dataset",
    "print_dataset_summary",

    # Fold Preprocessor
    "FoldPreprocessor",
    "TargetScaler",
    "build_cv_folds",

    # Architecture Loading
    "validate_architecture_spec",
    "validate_train_source",
    "load_train_definition",
    "_extract_loaded_architecture",

    # Training Harness
    "instantiate_model",
    "create_dataloader",
    "predict_regression",
    "_train_epoch",
    "train_fold",
    "train_final_holdout",
    "aggregate_fold_results",
    "_is_defined",
    "_pick_fold_by_metric",
    "validate_budget",
    "build_run_diagnostics",

    # Orchestration
    "save_run_summary",
    "print_experiment_summary",
    "run_experiment",

    # CLI
    "parse_args",
    "main",
]
