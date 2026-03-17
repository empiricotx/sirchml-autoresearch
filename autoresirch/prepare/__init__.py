from .schemas import *
from .utils import *
from .dataset_preparation import *
from .fold_preprocessor import *
from .architecture_loading import *
from .training_harness import *
from .orchestration import *
from .cli import *

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