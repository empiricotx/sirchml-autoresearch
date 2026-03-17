from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd
import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = REPO_ROOT / ".cache" / "sirna_regression"
RUNS_DIR = REPO_ROOT / "runs"
DATA_DIR = REPO_ROOT / "data"
RESULTS_TSV = REPO_ROOT / "results.tsv"
EDITABLE_TRAIN_FILE = REPO_ROOT / "autoresirch" / "train.py"

@dataclass(frozen=True)
class DatasetConfig:
    raw_data_path: Path = DATA_DIR / "feature_subset_df.csv"
    target_column: str = "rel_exp_individual"
    gene_column: str = "transcript_gene"
    gene_normalization: str = "upper"
    sequence_columns: tuple[str, ...] = ()
    numeric_columns: tuple[str, ...] = ()
    categorical_columns: tuple[str, ...] = ()
    drop_columns: tuple[str, ...] = ("antisense_strand_seq",)
    explicit_test_genes: tuple[str, ...] = ()
    explicit_cv_genes: tuple[str, ...] = ()
    test_fraction: float = 0.0
    max_sequence_length: int = 32
    allowed_bases: str = "ACGU"
    unknown_base: str = "N"


@dataclass(frozen=True)
class SplitConfig:
    random_seed: int = 42
    min_train_genes: int = 2
    max_cv_folds: int | None = None


@dataclass(frozen=True)
class TrainingConfig:
    total_time_budget_seconds: float = 90.0
    cv_budget_ratio: float = 0.8
    evaluate_test_split: bool = False
    batch_size: int = 128
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip_norm: float | None = 5.0
    min_fold_budget_seconds: float = 1.0
    min_final_fit_budget_seconds: float = 5.0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass(frozen=True)
class MetricConfig:
    primary_metric_name: str = "weighted_cv_auc"
    primary_metric_direction: str = "higher_is_better"
    improvement_epsilon: float = 1e-4
    prediction_scale_min: float = 0.45
    prediction_scale_max: float = 0.9
    effective_threshold: float = 0.4


@dataclass(frozen=True)
class ArchitectureConstraints:
    allowed_families: tuple[str, ...] = ("mlp", "residual_mlp")
    allowed_activations: tuple[str, ...] = ("relu", "gelu", "silu")
    allowed_normalizations: tuple[str, ...] = ("none", "layernorm", "batchnorm")
    max_hidden_layers: int = 8
    max_hidden_width: int = 1024
    max_dropout: float = 0.5
    max_parameters: int = 2_000_000
    allow_bias: bool = True


DATASET_CONFIG = DatasetConfig()
SPLIT_CONFIG = SplitConfig()
TRAINING_CONFIG = TrainingConfig()
METRIC_CONFIG = MetricConfig()
ARCHITECTURE_CONSTRAINTS = ArchitectureConstraints()

RESULTS_HEADER = (
    "commit\tweighted_cv_rmse_mean\tcv_rmse_std\tweighted_cv_auc\tweighted_cv_pearson_r\tweighted_cv_spearman_r\tstatus\tnum_params\ttrain_seconds\tdescription\n"
)

ALLOWED_TRAIN_IMPORTS = {
    "__future__",
    "math",
    "prepare",
    "autoresirch.prepare",
    "torch",
    "torch.nn",
    "torch.nn.functional",
}
FORBIDDEN_CALL_NAMES = {"open", "eval", "exec", "compile", "__import__"}
FORBIDDEN_ATTRIBUTE_NAMES = {
    "check_call",
    "check_output",
    "mkdir",
    "popen",
    "rename",
    "replace",
    "rmdir",
    "run",
    "system",
    "unlink",
    "write_bytes",
    "write_text",
}


# ---------------------------------------------------------------------------
# Data contracts
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ArchitectureSpec:
    family: str
    hidden_dims: tuple[int, ...]
    activation: str = "silu"
    dropout: float = 0.1
    normalization: str = "layernorm"
    use_bias: bool = True


@dataclass(frozen=True)
class ArchitectureContext:
    input_dim: int
    output_dim: int
    train_size: int
    feature_names: tuple[str, ...]
    device: str


class ModelBuilder(Protocol):
    def __call__(self, context: ArchitectureContext) -> nn.Module:
        ...


@dataclass
class PreparedDataset:
    features: pd.DataFrame
    target: np.ndarray
    genes: np.ndarray
    row_ids: np.ndarray
    numeric_feature_columns: tuple[str, ...]
    categorical_feature_columns: tuple[str, ...]
    train_genes: tuple[str, ...]
    test_genes: tuple[str, ...]
    cv_genes: tuple[str, ...]
    source_path: str


@dataclass(frozen=True)
class FoldSpec:
    gene: str
    train_indices: np.ndarray
    val_indices: np.ndarray


@dataclass(frozen=True)
class RegressionMetrics:
    rmse: float
    mae: float
    r2: float
    squared_error_sum: float
    auc: float | None
    pearson_r: float | None
    spearman_r: float | None


@dataclass(frozen=True)
class FoldDiagnostics:
    scaled_prediction_mean: float | None = None
    scaled_prediction_std: float | None = None
    clipped_low_fraction: float | None = None
    clipped_high_fraction: float | None = None
    effective_positive_rate: float | None = None


@dataclass(frozen=True)
class FoldResult:
    gene: str
    count: int
    metrics: RegressionMetrics
    train_seconds: float
    epochs: int
    best_epoch: int
    num_params: int
    diagnostics: FoldDiagnostics = field(default_factory=FoldDiagnostics)


@dataclass(frozen=True)
class ExperimentSummary:
    primary_metric_name: str
    primary_metric_value: float
    metric_direction: str
    improvement_epsilon: float
    weighted_cv_rmse_mean: float
    cv_rmse_mean: float
    cv_rmse_std: float
    weighted_cv_mae_mean: float
    weighted_cv_r2_mean: float | None
    weighted_cv_auc_mean: float | None
    weighted_cv_pearson_r_mean: float | None
    weighted_cv_spearman_r_mean: float | None
    pooled_cv_rmse: float
    test_rmse: float | None
    test_mae: float | None
    test_r2: float | None
    test_auc: float | None
    test_pearson_r: float | None
    test_spearman_r: float | None
    num_params: int
    train_seconds: float
    feature_dim: int
    num_rows: int
    cv_folds: int
    train_genes: tuple[str, ...]
    test_genes: tuple[str, ...]
    cv_genes: tuple[str, ...]
    run_dir: str


@dataclass(frozen=True)
class LoadedArchitecture:
    spec: ArchitectureSpec
    build_model: ModelBuilder
    module_name: str
