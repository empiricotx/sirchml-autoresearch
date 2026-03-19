from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Protocol

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
ExperimentMode = Literal["standard", "comparative"]

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
    rnafm_sequence_column: str = "antisense_strand_seq"
    rnafm_embedding_dim: int = 16
    experiment_mode: ExperimentMode = "standard"


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
    early_stopping_patience: int | None = 20
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass(frozen=True)
class MetricConfig:
    primary_metric_name: str = "weighted_cv_auc"
    primary_metric_direction: str = "higher_is_better"
    improvement_epsilon: float = 1e-4
    prediction_scale_min: float = 0.45
    prediction_scale_max: float = 0.9
    effective_threshold: float = 0.4
    comparative_no_effect_lower: float = -0.2
    comparative_no_effect_upper: float = 0.2
    comparative_auc_strategy: str = "ovr"


@dataclass(frozen=True)
class ArchitectureConstraints:
    allowed_families: tuple[str, ...] = ("mlp", "residual_mlp", "cnn", "hybrid_cnn_mlp")
    allowed_activations: tuple[str, ...] = ("relu", "gelu", "silu")
    allowed_normalizations: tuple[str, ...] = ("none", "layernorm", "batchnorm")
    allowed_pooling: tuple[str, ...] = ("mean", "max")
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
    "commit\texperiment_mode\tprimary_metric_name\tprimary_metric_value\tweighted_cv_rmse_mean\t"
    "cv_rmse_std\tweighted_cv_auc\tweighted_cv_overall_auc\tweighted_cv_auc_pos_vs_neg\t"
    "weighted_cv_pearson_r\tweighted_cv_spearman_r\tstatus\tnum_params\ttrain_seconds\t"
    "description\n"
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
    hidden_dims: tuple[int, ...] = ()
    activation: str = "silu"
    dropout: float = 0.1
    normalization: str = "layernorm"
    use_bias: bool = True
    use_rnafm_embeddings: bool = False
    sequence_feature_source: str | None = None
    conv_channels: tuple[int, ...] = ()
    kernel_sizes: tuple[int, ...] = ()
    pooling: str = "mean"
    sequence_encoder_dim: int | None = None
    flat_hidden_dims: tuple[int, ...] = ()
    fusion_hidden_dims: tuple[int, ...] = ()
    rnafm_embedding_dim: int | None = None
    rnafm_pooling_strategy: str = "none"


@dataclass(frozen=True)
class ArchitectureContext:
    input_dim: int
    output_dim: int
    train_size: int
    feature_names: tuple[str, ...]
    device: str
    flat_input_dim: int | None = None
    sequence_length: int | None = None
    sequence_channels: int | None = None
    sequence_embedding_dim: int | None = None
    has_flat_features: bool = True
    has_sequence_features: bool = False
    flat_feature_names: tuple[str, ...] = ()
    sequence_source_name: str | None = None


class ModelBuilder(Protocol):
    def __call__(self, context: ArchitectureContext) -> nn.Module:
        ...


@dataclass
class PreparedDataset:
    flat_features: pd.DataFrame | None
    sequence_features: np.ndarray | None
    target: np.ndarray
    genes: np.ndarray
    row_ids: np.ndarray
    numeric_feature_columns: tuple[str, ...]
    categorical_feature_columns: tuple[str, ...]
    train_genes: tuple[str, ...]
    test_genes: tuple[str, ...]
    cv_genes: tuple[str, ...]
    source_path: str
    sequence_feature_name: str | None = None
    sequence_feature_shape: tuple[int, ...] | None = None
    experiment_mode: ExperimentMode = "standard"
    target_class: np.ndarray | None = None
    left_row_ids: np.ndarray | None = None
    right_row_ids: np.ndarray | None = None

    @property
    def features(self) -> pd.DataFrame:
        if self.flat_features is None:
            return pd.DataFrame(index=pd.RangeIndex(len(self.target)))
        return self.flat_features

    @property
    def has_flat_features(self) -> bool:
        return self.flat_features is not None and not self.flat_features.empty

    @property
    def has_sequence_features(self) -> bool:
        return self.sequence_features is not None

    @property
    def feature_dim(self) -> int:
        flat_dim = 0 if self.flat_features is None else int(self.flat_features.shape[1])
        if self.sequence_features is None:
            return flat_dim
        return flat_dim + int(np.prod(self.sequence_features.shape[1:], dtype=np.int64))


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
    overall_auc: float | None = None
    auc_class_neg1: float | None = None
    auc_class_0: float | None = None
    auc_class_pos1: float | None = None
    auc_pos_vs_neg: float | None = None


@dataclass(frozen=True)
class FoldDiagnostics:
    scaled_prediction_mean: float | None = None
    scaled_prediction_std: float | None = None
    clipped_low_fraction: float | None = None
    clipped_high_fraction: float | None = None
    effective_positive_rate: float | None = None
    class_count_neg1: int | None = None
    class_count_0: int | None = None
    class_count_pos1: int | None = None
    undefined_auc_metrics: tuple[str, ...] = ()


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
    experiment_mode: ExperimentMode = "standard"
    label_threshold_lower: float | None = None
    label_threshold_upper: float | None = None
    weighted_cv_overall_auc: float | None = None
    weighted_cv_auc_class_neg1: float | None = None
    weighted_cv_auc_class_0: float | None = None
    weighted_cv_auc_class_pos1: float | None = None
    weighted_cv_auc_pos_vs_neg: float | None = None
    test_overall_auc: float | None = None
    test_auc_class_neg1: float | None = None
    test_auc_class_0: float | None = None
    test_auc_class_pos1: float | None = None
    test_auc_pos_vs_neg: float | None = None


@dataclass(frozen=True)
class LoadedArchitecture:
    spec: ArchitectureSpec
    build_model: ModelBuilder
    module_name: str
