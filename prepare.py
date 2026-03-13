from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import math
import pickle
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Protocol

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

if __name__ == "__main__":
    sys.modules.setdefault("prepare", sys.modules[__name__])

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent
CACHE_DIR = REPO_ROOT / ".cache" / "sirna_regression"
RUNS_DIR = REPO_ROOT / "runs"
DATA_DIR = REPO_ROOT / "data"
RESULTS_TSV = REPO_ROOT / "results.tsv"
EDITABLE_TRAIN_FILE = REPO_ROOT / "train.py"

# ---------------------------------------------------------------------------
# Fixed configuration (human-editable, agent must not touch)
# ---------------------------------------------------------------------------


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
    primary_metric_name: str = "weighted_cv_rmse_mean"
    primary_metric_direction: str = "lower_is_better"
    improvement_epsilon: float = 1e-4


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
    "commit\tweighted_cv_rmse_mean\tcv_rmse_std\tstatus\tnum_params\ttrain_seconds\tdescription\n"
)

ALLOWED_TRAIN_IMPORTS = {
    "__future__",
    "math",
    "prepare",
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


@dataclass(frozen=True)
class FoldResult:
    gene: str
    count: int
    metrics: RegressionMetrics
    train_seconds: float
    epochs: int
    best_epoch: int
    num_params: int


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
    pooled_cv_rmse: float
    test_rmse: float | None
    test_mae: float | None
    test_r2: float | None
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


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    raise TypeError(f"Value is not JSON serializable: {type(value)!r}")


def _config_fingerprint() -> str:
    payload = {
        "dataset": asdict(DATASET_CONFIG),
        "split": asdict(SPLIT_CONFIG),
        "training": asdict(TRAINING_CONFIG),
        "metric": asdict(METRIC_CONFIG),
        "constraints": asdict(ARCHITECTURE_CONSTRAINTS),
    }
    return json.dumps(payload, sort_keys=True, default=_json_default)


def ensure_runtime_dirs() -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    ensure_results_tsv()


def ensure_results_tsv() -> None:
    if not RESULTS_TSV.exists():
        RESULTS_TSV.write_text(RESULTS_HEADER, encoding="utf-8")


def set_random_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(y_pred - y_true))))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_pred - y_true)))


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size < 2:
        return math.nan
    total = float(np.sum(np.square(y_true - y_true.mean())))
    if total <= 0:
        return math.nan
    residual = float(np.sum(np.square(y_true - y_pred)))
    return 1.0 - (residual / total)


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> RegressionMetrics:
    squared_error_sum = float(np.sum(np.square(y_pred - y_true)))
    return RegressionMetrics(
        rmse=rmse(y_true, y_pred),
        mae=mae(y_true, y_pred),
        r2=r2_score(y_true, y_pred),
        squared_error_sum=squared_error_sum,
    )


def _state_dict_to_cpu(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}


def _normalize_model_output(output: torch.Tensor) -> torch.Tensor:
    if output.ndim == 2 and output.shape[1] == 1:
        return output[:, 0]
    if output.ndim == 1:
        return output
    raise ValueError(
        "Model output must have shape [batch] or [batch, 1]. "
        f"Received {tuple(output.shape)}."
    )


def _make_run_dir() -> Path:
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    run_dir = RUNS_DIR / timestamp
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


# ---------------------------------------------------------------------------
# Dataset preparation
# ---------------------------------------------------------------------------


def read_raw_dataframe(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Raw dataset not found at {path}. Place your siRNA dataset there or update "
            "DATASET_CONFIG.raw_data_path in prepare.py."
        )

    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".tsv":
        return pd.read_csv(path, sep="\t")
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported dataset format: {suffix}")


def normalize_sequence(value: Any, allowed_bases: set[str], unknown_base: str) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    sequence = str(value).strip().upper().replace("T", "U")
    normalized = []
    for char in sequence:
        normalized.append(char if char in allowed_bases else unknown_base)
    return "".join(normalized)


def build_sequence_feature_frame(
    sequences: pd.Series,
    *,
    column_name: str,
    max_length: int,
    allowed_bases: str,
    unknown_base: str,
) -> pd.DataFrame:
    alphabet = tuple(allowed_bases) + (unknown_base,)
    allowed_set = set(allowed_bases)
    normalized = [
        normalize_sequence(value, allowed_set, unknown_base)
        for value in sequences.fillna("")
    ]

    feature_data: dict[str, np.ndarray] = {}
    lengths = np.array([len(seq) for seq in normalized], dtype=np.float32)
    feature_data[f"{column_name}__length"] = lengths

    gc_values = np.array(
        [
            ((seq.count("G") + seq.count("C")) / len(seq)) if seq else 0.0
            for seq in normalized
        ],
        dtype=np.float32,
    )
    feature_data[f"{column_name}__gc_fraction"] = gc_values

    for base in alphabet:
        fractions = np.array(
            [(seq.count(base) / len(seq)) if seq else 0.0 for seq in normalized],
            dtype=np.float32,
        )
        feature_data[f"{column_name}__frac_{base}"] = fractions

    for position in range(max_length):
        for base in alphabet:
            feature_name = f"{column_name}__pos_{position:02d}_{base}"
            feature_data[feature_name] = np.array(
                [
                    1.0 if position < len(seq) and seq[position] == base else 0.0
                    for seq in normalized
                ],
                dtype=np.float32,
            )

    return pd.DataFrame(feature_data, index=sequences.index)


def infer_feature_columns(
    dataframe: pd.DataFrame,
    dataset_config: DatasetConfig,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    reserved = {
        dataset_config.target_column,
        dataset_config.gene_column,
        *dataset_config.sequence_columns,
        *dataset_config.drop_columns,
    }

    if dataset_config.numeric_columns:
        numeric_columns = tuple(dataset_config.numeric_columns)
    else:
        numeric_columns = tuple(
            str(column)
            for column in dataframe.select_dtypes(include=["number", "bool"]).columns
            if column not in reserved
        )

    if dataset_config.categorical_columns:
        categorical_columns = tuple(dataset_config.categorical_columns)
    else:
        categorical_columns = tuple(
            str(column)
            for column in dataframe.columns
            if column not in reserved and column not in numeric_columns
        )

    return numeric_columns, categorical_columns


def normalize_gene_label(value: Any, mode: str) -> str:
    gene = str(value).strip()
    if mode == "identity":
        return gene
    if mode == "upper":
        return gene.upper()
    raise ValueError(f"Unsupported gene normalization mode: {mode!r}")


def choose_test_genes(
    unique_genes: tuple[str, ...],
    dataset_config: DatasetConfig,
    split_config: SplitConfig,
) -> tuple[str, ...]:
    if dataset_config.explicit_test_genes:
        explicit_test_genes = tuple(
            normalize_gene_label(gene, dataset_config.gene_normalization)
            for gene in dataset_config.explicit_test_genes
        )
        test_genes = tuple(
            gene for gene in explicit_test_genes if gene in unique_genes
        )
        if not test_genes:
            raise ValueError("No explicit_test_genes are present in the dataset.")
        return tuple(sorted(test_genes))

    if dataset_config.test_fraction <= 0:
        return ()

    max_test_genes = len(unique_genes) - split_config.min_train_genes
    if max_test_genes <= 0:
        raise ValueError(
            "Not enough unique genes to create a train/test split. "
            "Increase dataset size or lower SplitConfig.min_train_genes."
        )

    requested = max(1, math.ceil(len(unique_genes) * dataset_config.test_fraction))
    requested = min(requested, max_test_genes)

    rng = np.random.default_rng(split_config.random_seed)
    shuffled = list(unique_genes)
    rng.shuffle(shuffled)
    return tuple(sorted(shuffled[:requested]))


def choose_cv_genes(
    train_genes: tuple[str, ...],
    dataset_config: DatasetConfig,
    split_config: SplitConfig,
) -> tuple[str, ...]:
    if dataset_config.explicit_cv_genes:
        explicit_cv_genes = tuple(
            normalize_gene_label(gene, dataset_config.gene_normalization)
            for gene in dataset_config.explicit_cv_genes
        )
        cv_genes = tuple(
            gene for gene in explicit_cv_genes if gene in train_genes
        )
    else:
        cv_genes = train_genes

    if split_config.max_cv_folds is not None:
        cv_genes = cv_genes[: split_config.max_cv_folds]

    if not cv_genes:
        raise ValueError("No genes available for CV after applying the configured constraints.")

    return tuple(cv_genes)


def build_prepared_dataset_from_frame(
    dataframe: pd.DataFrame,
    *,
    dataset_config: DatasetConfig = DATASET_CONFIG,
    split_config: SplitConfig = SPLIT_CONFIG,
) -> PreparedDataset:
    required_columns = {
        dataset_config.target_column,
        dataset_config.gene_column,
        *dataset_config.sequence_columns,
    }
    missing_columns = sorted(required_columns.difference(dataframe.columns))
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    working_frame = dataframe.copy()
    working_frame = working_frame.drop(columns=list(dataset_config.drop_columns), errors="ignore")
    working_frame = working_frame.dropna(
        subset=[dataset_config.target_column, dataset_config.gene_column]
    ).reset_index(drop=True)

    if working_frame.empty:
        raise ValueError("Dataset is empty after dropping rows with missing gene/target values.")

    numeric_columns, categorical_columns = infer_feature_columns(working_frame, dataset_config)

    numeric_frames: list[pd.DataFrame] = []
    if numeric_columns:
        numeric_frame = working_frame.loc[:, list(numeric_columns)].apply(
            pd.to_numeric, errors="coerce"
        )
        numeric_frames.append(numeric_frame.astype(np.float32))

    sequence_frames = [
        build_sequence_feature_frame(
            working_frame[column],
            column_name=column,
            max_length=dataset_config.max_sequence_length,
            allowed_bases=dataset_config.allowed_bases,
            unknown_base=dataset_config.unknown_base,
        )
        for column in dataset_config.sequence_columns
    ]
    numeric_frames.extend(sequence_frames)

    numeric_feature_frame = (
        pd.concat(numeric_frames, axis=1) if numeric_frames else pd.DataFrame(index=working_frame.index)
    )
    categorical_feature_frame = pd.DataFrame(index=working_frame.index)
    if categorical_columns:
        categorical_feature_frame = (
            working_frame.loc[:, list(categorical_columns)].fillna("__MISSING__").astype(str)
        )

    feature_frame = pd.concat([numeric_feature_frame, categorical_feature_frame], axis=1)
    if feature_frame.empty:
        raise ValueError("Feature matrix is empty. Configure numeric/categorical/sequence columns.")

    genes = working_frame[dataset_config.gene_column].map(
        lambda value: normalize_gene_label(value, dataset_config.gene_normalization)
    ).to_numpy(dtype=object)
    unique_genes = tuple(sorted(pd.unique(genes)))
    test_genes = choose_test_genes(unique_genes, dataset_config, split_config)
    test_gene_set = set(test_genes)
    train_genes = tuple(gene for gene in unique_genes if gene not in test_gene_set)
    if len(train_genes) < split_config.min_train_genes:
        raise ValueError(
            "Train split has too few genes. Adjust explicit_test_genes or test_fraction."
        )
    cv_genes = choose_cv_genes(train_genes, dataset_config, split_config)

    target = pd.to_numeric(
        working_frame[dataset_config.target_column], errors="coerce"
    ).to_numpy(dtype=np.float32)
    if np.isnan(target).any():
        raise ValueError("Target column contains non-numeric values after coercion.")

    return PreparedDataset(
        features=feature_frame,
        target=target,
        genes=genes,
        row_ids=np.arange(len(working_frame), dtype=np.int64),
        numeric_feature_columns=tuple(numeric_feature_frame.columns),
        categorical_feature_columns=tuple(categorical_feature_frame.columns),
        train_genes=train_genes,
        test_genes=test_genes,
        cv_genes=cv_genes,
        source_path=str(dataset_config.raw_data_path),
    )


def prepare_dataset(
    *,
    dataset_config: DatasetConfig = DATASET_CONFIG,
    split_config: SplitConfig = SPLIT_CONFIG,
    force: bool = False,
) -> PreparedDataset:
    ensure_runtime_dirs()
    artifact_path = CACHE_DIR / "prepared_dataset.pkl"
    metadata_path = CACHE_DIR / "prepared_dataset_metadata.json"

    source_path = dataset_config.raw_data_path
    source_mtime = source_path.stat().st_mtime if source_path.exists() else None
    fingerprint = _config_fingerprint()

    if not force and artifact_path.exists() and metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if (
            metadata.get("config_fingerprint") == fingerprint
            and metadata.get("source_mtime") == source_mtime
        ):
            try:
                with artifact_path.open("rb") as handle:
                    return pickle.load(handle)
            except (AttributeError, ModuleNotFoundError, pickle.PickleError):
                pass

    dataframe = read_raw_dataframe(source_path)
    prepared = build_prepared_dataset_from_frame(
        dataframe,
        dataset_config=dataset_config,
        split_config=split_config,
    )

    with artifact_path.open("wb") as handle:
        pickle.dump(prepared, handle)
    metadata_path.write_text(
        json.dumps(
            {
                "config_fingerprint": fingerprint,
                "source_mtime": source_mtime,
                "num_rows": len(prepared.target),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return prepared


def print_dataset_summary(prepared: PreparedDataset) -> None:
    print("---")
    print(f"dataset_path:      {prepared.source_path}")
    print(f"rows:              {len(prepared.target)}")
    print(f"features:          {prepared.features.shape[1]}")
    print(f"train_genes:       {len(prepared.train_genes)}")
    print(f"test_genes:        {len(prepared.test_genes)}")
    print(f"cv_folds:          {len(prepared.cv_genes)}")
    print(
        "primary_metric:    "
        f"{METRIC_CONFIG.primary_metric_name} ({METRIC_CONFIG.primary_metric_direction})"
    )


# ---------------------------------------------------------------------------
# Fold-local preprocessing
# ---------------------------------------------------------------------------


@dataclass
class FoldPreprocessor:
    numeric_columns: tuple[str, ...]
    categorical_columns: tuple[str, ...]

    numeric_fill_values: dict[str, float] | None = None
    numeric_means: dict[str, float] | None = None
    numeric_stds: dict[str, float] | None = None
    categorical_levels: dict[str, tuple[str, ...]] | None = None
    feature_names: tuple[str, ...] = ()

    def fit(self, frame: pd.DataFrame) -> "FoldPreprocessor":
        self.numeric_fill_values = {}
        self.numeric_means = {}
        self.numeric_stds = {}
        self.categorical_levels = {}
        feature_names: list[str] = []

        for column in self.numeric_columns:
            values = pd.to_numeric(frame[column], errors="coerce")
            median = float(values.median()) if not values.dropna().empty else 0.0
            filled = values.fillna(median)
            mean = float(filled.mean())
            std = float(filled.std(ddof=0))
            if std <= 0:
                std = 1.0
            self.numeric_fill_values[column] = median
            self.numeric_means[column] = mean
            self.numeric_stds[column] = std
            feature_names.append(column)

        for column in self.categorical_columns:
            values = frame[column].fillna("__MISSING__").astype(str)
            levels = tuple(sorted(values.unique()))
            if "__UNK__" not in levels:
                levels = levels + ("__UNK__",)
            self.categorical_levels[column] = levels
            feature_names.extend(f"{column}=={level}" for level in levels)

        self.feature_names = tuple(feature_names)
        return self

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        if (
            self.numeric_fill_values is None
            or self.numeric_means is None
            or self.numeric_stds is None
            or self.categorical_levels is None
        ):
            raise RuntimeError("Preprocessor must be fit before calling transform().")

        arrays: list[np.ndarray] = []

        for column in self.numeric_columns:
            values = pd.to_numeric(frame[column], errors="coerce")
            filled = values.fillna(self.numeric_fill_values[column]).to_numpy(dtype=np.float32)
            standardized = (filled - self.numeric_means[column]) / self.numeric_stds[column]
            arrays.append(standardized[:, None].astype(np.float32))

        for column in self.categorical_columns:
            values = frame[column].fillna("__MISSING__").astype(str)
            levels = self.categorical_levels[column]
            known_levels = set(levels)
            normalized = values.where(values.isin(known_levels), "__UNK__")
            encoded = np.stack(
                [
                    (normalized == level).to_numpy(dtype=np.float32)
                    for level in levels
                ],
                axis=1,
            )
            arrays.append(encoded.astype(np.float32))

        if not arrays:
            raise ValueError("FoldPreprocessor produced an empty feature matrix.")

        return np.concatenate(arrays, axis=1).astype(np.float32, copy=False)


@dataclass(frozen=True)
class TargetScaler:
    mean: float
    std: float

    @classmethod
    def fit(cls, target: np.ndarray) -> "TargetScaler":
        mean = float(target.mean())
        std = float(target.std())
        if std <= 0:
            std = 1.0
        return cls(mean=mean, std=std)

    def transform(self, target: np.ndarray) -> np.ndarray:
        return ((target - self.mean) / self.std).astype(np.float32)

    def inverse_transform(self, target: np.ndarray) -> np.ndarray:
        return (target * self.std + self.mean).astype(np.float32)


def build_cv_folds(prepared: PreparedDataset) -> list[FoldSpec]:
    train_gene_set = set(prepared.train_genes)
    folds: list[FoldSpec] = []
    for gene in prepared.cv_genes:
        val_mask = prepared.genes == gene
        train_mask = np.isin(prepared.genes, tuple(train_gene_set - {gene}))
        val_indices = np.flatnonzero(val_mask)
        train_indices = np.flatnonzero(train_mask)
        if val_indices.size == 0 or train_indices.size == 0:
            raise ValueError(f"Fold for gene {gene!r} has empty train or validation rows.")
        folds.append(FoldSpec(gene=gene, train_indices=train_indices, val_indices=val_indices))
    return folds


# ---------------------------------------------------------------------------
# Architecture validation and loading
# ---------------------------------------------------------------------------


def validate_architecture_spec(
    spec: ArchitectureSpec,
    constraints: ArchitectureConstraints = ARCHITECTURE_CONSTRAINTS,
) -> None:
    if spec.family not in constraints.allowed_families:
        raise ValueError(f"Unsupported architecture family: {spec.family!r}")
    if spec.activation not in constraints.allowed_activations:
        raise ValueError(f"Unsupported activation: {spec.activation!r}")
    if spec.normalization not in constraints.allowed_normalizations:
        raise ValueError(f"Unsupported normalization: {spec.normalization!r}")
    if not constraints.allow_bias and spec.use_bias:
        raise ValueError("Bias parameters are disabled by ArchitectureConstraints.")
    if len(spec.hidden_dims) > constraints.max_hidden_layers:
        raise ValueError("Too many hidden layers in ArchitectureSpec.hidden_dims.")
    if any(width <= 0 or width > constraints.max_hidden_width for width in spec.hidden_dims):
        raise ValueError("ArchitectureSpec.hidden_dims contains an invalid layer width.")
    if spec.dropout < 0 or spec.dropout > constraints.max_dropout:
        raise ValueError(
            f"Dropout must be between 0 and {constraints.max_dropout}, got {spec.dropout}."
        )


def validate_train_source(path: Path) -> None:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    has_architecture = False
    has_build_model = False

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name not in ALLOWED_TRAIN_IMPORTS:
                    raise ValueError(f"Import not allowed in train.py: {alias.name!r}")
        elif isinstance(node, ast.ImportFrom):
            module_name = node.module or ""
            if module_name not in ALLOWED_TRAIN_IMPORTS:
                raise ValueError(f"Import not allowed in train.py: {module_name!r}")
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "ARCHITECTURE":
                    has_architecture = True
        elif isinstance(node, ast.FunctionDef) and node.name == "build_model":
            has_build_model = True
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in FORBIDDEN_CALL_NAMES:
                raise ValueError(f"Forbidden call in train.py: {node.func.id}()")
            if isinstance(node.func, ast.Attribute) and node.func.attr in FORBIDDEN_ATTRIBUTE_NAMES:
                raise ValueError(f"Forbidden method call in train.py: .{node.func.attr}()")

    if not has_architecture:
        raise ValueError("train.py must define ARCHITECTURE.")
    if not has_build_model:
        raise ValueError("train.py must define build_model(context).")


def load_train_definition(path: Path = EDITABLE_TRAIN_FILE) -> LoadedArchitecture:
    validate_train_source(path)
    spec = importlib.util.spec_from_file_location("architecture_module", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module spec from {path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return _extract_loaded_architecture(module)


def _extract_loaded_architecture(module: ModuleType) -> LoadedArchitecture:
    architecture = getattr(module, "ARCHITECTURE", None)
    build_model = getattr(module, "build_model", None)

    if not isinstance(architecture, ArchitectureSpec):
        raise TypeError("train.py must define ARCHITECTURE as an ArchitectureSpec instance.")
    if not callable(build_model):
        raise TypeError("train.py must define a callable build_model(context).")

    validate_architecture_spec(architecture)
    return LoadedArchitecture(
        spec=architecture,
        build_model=build_model,
        module_name=module.__name__,
    )


# ---------------------------------------------------------------------------
# Training harness
# ---------------------------------------------------------------------------


def instantiate_model(
    architecture: ArchitectureSpec,
    build_model: ModelBuilder,
    context: ArchitectureContext,
    constraints: ArchitectureConstraints = ARCHITECTURE_CONSTRAINTS,
) -> tuple[nn.Module, int]:
    validate_architecture_spec(architecture, constraints)
    model = build_model(context)
    if not isinstance(model, nn.Module):
        raise TypeError("build_model(context) must return a torch.nn.Module.")

    with torch.no_grad():
        dummy = torch.zeros(2, context.input_dim, dtype=torch.float32)
        output = _normalize_model_output(model(dummy))
    if output.shape[0] != 2:
        raise ValueError("Model output does not preserve batch size.")

    num_params = count_parameters(model)
    if num_params > constraints.max_parameters:
        raise ValueError(
            f"Model has {num_params:,} parameters, above the limit of "
            f"{constraints.max_parameters:,}."
        )

    return model.to(context.device), num_params


def create_dataloader(
    features: np.ndarray,
    target: np.ndarray,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
    pin_memory: bool,
) -> DataLoader:
    feature_tensor = torch.from_numpy(features.astype(np.float32))
    target_tensor = torch.from_numpy(target.astype(np.float32)).unsqueeze(1)
    generator = torch.Generator()
    generator.manual_seed(seed)
    dataset = TensorDataset(feature_tensor, target_tensor)
    effective_batch_size = min(batch_size, len(dataset))
    return DataLoader(
        dataset,
        batch_size=effective_batch_size,
        shuffle=shuffle,
        generator=generator,
        pin_memory=pin_memory,
    )


def predict_regression(model: nn.Module, features: np.ndarray, device: str) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        tensor = torch.from_numpy(features.astype(np.float32)).to(device)
        prediction = _normalize_model_output(model(tensor))
    return prediction.detach().cpu().numpy().astype(np.float32)


def _train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    *,
    optimizer: torch.optim.Optimizer,
    device: str,
    grad_clip_norm: float | None,
) -> None:
    model.train()
    for batch_features, batch_target in dataloader:
        batch_features = batch_features.to(device, non_blocking=True)
        batch_target = batch_target.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        prediction = _normalize_model_output(model(batch_features))
        loss = F.mse_loss(prediction, batch_target[:, 0])
        loss.backward()
        if grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizer.step()


def train_fold(
    prepared: PreparedDataset,
    fold: FoldSpec,
    architecture: ArchitectureSpec,
    build_model: ModelBuilder,
    *,
    training_config: TrainingConfig = TRAINING_CONFIG,
    constraints: ArchitectureConstraints = ARCHITECTURE_CONSTRAINTS,
    seed: int,
    budget_seconds: float,
) -> FoldResult:
    train_frame = prepared.features.iloc[fold.train_indices]
    val_frame = prepared.features.iloc[fold.val_indices]
    train_target = prepared.target[fold.train_indices]
    val_target = prepared.target[fold.val_indices]

    preprocessor = FoldPreprocessor(
        numeric_columns=prepared.numeric_feature_columns,
        categorical_columns=prepared.categorical_feature_columns,
    ).fit(train_frame)
    train_features = preprocessor.transform(train_frame)
    val_features = preprocessor.transform(val_frame)

    target_scaler = TargetScaler.fit(train_target)
    train_target_scaled = target_scaler.transform(train_target)

    context = ArchitectureContext(
        input_dim=train_features.shape[1],
        output_dim=1,
        train_size=len(train_features),
        feature_names=preprocessor.feature_names,
        device=training_config.device,
    )
    set_random_seed(seed)
    model, num_params = instantiate_model(architecture, build_model, context, constraints)

    pin_memory = training_config.device.startswith("cuda")
    dataloader = create_dataloader(
        train_features,
        train_target_scaled,
        batch_size=training_config.batch_size,
        shuffle=True,
        seed=seed,
        pin_memory=pin_memory,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )

    best_metrics: RegressionMetrics | None = None
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    deadline = time.perf_counter() + budget_seconds
    epoch = 0
    start = time.perf_counter()

    while True:
        epoch += 1
        _train_epoch(
            model,
            dataloader,
            optimizer=optimizer,
            device=training_config.device,
            grad_clip_norm=training_config.grad_clip_norm,
        )
        val_prediction = target_scaler.inverse_transform(
            predict_regression(model, val_features, training_config.device)
        )
        metrics = evaluate_predictions(val_target, val_prediction)
        if best_metrics is None or metrics.rmse < best_metrics.rmse:
            best_metrics = metrics
            best_epoch = epoch
            best_state = _state_dict_to_cpu(model)
        if time.perf_counter() >= deadline:
            break

    if best_state is None or best_metrics is None:
        raise RuntimeError("Fold training did not produce any validation metrics.")

    model.load_state_dict(best_state)
    train_seconds = time.perf_counter() - start
    return FoldResult(
        gene=fold.gene,
        count=len(val_target),
        metrics=best_metrics,
        train_seconds=train_seconds,
        epochs=epoch,
        best_epoch=best_epoch,
        num_params=num_params,
    )


def train_final_holdout(
    prepared: PreparedDataset,
    architecture: ArchitectureSpec,
    build_model: ModelBuilder,
    *,
    training_config: TrainingConfig = TRAINING_CONFIG,
    constraints: ArchitectureConstraints = ARCHITECTURE_CONSTRAINTS,
    seed: int,
    budget_seconds: float,
) -> RegressionMetrics | None:
    if not prepared.test_genes:
        return None

    train_mask = np.isin(prepared.genes, prepared.train_genes)
    test_mask = np.isin(prepared.genes, prepared.test_genes)

    train_frame = prepared.features.loc[train_mask]
    test_frame = prepared.features.loc[test_mask]
    train_target = prepared.target[train_mask]
    test_target = prepared.target[test_mask]

    preprocessor = FoldPreprocessor(
        numeric_columns=prepared.numeric_feature_columns,
        categorical_columns=prepared.categorical_feature_columns,
    ).fit(train_frame)
    train_features = preprocessor.transform(train_frame)
    test_features = preprocessor.transform(test_frame)

    target_scaler = TargetScaler.fit(train_target)
    train_target_scaled = target_scaler.transform(train_target)

    context = ArchitectureContext(
        input_dim=train_features.shape[1],
        output_dim=1,
        train_size=len(train_features),
        feature_names=preprocessor.feature_names,
        device=training_config.device,
    )
    set_random_seed(seed)
    model, _ = instantiate_model(architecture, build_model, context, constraints)

    dataloader = create_dataloader(
        train_features,
        train_target_scaled,
        batch_size=training_config.batch_size,
        shuffle=True,
        seed=seed,
        pin_memory=training_config.device.startswith("cuda"),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )

    deadline = time.perf_counter() + budget_seconds
    while True:
        _train_epoch(
            model,
            dataloader,
            optimizer=optimizer,
            device=training_config.device,
            grad_clip_norm=training_config.grad_clip_norm,
        )
        if time.perf_counter() >= deadline:
            break

    prediction = target_scaler.inverse_transform(
        predict_regression(model, test_features, training_config.device)
    )
    return evaluate_predictions(test_target, prediction)


def aggregate_fold_results(
    fold_results: list[FoldResult],
    *,
    metric_config: MetricConfig = METRIC_CONFIG,
) -> dict[str, float | None]:
    weights = np.array([result.count for result in fold_results], dtype=np.float64)
    rmse_values = np.array([result.metrics.rmse for result in fold_results], dtype=np.float64)
    mae_values = np.array([result.metrics.mae for result in fold_results], dtype=np.float64)
    squared_error_sum = sum(result.metrics.squared_error_sum for result in fold_results)
    total_examples = int(weights.sum())

    valid_r2_values = [
        (result.metrics.r2, result.count)
        for result in fold_results
        if not math.isnan(result.metrics.r2)
    ]
    weighted_r2 = None
    if valid_r2_values:
        r2_weights = np.array([weight for _, weight in valid_r2_values], dtype=np.float64)
        weighted_r2 = float(
            np.average(
                np.array([value for value, _ in valid_r2_values], dtype=np.float64),
                weights=r2_weights,
            )
        )

    weighted_rmse_mean = float(np.average(rmse_values, weights=weights))
    return {
        "primary_metric_value": weighted_rmse_mean,
        "weighted_cv_rmse_mean": weighted_rmse_mean,
        "cv_rmse_mean": float(np.mean(rmse_values)),
        "cv_rmse_std": float(np.std(rmse_values)),
        "weighted_cv_mae_mean": float(np.average(mae_values, weights=weights)),
        "weighted_cv_r2_mean": weighted_r2,
        "pooled_cv_rmse": float(math.sqrt(squared_error_sum / total_examples)),
        "primary_metric_name": metric_config.primary_metric_name,
    }


def validate_budget(
    num_folds: int,
    *,
    training_config: TrainingConfig = TRAINING_CONFIG,
    has_holdout_test: bool = False,
) -> tuple[float, float]:
    if num_folds <= 0:
        raise ValueError("At least one CV fold is required.")
    if not 0 < training_config.cv_budget_ratio <= 1:
        raise ValueError("TrainingConfig.cv_budget_ratio must be in (0, 1].")

    if training_config.evaluate_test_split and has_holdout_test:
        cv_budget = training_config.total_time_budget_seconds * training_config.cv_budget_ratio
    else:
        cv_budget = training_config.total_time_budget_seconds
    fold_budget = cv_budget / num_folds
    if fold_budget < training_config.min_fold_budget_seconds:
        raise ValueError(
            "Per-fold budget is too small for the configured gene CV. "
            "Increase total_time_budget_seconds, reduce cv folds, or lower "
            "TrainingConfig.min_fold_budget_seconds."
        )

    final_budget = training_config.total_time_budget_seconds - cv_budget
    if (
        training_config.evaluate_test_split
        and final_budget > 0
        and final_budget < training_config.min_final_fit_budget_seconds
    ):
        raise ValueError(
            "Final holdout budget is too small. Increase total_time_budget_seconds or "
            "adjust TrainingConfig.cv_budget_ratio."
        )
    return fold_budget, final_budget


# ---------------------------------------------------------------------------
# Experiment orchestration
# ---------------------------------------------------------------------------


def save_run_summary(
    summary: ExperimentSummary,
    fold_results: list[FoldResult],
    architecture: ArchitectureSpec,
    *,
    run_dir: Path,
) -> None:
    payload = {
        "summary": asdict(summary),
        "architecture": asdict(architecture),
        "fold_results": [
            {
                "gene": result.gene,
                "count": result.count,
                "train_seconds": result.train_seconds,
                "epochs": result.epochs,
                "best_epoch": result.best_epoch,
                "num_params": result.num_params,
                "metrics": asdict(result.metrics),
            }
            for result in fold_results
        ],
        "constraints": asdict(ARCHITECTURE_CONSTRAINTS),
        "training_config": asdict(TRAINING_CONFIG),
        "dataset_config": asdict(DATASET_CONFIG),
        "split_config": asdict(SPLIT_CONFIG),
        "metric_config": asdict(METRIC_CONFIG),
    }
    run_dir.joinpath("summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_json_default),
        encoding="utf-8",
    )
    RUNS_DIR.joinpath("latest_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_json_default),
        encoding="utf-8",
    )


def print_experiment_summary(summary: ExperimentSummary) -> None:
    print("---")
    print(f"primary_metric_name: {summary.primary_metric_name}")
    print(f"primary_metric:      {summary.primary_metric_value:.6f}")
    print(f"weighted_cv_rmse:    {summary.weighted_cv_rmse_mean:.6f}")
    print(f"cv_rmse_mean:        {summary.cv_rmse_mean:.6f}")
    print(f"cv_rmse_std:         {summary.cv_rmse_std:.6f}")
    print(f"weighted_cv_mae:     {summary.weighted_cv_mae_mean:.6f}")
    if summary.weighted_cv_r2_mean is None:
        print("weighted_cv_r2:      nan")
    else:
        print(f"weighted_cv_r2:      {summary.weighted_cv_r2_mean:.6f}")
    print(f"pooled_cv_rmse:      {summary.pooled_cv_rmse:.6f}")
    if summary.test_rmse is None:
        print("test_rmse:           nan")
        print("test_mae:            nan")
        print("test_r2:             nan")
    else:
        print(f"test_rmse:           {summary.test_rmse:.6f}")
        print(f"test_mae:            {summary.test_mae:.6f}")
        if summary.test_r2 is None:
            print("test_r2:             nan")
        else:
            print(f"test_r2:             {summary.test_r2:.6f}")
    print(f"train_seconds:       {summary.train_seconds:.1f}")
    print(f"num_params:          {summary.num_params}")
    print(f"feature_dim:         {summary.feature_dim}")
    print(f"num_rows:            {summary.num_rows}")
    print(f"cv_folds:            {summary.cv_folds}")
    print(f"train_genes:         {','.join(summary.train_genes)}")
    print(f"test_genes:          {','.join(summary.test_genes) if summary.test_genes else '-'}")
    print(f"run_dir:             {summary.run_dir}")


def run_experiment(
    architecture: ArchitectureSpec | None = None,
    build_model: ModelBuilder | None = None,
    *,
    dataset_config: DatasetConfig = DATASET_CONFIG,
    split_config: SplitConfig = SPLIT_CONFIG,
    training_config: TrainingConfig = TRAINING_CONFIG,
    metric_config: MetricConfig = METRIC_CONFIG,
    constraints: ArchitectureConstraints = ARCHITECTURE_CONSTRAINTS,
) -> ExperimentSummary:
    ensure_runtime_dirs()

    if architecture is None or build_model is None:
        loaded = load_train_definition()
        architecture = loaded.spec
        build_model = loaded.build_model

    validate_architecture_spec(architecture, constraints)
    prepared = prepare_dataset(dataset_config=dataset_config, split_config=split_config)
    folds = build_cv_folds(prepared)
    fold_budget, final_budget = validate_budget(
        len(folds),
        training_config=training_config,
        has_holdout_test=bool(prepared.test_genes),
    )

    start = time.perf_counter()
    fold_results: list[FoldResult] = []
    num_params: int | None = None

    for fold_index, fold in enumerate(folds):
        result = train_fold(
            prepared,
            fold,
            architecture,
            build_model,
            training_config=training_config,
            constraints=constraints,
            seed=split_config.random_seed + fold_index,
            budget_seconds=fold_budget,
        )
        fold_results.append(result)
        if num_params is None:
            num_params = result.num_params

    if num_params is None:
        raise RuntimeError("Experiment produced no fold results.")

    aggregate = aggregate_fold_results(fold_results, metric_config=metric_config)
    holdout_metrics = None
    if training_config.evaluate_test_split and prepared.test_genes and final_budget > 0:
        holdout_metrics = train_final_holdout(
            prepared,
            architecture,
            build_model,
            training_config=training_config,
            constraints=constraints,
            seed=split_config.random_seed + len(folds),
            budget_seconds=final_budget,
        )

    run_dir = _make_run_dir()
    summary = ExperimentSummary(
        primary_metric_name=str(aggregate["primary_metric_name"]),
        primary_metric_value=float(aggregate["primary_metric_value"]),
        metric_direction=metric_config.primary_metric_direction,
        improvement_epsilon=metric_config.improvement_epsilon,
        weighted_cv_rmse_mean=float(aggregate["weighted_cv_rmse_mean"]),
        cv_rmse_mean=float(aggregate["cv_rmse_mean"]),
        cv_rmse_std=float(aggregate["cv_rmse_std"]),
        weighted_cv_mae_mean=float(aggregate["weighted_cv_mae_mean"]),
        weighted_cv_r2_mean=(
            None if aggregate["weighted_cv_r2_mean"] is None else float(aggregate["weighted_cv_r2_mean"])
        ),
        pooled_cv_rmse=float(aggregate["pooled_cv_rmse"]),
        test_rmse=None if holdout_metrics is None else holdout_metrics.rmse,
        test_mae=None if holdout_metrics is None else holdout_metrics.mae,
        test_r2=None if holdout_metrics is None or math.isnan(holdout_metrics.r2) else holdout_metrics.r2,
        num_params=num_params,
        train_seconds=time.perf_counter() - start,
        feature_dim=prepared.features.shape[1],
        num_rows=len(prepared.target),
        cv_folds=len(folds),
        train_genes=prepared.train_genes,
        test_genes=prepared.test_genes,
        cv_genes=prepared.cv_genes,
        run_dir=str(run_dir),
    )
    save_run_summary(summary, fold_results, architecture, run_dir=run_dir)
    print_experiment_summary(summary)
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare the fixed siRNA regression dataset and experiment harness."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild the cached prepared dataset even if the cache looks current.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prepared = prepare_dataset(force=args.force)
    print_dataset_summary(prepared)


if __name__ == "__main__":
    main()
