from __future__ import annotations

import math
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from autoresirch.prepare.comparative.dataset import build_comparative_prepared_dataset_from_frame
from autoresirch.prepare.data.rnafm import build_rnafm_embedding_tensor
from autoresirch.prepare.shared.schemas import (
    CACHE_DIR,
    DATASET_CONFIG,
    METRIC_CONFIG,
    SPLIT_CONFIG,
    DatasetConfig,
    PreparedDataset,
    SplitConfig,
)
from autoresirch.prepare.shared.utils import (
    _config_fingerprint,
    ensure_runtime_dirs,
    resolve_primary_metric_name,
)

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
    include_rnafm_embeddings: bool = False,
) -> PreparedDataset:
    if dataset_config.experiment_mode == "comparative":
        if include_rnafm_embeddings:
            raise NotImplementedError(
                "Comparative mode does not support RNA-FM sequence tensors yet."
            )
        return build_comparative_prepared_dataset_from_frame(
            dataframe,
            dataset_config=dataset_config,
            split_config=split_config,
        )

    required_columns = {
        dataset_config.target_column,
        dataset_config.gene_column,
        *dataset_config.sequence_columns,
    }
    if include_rnafm_embeddings:
        required_columns.add(dataset_config.rnafm_sequence_column)
    missing_columns = sorted(required_columns.difference(dataframe.columns))
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    working_frame = dataframe.copy()
    drop_columns = list(dataset_config.drop_columns)
    if include_rnafm_embeddings and dataset_config.rnafm_sequence_column in drop_columns:
        drop_columns = [
            column for column in drop_columns if column != dataset_config.rnafm_sequence_column
        ]
    working_frame = working_frame.drop(columns=drop_columns, errors="ignore")
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
    sequence_tensor: np.ndarray | None = None
    sequence_feature_name: str | None = None
    sequence_feature_shape: tuple[int, ...] | None = None
    if include_rnafm_embeddings:
        sequence_tensor = build_rnafm_embedding_tensor(
            working_frame[dataset_config.rnafm_sequence_column],
            max_length=dataset_config.max_sequence_length,
            embedding_dim=dataset_config.rnafm_embedding_dim,
            allowed_bases=dataset_config.allowed_bases,
            unknown_base=dataset_config.unknown_base,
        )
        sequence_feature_name = f"rnafm::{dataset_config.rnafm_sequence_column}"
        sequence_feature_shape = tuple(sequence_tensor.shape[1:])

    if feature_frame.empty and sequence_tensor is None:
        raise ValueError(
            "Feature matrix is empty. Configure numeric/categorical/sequence columns or enable RNA-FM."
        )

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
        flat_features=feature_frame if not feature_frame.empty else None,
        sequence_features=sequence_tensor,
        target=target,
        genes=genes,
        row_ids=np.arange(len(working_frame), dtype=np.int64),
        numeric_feature_columns=tuple(numeric_feature_frame.columns),
        categorical_feature_columns=tuple(categorical_feature_frame.columns),
        train_genes=train_genes,
        test_genes=test_genes,
        cv_genes=cv_genes,
        source_path=str(dataset_config.raw_data_path),
        sequence_feature_name=sequence_feature_name,
        sequence_feature_shape=sequence_feature_shape,
        experiment_mode="standard",
    )


def prepare_dataset(
    *,
    dataset_config: DatasetConfig = DATASET_CONFIG,
    split_config: SplitConfig = SPLIT_CONFIG,
    force: bool = False,
    include_rnafm_embeddings: bool = False,
) -> PreparedDataset:
    ensure_runtime_dirs()
    artifact_suffix = f"_{dataset_config.experiment_mode}"
    if include_rnafm_embeddings:
        artifact_suffix = f"{artifact_suffix}_rnafm"
    artifact_path = CACHE_DIR / f"prepared_dataset{artifact_suffix}.pkl"
    metadata_path = CACHE_DIR / f"prepared_dataset_metadata{artifact_suffix}.json"

    source_path = dataset_config.raw_data_path
    source_mtime = source_path.stat().st_mtime if source_path.exists() else None
    fingerprint = json.dumps(
        {
            "base_fingerprint": _config_fingerprint(),
            "include_rnafm_embeddings": include_rnafm_embeddings,
        },
        sort_keys=True,
    )

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
        include_rnafm_embeddings=include_rnafm_embeddings,
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
    print(f"experiment_mode:   {prepared.experiment_mode}")
    print(f"dataset_path:      {prepared.source_path}")
    print(f"rows:              {len(prepared.target)}")
    print(f"flat_features:     {prepared.features.shape[1]}")
    if prepared.has_sequence_features and prepared.sequence_feature_shape is not None:
        print(f"sequence_features: {prepared.sequence_feature_name} {prepared.sequence_feature_shape}")
    else:
        print("sequence_features: -")
    print(f"train_genes:       {len(prepared.train_genes)}")
    print(f"test_genes:        {len(prepared.test_genes)}")
    print(f"cv_folds:          {len(prepared.cv_genes)}")
    print(
        "primary_metric:    "
        f"{resolve_primary_metric_name(prepared.experiment_mode)} "
        f"({METRIC_CONFIG.primary_metric_direction})"
    )
