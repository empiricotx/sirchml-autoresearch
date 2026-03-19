from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd

from autoresirch.prepare.comparative.metrics import comparative_class_labels
from autoresirch.prepare.schemas import (
    DATASET_CONFIG,
    METRIC_CONFIG,
    SPLIT_CONFIG,
    DatasetConfig,
    MetricConfig,
    PreparedDataset,
    SplitConfig,
)


def _validate_gene_pair_support(prepared: PreparedDataset) -> None:
    genes_to_validate = (*prepared.train_genes, *prepared.test_genes)
    for gene in genes_to_validate:
        gene_count = int(np.sum(prepared.genes == gene))
        if gene_count < 2:
            raise ValueError(
                f"Comparative mode requires at least two sequences for gene {gene!r}; "
                f"found {gene_count}."
            )


def build_comparative_prepared_dataset(
    sequence_prepared: PreparedDataset,
    *,
    metric_config: MetricConfig = METRIC_CONFIG,
) -> PreparedDataset:
    if sequence_prepared.experiment_mode != "standard":
        raise ValueError("Comparative dataset construction expects standard sequence-level input.")

    _validate_gene_pair_support(sequence_prepared)
    if not sequence_prepared.has_flat_features:
        raise ValueError("Comparative mode currently requires flat numeric features.")

    numeric_feature_columns = sequence_prepared.numeric_feature_columns
    if not numeric_feature_columns:
        raise ValueError(
            "Comparative mode requires at least one numeric or engineered sequence feature."
        )

    numeric_frame = sequence_prepared.features.loc[:, list(numeric_feature_columns)].apply(
        pd.to_numeric,
        errors="coerce",
    )
    if numeric_frame.isna().all(axis=None):
        raise ValueError("Comparative numeric feature frame is entirely missing after coercion.")

    pair_feature_rows: list[np.ndarray] = []
    pair_genes: list[str] = []
    target_delta_rows: list[float] = []
    left_row_ids: list[int] = []
    right_row_ids: list[int] = []

    for gene in sequence_prepared.train_genes + sequence_prepared.test_genes:
        gene_indices = np.flatnonzero(sequence_prepared.genes == gene)
        for left_index, right_index in combinations(gene_indices.tolist(), 2):
            left_features = numeric_frame.iloc[left_index].to_numpy(dtype=np.float32, copy=True)
            right_features = numeric_frame.iloc[right_index].to_numpy(dtype=np.float32, copy=True)
            pair_feature_rows.append(left_features - right_features)
            pair_genes.append(gene)
            target_delta_rows.append(
                float(sequence_prepared.target[left_index] - sequence_prepared.target[right_index])
            )
            left_row_ids.append(int(sequence_prepared.row_ids[left_index]))
            right_row_ids.append(int(sequence_prepared.row_ids[right_index]))

    if not pair_feature_rows:
        raise ValueError("Comparative mode produced zero within-gene comparison rows.")

    pair_feature_frame = pd.DataFrame(
        np.vstack(pair_feature_rows).astype(np.float32, copy=False),
        columns=[f"delta::{column}" for column in numeric_feature_columns],
    )
    target_delta = np.array(target_delta_rows, dtype=np.float32)
    return PreparedDataset(
        flat_features=pair_feature_frame,
        sequence_features=None,
        target=target_delta,
        genes=np.array(pair_genes, dtype=object),
        row_ids=np.arange(len(pair_feature_rows), dtype=np.int64),
        numeric_feature_columns=tuple(pair_feature_frame.columns),
        categorical_feature_columns=(),
        train_genes=sequence_prepared.train_genes,
        test_genes=sequence_prepared.test_genes,
        cv_genes=sequence_prepared.cv_genes,
        source_path=sequence_prepared.source_path,
        sequence_feature_name=None,
        sequence_feature_shape=None,
        experiment_mode="comparative",
        target_class=comparative_class_labels(target_delta, metric_config),
        left_row_ids=np.array(left_row_ids, dtype=np.int64),
        right_row_ids=np.array(right_row_ids, dtype=np.int64),
    )


def build_comparative_prepared_dataset_from_frame(
    dataframe: pd.DataFrame,
    *,
    dataset_config: DatasetConfig = DATASET_CONFIG,
    split_config: SplitConfig = SPLIT_CONFIG,
    metric_config: MetricConfig = METRIC_CONFIG,
) -> PreparedDataset:
    from autoresirch.prepare.dataset_preparation import build_prepared_dataset_from_frame

    sequence_dataset_config = DatasetConfig(
        raw_data_path=dataset_config.raw_data_path,
        target_column=dataset_config.target_column,
        gene_column=dataset_config.gene_column,
        gene_normalization=dataset_config.gene_normalization,
        sequence_columns=dataset_config.sequence_columns,
        numeric_columns=dataset_config.numeric_columns,
        categorical_columns=dataset_config.categorical_columns,
        drop_columns=dataset_config.drop_columns,
        explicit_test_genes=dataset_config.explicit_test_genes,
        explicit_cv_genes=dataset_config.explicit_cv_genes,
        test_fraction=dataset_config.test_fraction,
        max_sequence_length=dataset_config.max_sequence_length,
        allowed_bases=dataset_config.allowed_bases,
        unknown_base=dataset_config.unknown_base,
        rnafm_sequence_column=dataset_config.rnafm_sequence_column,
        rnafm_embedding_dim=dataset_config.rnafm_embedding_dim,
        experiment_mode="standard",
    )
    sequence_prepared = build_prepared_dataset_from_frame(
        dataframe,
        dataset_config=sequence_dataset_config,
        split_config=split_config,
        include_rnafm_embeddings=False,
    )
    return build_comparative_prepared_dataset(sequence_prepared, metric_config=metric_config)
