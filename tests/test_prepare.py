from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from prepare import (
    ArchitectureSpec,
    DatasetConfig,
    FoldResult,
    RegressionMetrics,
    SplitConfig,
    aggregate_fold_results,
    build_cv_folds,
    build_prepared_dataset_from_frame,
    load_train_definition,
    validate_architecture_spec,
    validate_train_source,
)


def make_dataset() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "gene": ["GENE1", "GENE1", "GENE2", "GENE2", "GENE3", "GENE3"],
            "target": [0.2, 0.3, 0.5, 0.6, 0.1, 0.15],
            "sirna_sequence": [
                "AUGCUA",
                "AUGCAA",
                "CCGAUU",
                "CCGAUC",
                "UUUGGA",
                "UUUGGC",
            ],
            "feature_score": [1.0, 1.5, 2.0, 2.5, 0.4, 0.6],
            "cell_line": ["A", "A", "B", "B", "A", "C"],
        }
    )


def test_build_prepared_dataset_respects_gene_splits() -> None:
    prepared = build_prepared_dataset_from_frame(
        make_dataset(),
        dataset_config=DatasetConfig(
            raw_data_path=Path("data/mock.csv"),
            target_column="target",
            gene_column="gene",
            sequence_columns=(),
            numeric_columns=(),
            categorical_columns=(),
            drop_columns=("sirna_sequence",),
            explicit_test_genes=("GENE3",),
            explicit_cv_genes=("GENE1", "GENE2"),
            max_sequence_length=6,
        ),
        split_config=SplitConfig(random_seed=7),
    )

    assert prepared.train_genes == ("GENE1", "GENE2")
    assert prepared.test_genes == ("GENE3",)
    assert prepared.cv_genes == ("GENE1", "GENE2")
    assert "feature_score" in prepared.numeric_feature_columns
    assert prepared.categorical_feature_columns == ("cell_line",)


def test_build_cv_folds_holds_out_one_gene_per_fold() -> None:
    prepared = build_prepared_dataset_from_frame(
        make_dataset(),
        dataset_config=DatasetConfig(
            raw_data_path=Path("data/mock.csv"),
            target_column="target",
            gene_column="gene",
            sequence_columns=(),
            drop_columns=("sirna_sequence",),
            explicit_test_genes=("GENE3",),
            explicit_cv_genes=("GENE1", "GENE2"),
            max_sequence_length=6,
        ),
    )

    folds = build_cv_folds(prepared)

    assert [fold.gene for fold in folds] == ["GENE1", "GENE2"]

    for fold in folds:
        val_genes = set(prepared.genes[fold.val_indices])
        train_genes = set(prepared.genes[fold.train_indices])
        assert val_genes == {fold.gene}
        assert fold.gene not in train_genes
        assert "GENE3" not in train_genes


def test_gene_normalization_collapses_case_variants() -> None:
    dataframe = pd.DataFrame(
        {
            "gene": ["CPN1", "Cpn1", "OTHER", "OTHER"],
            "target": [0.1, 0.2, 0.3, 0.4],
            "feature_score": [1.0, 1.1, 2.0, 2.1],
        }
    )

    prepared = build_prepared_dataset_from_frame(
        dataframe,
        dataset_config=DatasetConfig(
            raw_data_path=Path("data/mock.csv"),
            target_column="target",
            gene_column="gene",
            sequence_columns=(),
            explicit_test_genes=(),
            explicit_cv_genes=(),
            test_fraction=0.0,
            gene_normalization="upper",
        ),
    )

    assert tuple(prepared.genes) == ("CPN1", "CPN1", "OTHER", "OTHER")
    assert prepared.train_genes == ("CPN1", "OTHER")
    assert prepared.cv_genes == ("CPN1", "OTHER")


def test_weighted_cv_rmse_mean_uses_fold_sizes() -> None:
    fold_results = [
        FoldResult(
            gene="GENE1",
            count=2,
            metrics=RegressionMetrics(rmse=1.0, mae=0.8, r2=0.2, squared_error_sum=2.0),
            train_seconds=1.0,
            epochs=2,
            best_epoch=2,
            num_params=100,
        ),
        FoldResult(
            gene="GENE2",
            count=6,
            metrics=RegressionMetrics(rmse=0.5, mae=0.4, r2=0.4, squared_error_sum=1.5),
            train_seconds=1.0,
            epochs=2,
            best_epoch=1,
            num_params=100,
        ),
    ]

    aggregate = aggregate_fold_results(fold_results)
    expected_weighted_rmse = (2 * 1.0 + 6 * 0.5) / 8

    assert aggregate["weighted_cv_rmse_mean"] == expected_weighted_rmse
    assert aggregate["cv_rmse_mean"] == 0.75
    assert aggregate["pooled_cv_rmse"] == np.sqrt((2.0 + 1.5) / 8)


def test_validate_train_source_rejects_forbidden_calls(tmp_path: Path) -> None:
    train_file = tmp_path / "train.py"
    train_file.write_text(
        "\n".join(
            [
                "from prepare import ArchitectureContext, ArchitectureSpec",
                "ARCHITECTURE = ArchitectureSpec(family='mlp', hidden_dims=(16,))",
                "def build_model(context: ArchitectureContext):",
                "    open('forbidden.txt', 'w')",
                "    return None",
            ]
        ),
        encoding="utf-8",
    )

    try:
        validate_train_source(train_file)
    except ValueError as exc:
        assert "Forbidden call" in str(exc)
    else:
        raise AssertionError("validate_train_source() did not reject forbidden open().")


def test_actual_train_module_matches_contract() -> None:
    loaded = load_train_definition()

    assert isinstance(loaded.spec, ArchitectureSpec)
    validate_architecture_spec(loaded.spec)
