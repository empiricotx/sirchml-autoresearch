from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from prepare import (
    ArchitectureSpec,
    DatasetConfig,
    FoldResult,
    MetricConfig,
    RegressionMetrics,
    evaluate_predictions,
    SplitConfig,
    aggregate_fold_results,
    build_cv_folds,
    build_prepared_dataset_from_frame,
    load_train_definition,
    scale_regression_predictions,
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
            metrics=RegressionMetrics(
                rmse=1.0,
                mae=0.8,
                r2=0.2,
                squared_error_sum=2.0,
                auc=0.9,
                pearson_r=0.1,
                spearman_r=0.3,
            ),
            train_seconds=1.0,
            epochs=2,
            best_epoch=2,
            num_params=100,
        ),
        FoldResult(
            gene="GENE2",
            count=6,
            metrics=RegressionMetrics(
                rmse=0.5,
                mae=0.4,
                r2=0.4,
                squared_error_sum=1.5,
                auc=0.6,
                pearson_r=0.7,
                spearman_r=0.9,
            ),
            train_seconds=1.0,
            epochs=2,
            best_epoch=1,
            num_params=100,
        ),
    ]

    aggregate = aggregate_fold_results(fold_results)
    expected_weighted_rmse = (2 * 1.0 + 6 * 0.5) / 8
    expected_weighted_auc = (2 * 0.9 + 6 * 0.6) / 8

    assert aggregate["primary_metric_name"] == "weighted_cv_auc"
    assert aggregate["primary_metric_value"] == expected_weighted_auc
    assert aggregate["weighted_cv_rmse_mean"] == expected_weighted_rmse
    assert aggregate["cv_rmse_mean"] == 0.75
    assert aggregate["pooled_cv_rmse"] == np.sqrt((2.0 + 1.5) / 8)
    assert aggregate["weighted_cv_auc_mean"] == expected_weighted_auc
    assert aggregate["weighted_cv_pearson_r_mean"] == (2 * 0.1 + 6 * 0.7) / 8
    assert aggregate["weighted_cv_spearman_r_mean"] == (2 * 0.3 + 6 * 0.9) / 8


def test_prediction_scaling_and_auc_follow_configured_rule() -> None:
    metric_config = MetricConfig(
        prediction_scale_min=0.45,
        prediction_scale_max=0.9,
        effective_threshold=0.4,
    )
    y_true = np.array([0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0], dtype=np.float32)
    raw_predictions = np.array([0.45, 0.60, 0.75, 0.90], dtype=np.float32)

    scaled_predictions = scale_regression_predictions(raw_predictions, metric_config)
    metrics = evaluate_predictions(y_true, raw_predictions, metric_config)

    np.testing.assert_allclose(
        scaled_predictions,
        y_true,
        rtol=1e-6,
        atol=1e-6,
    )
    assert metrics.rmse == np.sqrt(np.mean(np.square(scaled_predictions - y_true)))
    assert metrics.auc == 1.0
    np.testing.assert_allclose(
        np.array([metrics.pearson_r, metrics.spearman_r], dtype=np.float64),
        np.array([1.0, 1.0], dtype=np.float64),
        rtol=1e-12,
        atol=1e-12,
    )


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
