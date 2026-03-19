from __future__ import annotations

import math
import time
from typing import Any

import numpy as np
import torch

from autoresirch.prepare.comparative.metrics import (
    build_comparative_fold_diagnostics,
    evaluate_comparative_predictions,
)
from autoresirch.prepare.shared.schemas import (
    ARCHITECTURE_CONSTRAINTS,
    METRIC_CONFIG,
    TRAINING_CONFIG,
    ArchitectureConstraints,
    ArchitectureSpec,
    FoldResult,
    FoldSpec,
    MetricConfig,
    ModelBuilder,
    PreparedDataset,
    RegressionMetrics,
    TrainingConfig,
)
from autoresirch.prepare.standard.training import (
    _build_context,
    _fit_flat_preprocessor,
    _is_defined,
    _metrics_improved_for_checkpoint_selection,
    _pick_fold_by_metric,
    _slice_sequence_features,
    _state_dict_to_cpu,
    _train_epoch,
    create_dataloader,
    instantiate_model,
    predict_regression,
    validate_budget,
)
from autoresirch.prepare.shared.utils import set_random_seed


def _validate_comparative_architecture(architecture: ArchitectureSpec) -> None:
    if architecture.use_rnafm_embeddings or architecture.family in {"cnn", "hybrid_cnn_mlp"}:
        raise NotImplementedError(
            "Comparative CNN and hybrid comparative architectures are scaffolded but not implemented yet."
        )


def train_comparative_fold(
    prepared: PreparedDataset,
    fold: FoldSpec,
    architecture: ArchitectureSpec,
    build_model: ModelBuilder,
    *,
    training_config: TrainingConfig = TRAINING_CONFIG,
    metric_config: MetricConfig = METRIC_CONFIG,
    constraints: ArchitectureConstraints = ARCHITECTURE_CONSTRAINTS,
    seed: int,
    budget_seconds: float,
) -> FoldResult:
    _validate_comparative_architecture(architecture)
    train_target = prepared.target[fold.train_indices]
    val_target = prepared.target[fold.val_indices]

    preprocessor, train_flat_features, val_flat_features = _fit_flat_preprocessor(
        prepared,
        fold.train_indices,
        fold.val_indices,
    )
    train_sequence_features, val_sequence_features = _slice_sequence_features(
        prepared,
        fold.train_indices,
        fold.val_indices,
    )

    from autoresirch.prepare.standard.preprocessing import TargetScaler

    target_scaler = TargetScaler.fit(train_target)
    train_target_scaled = target_scaler.transform(train_target)

    context = _build_context(
        training_config=training_config,
        train_size=len(train_target),
        preprocessor=preprocessor,
        train_flat_features=train_flat_features,
        train_sequence_features=train_sequence_features,
        prepared=prepared,
    )
    set_random_seed(seed)
    model, num_params = instantiate_model(architecture, build_model, context, constraints)
    dataloader = create_dataloader(
        train_flat_features,
        train_sequence_features,
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

    best_metrics: RegressionMetrics | None = None
    best_state: dict[str, torch.Tensor] | None = None
    best_diagnostics = None
    best_epoch = 0
    epochs_without_improvement = 0
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
            predict_regression(
                model,
                val_flat_features,
                val_sequence_features,
                training_config.device,
            )
        )
        metrics = evaluate_comparative_predictions(val_target, val_prediction)
        if _metrics_improved_for_checkpoint_selection(
            metrics,
            best_metrics,
            metric_config=metric_config,
        ):
            best_metrics = metrics
            best_diagnostics = build_comparative_fold_diagnostics(val_target, val_prediction)
            best_state = _state_dict_to_cpu(model)
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if (
            training_config.early_stopping_patience is not None
            and epochs_without_improvement >= training_config.early_stopping_patience
        ):
            break
        if time.perf_counter() >= deadline:
            break

    if best_state is None or best_metrics is None or best_diagnostics is None:
        raise RuntimeError("Comparative fold training did not produce any validation metrics.")

    model.load_state_dict(best_state)
    return FoldResult(
        gene=fold.gene,
        count=len(val_target),
        metrics=best_metrics,
        train_seconds=time.perf_counter() - start,
        epochs=epoch,
        best_epoch=best_epoch,
        num_params=num_params,
        diagnostics=best_diagnostics,
    )


def train_comparative_final_holdout(
    prepared: PreparedDataset,
    architecture: ArchitectureSpec,
    build_model: ModelBuilder,
    *,
    training_config: TrainingConfig = TRAINING_CONFIG,
    constraints: ArchitectureConstraints = ARCHITECTURE_CONSTRAINTS,
    seed: int,
    budget_seconds: float,
) -> RegressionMetrics | None:
    _validate_comparative_architecture(architecture)
    if not prepared.test_genes:
        return None

    train_mask = np.isin(prepared.genes, prepared.train_genes)
    test_mask = np.isin(prepared.genes, prepared.test_genes)
    train_indices = np.flatnonzero(train_mask)
    test_indices = np.flatnonzero(test_mask)
    train_target = prepared.target[train_indices]
    test_target = prepared.target[test_indices]

    preprocessor, train_flat_features, test_flat_features = _fit_flat_preprocessor(
        prepared,
        train_indices,
        test_indices,
    )
    train_sequence_features, test_sequence_features = _slice_sequence_features(
        prepared,
        train_indices,
        test_indices,
    )
    from autoresirch.prepare.standard.preprocessing import TargetScaler

    target_scaler = TargetScaler.fit(train_target)
    train_target_scaled = target_scaler.transform(train_target)
    context = _build_context(
        training_config=training_config,
        train_size=len(train_target),
        preprocessor=preprocessor,
        train_flat_features=train_flat_features,
        train_sequence_features=train_sequence_features,
        prepared=prepared,
    )
    set_random_seed(seed)
    model, _ = instantiate_model(architecture, build_model, context, constraints)
    dataloader = create_dataloader(
        train_flat_features,
        train_sequence_features,
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
        predict_regression(
            model,
            test_flat_features,
            test_sequence_features,
            training_config.device,
        )
    )
    return evaluate_comparative_predictions(test_target, prediction)


def aggregate_comparative_fold_results(
    fold_results: list[FoldResult],
    *,
    metric_config: MetricConfig = METRIC_CONFIG,
) -> dict[str, float | None]:
    weights = np.array([result.count for result in fold_results], dtype=np.float64)
    rmse_values = np.array([result.metrics.rmse for result in fold_results], dtype=np.float64)
    mae_values = np.array([result.metrics.mae for result in fold_results], dtype=np.float64)
    squared_error_sum = sum(result.metrics.squared_error_sum for result in fold_results)
    total_examples = int(weights.sum())

    def weighted_optional_metric_mean(values: list[tuple[float | None, int]]) -> float | None:
        valid_values = [
            (value, weight)
            for value, weight in values
            if value is not None and not math.isnan(value)
        ]
        if not valid_values:
            return None
        value_array = np.array([value for value, _ in valid_values], dtype=np.float64)
        weight_array = np.array([weight for _, weight in valid_values], dtype=np.float64)
        return float(np.average(value_array, weights=weight_array))

    weighted_rmse_mean = float(np.average(rmse_values, weights=weights))
    weighted_r2 = weighted_optional_metric_mean(
        [(result.metrics.r2, result.count) for result in fold_results]
    )
    weighted_overall_auc = weighted_optional_metric_mean(
        [(result.metrics.overall_auc, result.count) for result in fold_results]
    )
    weighted_auc_class_neg1 = weighted_optional_metric_mean(
        [(result.metrics.auc_class_neg1, result.count) for result in fold_results]
    )
    weighted_auc_class_0 = weighted_optional_metric_mean(
        [(result.metrics.auc_class_0, result.count) for result in fold_results]
    )
    weighted_auc_class_pos1 = weighted_optional_metric_mean(
        [(result.metrics.auc_class_pos1, result.count) for result in fold_results]
    )
    weighted_auc_pos_vs_neg = weighted_optional_metric_mean(
        [(result.metrics.auc_pos_vs_neg, result.count) for result in fold_results]
    )
    weighted_pearson_r = weighted_optional_metric_mean(
        [(result.metrics.pearson_r, result.count) for result in fold_results]
    )
    weighted_spearman_r = weighted_optional_metric_mean(
        [(result.metrics.spearman_r, result.count) for result in fold_results]
    )

    metric_values: dict[str, float | None] = {
        "weighted_cv_overall_auc": weighted_overall_auc,
        "weighted_cv_auc_pos_vs_neg": weighted_auc_pos_vs_neg,
        "weighted_cv_rmse_mean": weighted_rmse_mean,
    }
    if metric_config.primary_metric_name not in metric_values:
        raise ValueError(f"Unsupported comparative primary metric: {metric_config.primary_metric_name}")
    primary_metric_value = metric_values[metric_config.primary_metric_name]
    if primary_metric_value is None:
        raise ValueError(
            f"Primary metric {metric_config.primary_metric_name} is undefined for the current fold results."
        )

    return {
        "primary_metric_name": metric_config.primary_metric_name,
        "primary_metric_value": float(primary_metric_value),
        "weighted_cv_rmse_mean": weighted_rmse_mean,
        "cv_rmse_mean": float(np.mean(rmse_values)),
        "cv_rmse_std": float(np.std(rmse_values)),
        "weighted_cv_mae_mean": float(np.average(mae_values, weights=weights)),
        "weighted_cv_r2_mean": weighted_r2,
        "weighted_cv_auc_mean": None,
        "weighted_cv_pearson_r_mean": weighted_pearson_r,
        "weighted_cv_spearman_r_mean": weighted_spearman_r,
        "weighted_cv_overall_auc": weighted_overall_auc,
        "weighted_cv_auc_class_neg1": weighted_auc_class_neg1,
        "weighted_cv_auc_class_0": weighted_auc_class_0,
        "weighted_cv_auc_class_pos1": weighted_auc_class_pos1,
        "weighted_cv_auc_pos_vs_neg": weighted_auc_pos_vs_neg,
        "pooled_cv_rmse": float(math.sqrt(squared_error_sum / total_examples)),
    }


def build_comparative_run_diagnostics(fold_results: list[FoldResult]) -> dict[str, Any]:
    fold_sizes = np.array([result.count for result in fold_results], dtype=np.float64)
    epochs = np.array([result.epochs for result in fold_results], dtype=np.float64)
    best_epochs = np.array([result.best_epoch for result in fold_results], dtype=np.float64)

    return {
        "fold_count": len(fold_results),
        "fold_sizes": {
            "min": int(np.min(fold_sizes)),
            "median": float(np.median(fold_sizes)),
            "max": int(np.max(fold_sizes)),
        },
        "nan_metric_counts": {
            "auc": sum(not _is_defined(result.metrics.overall_auc) for result in fold_results),
            "pearson_r": sum(not _is_defined(result.metrics.pearson_r) for result in fold_results),
            "spearman_r": sum(not _is_defined(result.metrics.spearman_r) for result in fold_results),
        },
        "undefined_metric_counts": {
            "overall_auc": sum(not _is_defined(result.metrics.overall_auc) for result in fold_results),
            "auc_class_neg1": sum(not _is_defined(result.metrics.auc_class_neg1) for result in fold_results),
            "auc_class_0": sum(not _is_defined(result.metrics.auc_class_0) for result in fold_results),
            "auc_class_pos1": sum(not _is_defined(result.metrics.auc_class_pos1) for result in fold_results),
            "auc_pos_vs_neg": sum(not _is_defined(result.metrics.auc_pos_vs_neg) for result in fold_results),
        },
        "best_auc_fold": _pick_fold_by_metric(fold_results, metric_name="overall_auc", reverse=True),
        "worst_auc_fold": _pick_fold_by_metric(fold_results, metric_name="overall_auc", reverse=False),
        "best_rmse_fold": _pick_fold_by_metric(fold_results, metric_name="rmse", reverse=False),
        "worst_rmse_fold": _pick_fold_by_metric(fold_results, metric_name="rmse", reverse=True),
        "largest_gene_fold": {
            "gene": max(fold_results, key=lambda result: result.count).gene,
            "count": max(fold_results, key=lambda result: result.count).count,
        },
        "smallest_gene_fold": {
            "gene": min(fold_results, key=lambda result: result.count).gene,
            "count": min(fold_results, key=lambda result: result.count).count,
        },
        "training_dynamics": {
            "epoch_count_min": int(np.min(epochs)),
            "epoch_count_median": float(np.median(epochs)),
            "epoch_count_max": int(np.max(epochs)),
            "best_epoch_min": int(np.min(best_epochs)),
            "best_epoch_median": float(np.median(best_epochs)),
            "best_epoch_max": int(np.max(best_epochs)),
            "best_epoch_ratio_mean": float(np.mean(best_epochs / epochs)),
            "seconds_per_fold_mean": float(np.mean([result.train_seconds for result in fold_results])),
        },
        "class_support": {
            "folds_missing_neg1": sum((result.diagnostics.class_count_neg1 or 0) == 0 for result in fold_results),
            "folds_missing_0": sum((result.diagnostics.class_count_0 or 0) == 0 for result in fold_results),
            "folds_missing_pos1": sum((result.diagnostics.class_count_pos1 or 0) == 0 for result in fold_results),
        },
    }
