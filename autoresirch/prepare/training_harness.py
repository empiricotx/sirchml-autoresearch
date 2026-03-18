from __future__ import annotations

import math
import time
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from .architecture_loading import validate_architecture_spec
from .fold_preprocessor import FoldPreprocessor, TargetScaler
from .schemas import (
    ARCHITECTURE_CONSTRAINTS,
    METRIC_CONFIG,
    TRAINING_CONFIG,
    ArchitectureConstraints,
    ArchitectureContext,
    ArchitectureSpec,
    FoldDiagnostics,
    FoldResult,
    FoldSpec,
    MetricConfig,
    ModelBuilder,
    PreparedDataset,
    RegressionMetrics,
    TrainingConfig,
)
from .utils import (
    _normalize_model_output,
    _state_dict_to_cpu,
    build_fold_diagnostics,
    count_parameters,
    evaluate_predictions,
    set_random_seed,
)


BatchPayload = dict[str, torch.Tensor]


class MultiModalTensorDataset(Dataset[BatchPayload]):
    def __init__(
        self,
        *,
        flat_features: np.ndarray | None,
        sequence_features: np.ndarray | None,
        target: np.ndarray,
    ) -> None:
        if flat_features is None and sequence_features is None:
            raise ValueError("At least one feature block must be provided.")
        self.flat_features = (
            None if flat_features is None else torch.from_numpy(flat_features.astype(np.float32))
        )
        self.sequence_features = (
            None
            if sequence_features is None
            else torch.from_numpy(sequence_features.astype(np.float32))
        )
        self.target = torch.from_numpy(target.astype(np.float32)).unsqueeze(1)

    def __len__(self) -> int:
        return int(self.target.shape[0])

    def __getitem__(self, index: int) -> BatchPayload:
        batch: BatchPayload = {"target": self.target[index]}
        if self.flat_features is not None:
            batch["flat"] = self.flat_features[index]
        if self.sequence_features is not None:
            batch["sequence"] = self.sequence_features[index]
        return batch


def _call_model(
    model: nn.Module,
    *,
    flat: torch.Tensor | None,
    sequence: torch.Tensor | None,
) -> torch.Tensor:
    if sequence is None:
        if flat is None:
            raise ValueError("Model input is empty.")
        return _normalize_model_output(model(flat))
    return _normalize_model_output(model(flat=flat, sequence=sequence))


def _make_dummy_output(
    model: nn.Module,
    *,
    context: ArchitectureContext,
) -> torch.Tensor:
    batch_size = 2
    dummy_flat: torch.Tensor | None = None
    dummy_sequence: torch.Tensor | None = None
    if context.has_flat_features and context.flat_input_dim:
        dummy_flat = torch.zeros(batch_size, context.flat_input_dim, dtype=torch.float32)
    if (
        context.has_sequence_features
        and context.sequence_length is not None
        and context.sequence_embedding_dim is not None
    ):
        dummy_sequence = torch.zeros(
            batch_size,
            context.sequence_length,
            context.sequence_embedding_dim,
            dtype=torch.float32,
        )
    return _call_model(model, flat=dummy_flat, sequence=dummy_sequence)


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
        output = _make_dummy_output(model, context=context)
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
    flat_features: np.ndarray | None,
    sequence_features: np.ndarray | None,
    target: np.ndarray,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
    pin_memory: bool,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    dataset = MultiModalTensorDataset(
        flat_features=flat_features,
        sequence_features=sequence_features,
        target=target,
    )
    effective_batch_size = min(batch_size, len(dataset))
    return DataLoader(
        dataset,
        batch_size=effective_batch_size,
        shuffle=shuffle,
        generator=generator,
        pin_memory=pin_memory,
    )


def predict_regression(
    model: nn.Module,
    flat_features: np.ndarray | None,
    sequence_features: np.ndarray | None,
    device: str,
) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        flat_tensor = None
        sequence_tensor = None
        if flat_features is not None:
            flat_tensor = torch.from_numpy(flat_features.astype(np.float32)).to(device)
        if sequence_features is not None:
            sequence_tensor = torch.from_numpy(sequence_features.astype(np.float32)).to(device)
        prediction = _call_model(model, flat=flat_tensor, sequence=sequence_tensor)
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
    for batch in dataloader:
        batch_target = batch["target"].to(device, non_blocking=True)
        flat = batch.get("flat")
        sequence = batch.get("sequence")
        if flat is not None:
            flat = flat.to(device, non_blocking=True)
        if sequence is not None:
            sequence = sequence.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        prediction = _call_model(model, flat=flat, sequence=sequence)
        loss = F.mse_loss(prediction, batch_target[:, 0])
        loss.backward()
        if grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizer.step()


def _fit_flat_preprocessor(
    prepared: PreparedDataset,
    train_indices: np.ndarray,
    eval_indices: np.ndarray,
) -> tuple[FoldPreprocessor | None, np.ndarray | None, np.ndarray | None]:
    if not prepared.has_flat_features:
        return None, None, None
    train_frame = prepared.features.iloc[train_indices]
    eval_frame = prepared.features.iloc[eval_indices]
    preprocessor = FoldPreprocessor(
        numeric_columns=prepared.numeric_feature_columns,
        categorical_columns=prepared.categorical_feature_columns,
    ).fit(train_frame)
    return preprocessor, preprocessor.transform(train_frame), preprocessor.transform(eval_frame)


def _slice_sequence_features(
    prepared: PreparedDataset,
    train_indices: np.ndarray,
    eval_indices: np.ndarray,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    if prepared.sequence_features is None:
        return None, None
    return (
        prepared.sequence_features[train_indices].astype(np.float32, copy=False),
        prepared.sequence_features[eval_indices].astype(np.float32, copy=False),
    )


def _build_context(
    *,
    training_config: TrainingConfig,
    train_size: int,
    preprocessor: FoldPreprocessor | None,
    train_flat_features: np.ndarray | None,
    train_sequence_features: np.ndarray | None,
    prepared: PreparedDataset,
) -> ArchitectureContext:
    flat_feature_names = () if preprocessor is None else preprocessor.feature_names
    flat_input_dim = None if train_flat_features is None else int(train_flat_features.shape[1])
    sequence_length = None
    sequence_embedding_dim = None
    if train_sequence_features is not None:
        sequence_length = int(train_sequence_features.shape[1])
        sequence_embedding_dim = int(train_sequence_features.shape[2])
    return ArchitectureContext(
        input_dim=0 if flat_input_dim is None else flat_input_dim,
        output_dim=1,
        train_size=train_size,
        feature_names=flat_feature_names,
        device=training_config.device,
        flat_input_dim=flat_input_dim,
        sequence_length=sequence_length,
        sequence_channels=sequence_embedding_dim,
        sequence_embedding_dim=sequence_embedding_dim,
        has_flat_features=train_flat_features is not None,
        has_sequence_features=train_sequence_features is not None,
        flat_feature_names=flat_feature_names,
        sequence_source_name=prepared.sequence_feature_name,
    )


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

    pin_memory = training_config.device.startswith("cuda")
    dataloader = create_dataloader(
        train_flat_features,
        train_sequence_features,
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
    best_diagnostics: FoldDiagnostics | None = None
    best_state: dict[str, torch.Tensor] | None = None
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
        metrics = evaluate_predictions(val_target, val_prediction)
        if best_metrics is None or metrics.rmse < best_metrics.rmse:
            best_metrics = metrics
            best_diagnostics = build_fold_diagnostics(val_target, val_prediction)
            best_epoch = epoch
            best_state = _state_dict_to_cpu(model)
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
        diagnostics=best_diagnostics,
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

    weighted_r2 = weighted_optional_metric_mean(
        [(result.metrics.r2, result.count) for result in fold_results]
    )
    weighted_auc = weighted_optional_metric_mean(
        [(result.metrics.auc, result.count) for result in fold_results]
    )
    weighted_pearson_r = weighted_optional_metric_mean(
        [(result.metrics.pearson_r, result.count) for result in fold_results]
    )
    weighted_spearman_r = weighted_optional_metric_mean(
        [(result.metrics.spearman_r, result.count) for result in fold_results]
    )

    weighted_rmse_mean = float(np.average(rmse_values, weights=weights))
    metric_values: dict[str, float | None] = {
        "weighted_cv_rmse_mean": weighted_rmse_mean,
        "weighted_cv_auc": weighted_auc,
    }
    if metric_config.primary_metric_name not in metric_values:
        raise ValueError(f"Unsupported primary metric: {metric_config.primary_metric_name}")
    primary_metric_value = metric_values[metric_config.primary_metric_name]
    if primary_metric_value is None:
        raise ValueError(
            f"Primary metric {metric_config.primary_metric_name} is undefined for the current fold results."
        )
    return {
        "primary_metric_value": float(primary_metric_value),
        "weighted_cv_rmse_mean": weighted_rmse_mean,
        "cv_rmse_mean": float(np.mean(rmse_values)),
        "cv_rmse_std": float(np.std(rmse_values)),
        "weighted_cv_mae_mean": float(np.average(mae_values, weights=weights)),
        "weighted_cv_r2_mean": weighted_r2,
        "weighted_cv_auc_mean": weighted_auc,
        "weighted_cv_pearson_r_mean": weighted_pearson_r,
        "weighted_cv_spearman_r_mean": weighted_spearman_r,
        "pooled_cv_rmse": float(math.sqrt(squared_error_sum / total_examples)),
        "primary_metric_name": metric_config.primary_metric_name,
    }


def _is_defined(value: float | None) -> bool:
    return value is not None and not math.isnan(value)


def _pick_fold_by_metric(
    fold_results: list[FoldResult],
    *,
    metric_name: str,
    reverse: bool,
) -> dict[str, Any] | None:
    candidates: list[tuple[FoldResult, float]] = []
    for result in fold_results:
        value = getattr(result.metrics, metric_name)
        if _is_defined(value):
            candidates.append((result, float(value)))
    if not candidates:
        return None
    selected_result, selected_value = sorted(
        candidates,
        key=lambda item: item[1],
        reverse=reverse,
    )[0]
    return {
        "gene": selected_result.gene,
        "count": selected_result.count,
        metric_name: selected_value,
    }


def build_run_diagnostics(fold_results: list[FoldResult]) -> dict[str, Any]:
    fold_sizes = np.array([result.count for result in fold_results], dtype=np.float64)
    epochs = np.array([result.epochs for result in fold_results], dtype=np.float64)
    best_epochs = np.array([result.best_epoch for result in fold_results], dtype=np.float64)

    def weighted_optional_diagnostic_mean(values: list[tuple[float | None, int]]) -> float | None:
        valid_values = [
            (value, weight)
            for value, weight in values
            if value is not None
        ]
        if not valid_values:
            return None
        value_array = np.array([value for value, _ in valid_values], dtype=np.float64)
        weight_array = np.array([weight for _, weight in valid_values], dtype=np.float64)
        return float(np.average(value_array, weights=weight_array))

    return {
        "fold_count": len(fold_results),
        "fold_sizes": {
            "min": int(np.min(fold_sizes)),
            "median": float(np.median(fold_sizes)),
            "max": int(np.max(fold_sizes)),
        },
        "nan_metric_counts": {
            "auc": sum(not _is_defined(result.metrics.auc) for result in fold_results),
            "pearson_r": sum(not _is_defined(result.metrics.pearson_r) for result in fold_results),
            "spearman_r": sum(not _is_defined(result.metrics.spearman_r) for result in fold_results),
        },
        "best_auc_fold": _pick_fold_by_metric(fold_results, metric_name="auc", reverse=True),
        "worst_auc_fold": _pick_fold_by_metric(fold_results, metric_name="auc", reverse=False),
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
        "prediction_behavior": {
            "weighted_scaled_prediction_mean": weighted_optional_diagnostic_mean(
                [
                    (result.diagnostics.scaled_prediction_mean, result.count)
                    for result in fold_results
                ]
            ),
            "weighted_scaled_prediction_std": weighted_optional_diagnostic_mean(
                [
                    (result.diagnostics.scaled_prediction_std, result.count)
                    for result in fold_results
                ]
            ),
            "weighted_clipped_low_fraction": weighted_optional_diagnostic_mean(
                [
                    (result.diagnostics.clipped_low_fraction, result.count)
                    for result in fold_results
                ]
            ),
            "weighted_clipped_high_fraction": weighted_optional_diagnostic_mean(
                [
                    (result.diagnostics.clipped_high_fraction, result.count)
                    for result in fold_results
                ]
            ),
            "weighted_effective_positive_rate": weighted_optional_diagnostic_mean(
                [
                    (result.diagnostics.effective_positive_rate, result.count)
                    for result in fold_results
                ]
            ),
        },
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
