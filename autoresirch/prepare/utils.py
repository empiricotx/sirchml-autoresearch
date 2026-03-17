from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from dataclasses import asdict
import math
import time

import numpy as np
import pandas as pd
import torch
from torch import nn

from .schemas import (
    ARCHITECTURE_CONSTRAINTS,
    DATASET_CONFIG,
    METRIC_CONFIG,
    SPLIT_CONFIG,
    TRAINING_CONFIG,
    CACHE_DIR,
    RUNS_DIR,
    RESULTS_TSV,
    RESULTS_HEADER,
    MetricConfig,
    RegressionMetrics,
    FoldDiagnostics,
)


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


def pearson_r_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size < 2:
        return math.nan
    centered_true = y_true.astype(np.float64) - float(np.mean(y_true))
    centered_pred = y_pred.astype(np.float64) - float(np.mean(y_pred))
    denominator = float(
        np.sqrt(np.sum(np.square(centered_true)) * np.sum(np.square(centered_pred)))
    )
    if denominator <= 0:
        return math.nan
    return float(np.sum(centered_true * centered_pred) / denominator)


def spearman_r_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size < 2:
        return math.nan
    ranked_true = pd.Series(y_true).rank(method="average").to_numpy(dtype=np.float64)
    ranked_pred = pd.Series(y_pred).rank(method="average").to_numpy(dtype=np.float64)
    return pearson_r_score(ranked_true, ranked_pred)


def scale_regression_predictions(
    y_pred: np.ndarray,
    metric_config: MetricConfig = METRIC_CONFIG,
) -> np.ndarray:
    prediction_span = metric_config.prediction_scale_max - metric_config.prediction_scale_min
    if prediction_span <= 0:
        raise ValueError("MetricConfig prediction scale must have a positive span.")
    scaled_predictions = (y_pred - metric_config.prediction_scale_min) / prediction_span
    return np.clip(scaled_predictions, 0.0, 1.0).astype(np.float32)


def binary_effective_labels(
    y_true: np.ndarray,
    metric_config: MetricConfig = METRIC_CONFIG,
) -> np.ndarray:
    return (y_true < metric_config.effective_threshold).astype(np.int8)


def build_fold_diagnostics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_config: MetricConfig = METRIC_CONFIG,
) -> FoldDiagnostics:
    scaled_predictions = scale_regression_predictions(y_pred, metric_config)
    effective_labels = binary_effective_labels(y_true, metric_config)
    return FoldDiagnostics(
        scaled_prediction_mean=float(np.mean(scaled_predictions)),
        scaled_prediction_std=float(np.std(scaled_predictions)),
        clipped_low_fraction=float(np.mean(scaled_predictions == 0.0)),
        clipped_high_fraction=float(np.mean(scaled_predictions == 1.0)),
        effective_positive_rate=float(np.mean(effective_labels)),
    )


def roc_auc_score_binary(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if y_true.ndim != 1 or y_score.ndim != 1:
        raise ValueError("roc_auc_score_binary expects 1D arrays.")
    if y_true.shape[0] != y_score.shape[0]:
        raise ValueError("roc_auc_score_binary expects arrays of equal length.")

    positives = int(y_true.sum())
    negatives = int(y_true.shape[0] - positives)
    if positives == 0 or negatives == 0:
        return math.nan

    ranks = pd.Series(y_score).rank(method="average").to_numpy(dtype=np.float64)
    positive_rank_sum = float(ranks[y_true.astype(bool)].sum())
    u_statistic = positive_rank_sum - (positives * (positives + 1) / 2.0)
    return float(u_statistic / (positives * negatives))


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_config: MetricConfig = METRIC_CONFIG,
) -> RegressionMetrics:
    scaled_predictions = scale_regression_predictions(y_pred, metric_config)
    squared_error_sum = float(np.sum(np.square(scaled_predictions - y_true)))
    effective_labels = binary_effective_labels(y_true, metric_config)
    effective_scores = 1.0 - scaled_predictions
    return RegressionMetrics(
        rmse=rmse(y_true, scaled_predictions),
        mae=mae(y_true, scaled_predictions),
        r2=r2_score(y_true, scaled_predictions),
        squared_error_sum=squared_error_sum,
        auc=roc_auc_score_binary(effective_labels, effective_scores),
        pearson_r=pearson_r_score(y_true, scaled_predictions),
        spearman_r=spearman_r_score(y_true, scaled_predictions),
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


def _make_run_dir(root: Path | None = None) -> Path:
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    run_root = root or RUNS_DIR
    run_dir = run_root / timestamp
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir