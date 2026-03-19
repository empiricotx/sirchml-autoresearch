from __future__ import annotations

import math

import numpy as np

from autoresirch.prepare.schemas import FoldDiagnostics, METRIC_CONFIG, MetricConfig, RegressionMetrics
from autoresirch.prepare.utils import (
    mae,
    pearson_r_score,
    r2_score,
    rmse,
    roc_auc_score_binary,
    spearman_r_score,
)


COMPARATIVE_CLASS_VALUES: tuple[int, ...] = (-1, 0, 1)


def comparative_class_labels(
    target_delta: np.ndarray,
    metric_config: MetricConfig = METRIC_CONFIG,
) -> np.ndarray:
    labels = np.zeros(target_delta.shape[0], dtype=np.int8)
    labels[target_delta < metric_config.comparative_no_effect_lower] = -1
    labels[target_delta > metric_config.comparative_no_effect_upper] = 1
    return labels


def _comparative_score_by_class(
    prediction_delta: np.ndarray,
) -> dict[int, np.ndarray]:
    return {
        -1: (-prediction_delta).astype(np.float32, copy=False),
        0: (-np.abs(prediction_delta)).astype(np.float32, copy=False),
        1: prediction_delta.astype(np.float32, copy=False),
    }


def _defined_metric(value: float | None) -> float | None:
    if value is None:
        return None
    if math.isnan(value):
        return None
    return float(value)


def _one_vs_rest_auc(
    y_true_class: np.ndarray,
    *,
    target_class: int,
    score: np.ndarray,
) -> float | None:
    binary_labels = (y_true_class == target_class).astype(np.int8)
    auc = roc_auc_score_binary(binary_labels, score)
    return _defined_metric(auc)


def evaluate_comparative_predictions(
    y_true_delta: np.ndarray,
    y_pred_delta: np.ndarray,
    metric_config: MetricConfig = METRIC_CONFIG,
) -> RegressionMetrics:
    y_true_class = comparative_class_labels(y_true_delta, metric_config)
    class_scores = _comparative_score_by_class(y_pred_delta)

    auc_class_neg1 = _one_vs_rest_auc(y_true_class, target_class=-1, score=class_scores[-1])
    auc_class_0 = _one_vs_rest_auc(y_true_class, target_class=0, score=class_scores[0])
    auc_class_pos1 = _one_vs_rest_auc(y_true_class, target_class=1, score=class_scores[1])
    class_auc_values = (auc_class_neg1, auc_class_0, auc_class_pos1)
    overall_auc = (
        None
        if any(value is None for value in class_auc_values)
        else float(np.mean(np.array(class_auc_values, dtype=np.float64)))
    )

    pos_neg_mask = y_true_class != 0
    auc_pos_vs_neg: float | None = None
    if int(np.sum(pos_neg_mask)) >= 2:
        pos_neg_labels = (y_true_class[pos_neg_mask] == 1).astype(np.int8)
        auc_pos_vs_neg = _defined_metric(
            roc_auc_score_binary(pos_neg_labels, y_pred_delta[pos_neg_mask].astype(np.float32, copy=False))
        )

    return RegressionMetrics(
        rmse=rmse(y_true_delta, y_pred_delta),
        mae=mae(y_true_delta, y_pred_delta),
        r2=r2_score(y_true_delta, y_pred_delta),
        squared_error_sum=float(np.sum(np.square(y_pred_delta - y_true_delta))),
        auc=None,
        pearson_r=_defined_metric(pearson_r_score(y_true_delta, y_pred_delta)),
        spearman_r=_defined_metric(spearman_r_score(y_true_delta, y_pred_delta)),
        overall_auc=overall_auc,
        auc_class_neg1=auc_class_neg1,
        auc_class_0=auc_class_0,
        auc_class_pos1=auc_class_pos1,
        auc_pos_vs_neg=auc_pos_vs_neg,
    )


def build_comparative_fold_diagnostics(
    y_true_delta: np.ndarray,
    y_pred_delta: np.ndarray,
    metric_config: MetricConfig = METRIC_CONFIG,
) -> FoldDiagnostics:
    metrics = evaluate_comparative_predictions(y_true_delta, y_pred_delta, metric_config)
    labels = comparative_class_labels(y_true_delta, metric_config)
    undefined_auc_metrics = tuple(
        metric_name
        for metric_name, value in (
            ("overall_auc", metrics.overall_auc),
            ("auc_class_neg1", metrics.auc_class_neg1),
            ("auc_class_0", metrics.auc_class_0),
            ("auc_class_pos1", metrics.auc_class_pos1),
            ("auc_pos_vs_neg", metrics.auc_pos_vs_neg),
        )
        if value is None
    )
    return FoldDiagnostics(
        class_count_neg1=int(np.sum(labels == -1)),
        class_count_0=int(np.sum(labels == 0)),
        class_count_pos1=int(np.sum(labels == 1)),
        undefined_auc_metrics=undefined_auc_metrics,
    )
