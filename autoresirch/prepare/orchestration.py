from __future__ import annotations
from pathlib import Path
import json
from dataclasses import asdict
import math
import time

from .schemas import ExperimentSummary, FoldResult, ArchitectureSpec, ModelBuilder, DatasetConfig, SplitConfig, TrainingConfig, MetricConfig, ArchitectureConstraints, ARCHITECTURE_CONSTRAINTS, TRAINING_CONFIG, DATASET_CONFIG, SPLIT_CONFIG, METRIC_CONFIG, RUNS_DIR, _json_default, _make_run_dir, build_run_diagnostics
from .dataset_preparation import prepare_dataset
from .fold_preprocessor import build_cv_folds
from .training_harness import train_fold, train_final_holdout, aggregate_fold_results, validate_budget
from .architecture_loading import validate_architecture_spec, load_train_definition
from .utils import ensure_runtime_dirs, _make_run_dir


def save_run_summary(
    summary: ExperimentSummary,
    fold_results: list[FoldResult],
    architecture: ArchitectureSpec,
    *,
    run_dir: Path,
    latest_summary_path: Path | None = None,
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
                "diagnostics": asdict(result.diagnostics),
            }
            for result in fold_results
        ],
        "diagnostics": build_run_diagnostics(fold_results),
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
    target_latest_summary_path = latest_summary_path or RUNS_DIR.joinpath("latest_summary.json")
    target_latest_summary_path.parent.mkdir(parents=True, exist_ok=True)
    target_latest_summary_path.write_text(
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
    if summary.weighted_cv_auc_mean is None:
        print("weighted_cv_auc:     nan")
    else:
        print(f"weighted_cv_auc:     {summary.weighted_cv_auc_mean:.6f}")
    if summary.weighted_cv_pearson_r_mean is None:
        print("weighted_cv_pearson_r: nan")
    else:
        print(f"weighted_cv_pearson_r: {summary.weighted_cv_pearson_r_mean:.6f}")
    if summary.weighted_cv_spearman_r_mean is None:
        print("weighted_cv_spearman_r: nan")
    else:
        print(f"weighted_cv_spearman_r: {summary.weighted_cv_spearman_r_mean:.6f}")
    print(f"pooled_cv_rmse:      {summary.pooled_cv_rmse:.6f}")
    if summary.test_rmse is None:
        print("test_rmse:           nan")
        print("test_mae:            nan")
        print("test_r2:             nan")
        print("test_auc:            nan")
        print("test_pearson_r:      nan")
        print("test_spearman_r:     nan")
    else:
        print(f"test_rmse:           {summary.test_rmse:.6f}")
        print(f"test_mae:            {summary.test_mae:.6f}")
        if summary.test_r2 is None:
            print("test_r2:             nan")
        else:
            print(f"test_r2:             {summary.test_r2:.6f}")
        if summary.test_auc is None:
            print("test_auc:            nan")
        else:
            print(f"test_auc:            {summary.test_auc:.6f}")
        if summary.test_pearson_r is None:
            print("test_pearson_r:      nan")
        else:
            print(f"test_pearson_r:      {summary.test_pearson_r:.6f}")
        if summary.test_spearman_r is None:
            print("test_spearman_r:     nan")
        else:
            print(f"test_spearman_r:     {summary.test_spearman_r:.6f}")
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
    run_dir: Path | None = None,
    latest_summary_path: Path | None = None,
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

    active_run_dir = run_dir
    if active_run_dir is None:
        active_run_dir = _make_run_dir()
    else:
        active_run_dir.mkdir(parents=True, exist_ok=True)
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
        weighted_cv_auc_mean=(
            None if aggregate["weighted_cv_auc_mean"] is None else float(aggregate["weighted_cv_auc_mean"])
        ),
        weighted_cv_pearson_r_mean=(
            None
            if aggregate["weighted_cv_pearson_r_mean"] is None
            else float(aggregate["weighted_cv_pearson_r_mean"])
        ),
        weighted_cv_spearman_r_mean=(
            None
            if aggregate["weighted_cv_spearman_r_mean"] is None
            else float(aggregate["weighted_cv_spearman_r_mean"])
        ),
        pooled_cv_rmse=float(aggregate["pooled_cv_rmse"]),
        test_rmse=None if holdout_metrics is None else holdout_metrics.rmse,
        test_mae=None if holdout_metrics is None else holdout_metrics.mae,
        test_r2=None if holdout_metrics is None or math.isnan(holdout_metrics.r2) else holdout_metrics.r2,
        test_auc=(
            None
            if holdout_metrics is None or holdout_metrics.auc is None or math.isnan(holdout_metrics.auc)
            else holdout_metrics.auc
        ),
        test_pearson_r=(
            None
            if holdout_metrics is None
            or holdout_metrics.pearson_r is None
            or math.isnan(holdout_metrics.pearson_r)
            else holdout_metrics.pearson_r
        ),
        test_spearman_r=(
            None
            if holdout_metrics is None
            or holdout_metrics.spearman_r is None
            or math.isnan(holdout_metrics.spearman_r)
            else holdout_metrics.spearman_r
        ),
        num_params=num_params,
        train_seconds=time.perf_counter() - start,
        feature_dim=prepared.features.shape[1],
        num_rows=len(prepared.target),
        cv_folds=len(folds),
        train_genes=prepared.train_genes,
        test_genes=prepared.test_genes,
        cv_genes=prepared.cv_genes,
        run_dir=str(active_run_dir),
    )
    save_run_summary(
        summary,
        fold_results,
        architecture,
        run_dir=active_run_dir,
        latest_summary_path=latest_summary_path,
    )
    print_experiment_summary(summary)
    return summary