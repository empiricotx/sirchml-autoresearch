from __future__ import annotations

from dataclasses import asdict, replace
import json
import math
import time
from pathlib import Path

from autoresirch.prepare.comparative import (
    aggregate_comparative_fold_results,
    build_comparative_run_diagnostics,
    train_comparative_final_holdout,
    train_comparative_fold,
)
from autoresirch.prepare.architecture_loading import load_train_definition, validate_architecture_spec
from autoresirch.prepare.shared.schemas import (
    ARCHITECTURE_CONSTRAINTS,
    DATASET_CONFIG,
    METRIC_CONFIG,
    RUNS_DIR,
    SPLIT_CONFIG,
    TRAINING_CONFIG,
    ArchitectureConstraints,
    ArchitectureSpec,
    DatasetConfig,
    ExperimentSummary,
    FoldResult,
    MetricConfig,
    ModelBuilder,
    SplitConfig,
    TrainingConfig,
)
from autoresirch.prepare.shared.utils import (
    _json_default,
    _make_run_dir,
    ensure_runtime_dirs,
    resolve_primary_metric_name,
)
from autoresirch.prepare.standard.dataset import prepare_dataset
from autoresirch.prepare.standard.preprocessing import build_cv_folds
from autoresirch.prepare.standard.training import (
    aggregate_fold_results,
    build_run_diagnostics,
    train_final_holdout,
    train_fold,
    validate_budget,
)

def save_run_summary(
    summary: ExperimentSummary,
    fold_results: list[FoldResult],
    architecture: ArchitectureSpec,
    *,
    run_dir: Path,
    latest_summary_path: Path | None = None,
    dataset_config: DatasetConfig = DATASET_CONFIG,
    split_config: SplitConfig = SPLIT_CONFIG,
    training_config: TrainingConfig = TRAINING_CONFIG,
    metric_config: MetricConfig = METRIC_CONFIG,
    constraints: ArchitectureConstraints = ARCHITECTURE_CONSTRAINTS,
) -> None:
    if summary.experiment_mode == "comparative":
        diagnostics_payload = build_comparative_run_diagnostics(fold_results)
    else:
        diagnostics_payload = build_run_diagnostics(fold_results)
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
        "diagnostics": diagnostics_payload,
        "constraints": asdict(constraints),
        "training_config": asdict(training_config),
        "dataset_config": asdict(dataset_config),
        "split_config": asdict(split_config),
        "metric_config": asdict(metric_config),
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
    print(f"experiment_mode:     {summary.experiment_mode}")
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
    if summary.experiment_mode == "comparative":
        if summary.weighted_cv_overall_auc is None:
            print("weighted_cv_overall_auc: nan")
        else:
            print(f"weighted_cv_overall_auc: {summary.weighted_cv_overall_auc:.6f}")
        for metric_name, metric_value in (
            ("weighted_cv_auc_class_neg1", summary.weighted_cv_auc_class_neg1),
            ("weighted_cv_auc_class_0", summary.weighted_cv_auc_class_0),
            ("weighted_cv_auc_class_pos1", summary.weighted_cv_auc_class_pos1),
            ("weighted_cv_auc_pos_vs_neg", summary.weighted_cv_auc_pos_vs_neg),
        ):
            if metric_value is None:
                print(f"{metric_name}: nan")
            else:
                print(f"{metric_name}: {metric_value:.6f}")
    elif summary.weighted_cv_auc_mean is None:
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
        if summary.experiment_mode == "comparative":
            print("test_overall_auc:    nan")
            print("test_auc_class_neg1: nan")
            print("test_auc_class_0:    nan")
            print("test_auc_class_pos1: nan")
            print("test_auc_pos_vs_neg: nan")
        else:
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
        if summary.experiment_mode == "comparative":
            for metric_name, metric_value in (
                ("test_overall_auc", summary.test_overall_auc),
                ("test_auc_class_neg1", summary.test_auc_class_neg1),
                ("test_auc_class_0", summary.test_auc_class_0),
                ("test_auc_class_pos1", summary.test_auc_class_pos1),
                ("test_auc_pos_vs_neg", summary.test_auc_pos_vs_neg),
            ):
                if metric_value is None:
                    print(f"{metric_name}: nan")
                else:
                    print(f"{metric_name}: {metric_value:.6f}")
        elif summary.test_auc is None:
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
    prepared_dataset_cache_dir: Path | None = None,
) -> ExperimentSummary:
    ensure_runtime_dirs()

    if architecture is None or build_model is None:
        loaded = load_train_definition()
        architecture = loaded.spec
        build_model = loaded.build_model

    validate_architecture_spec(architecture, constraints)
    active_metric_config = metric_config
    resolved_primary_metric_name = resolve_primary_metric_name(dataset_config.experiment_mode, metric_config)
    if resolved_primary_metric_name != metric_config.primary_metric_name:
        active_metric_config = replace(metric_config, primary_metric_name=resolved_primary_metric_name)
    prepared = prepare_dataset(
        dataset_config=dataset_config,
        split_config=split_config,
        include_rnafm_embeddings=architecture.use_rnafm_embeddings,
        artifact_root=prepared_dataset_cache_dir,
    )
    folds = build_cv_folds(prepared)
    fold_budget, final_budget = validate_budget(
        len(folds),
        training_config=training_config,
        has_holdout_test=bool(prepared.test_genes),
    )

    start = time.perf_counter()
    fold_results: list[FoldResult] = []
    num_params: int | None = None

    if prepared.experiment_mode == "comparative":
        fold_train_fn = train_comparative_fold
        aggregate_fn = aggregate_comparative_fold_results
        holdout_train_fn = train_comparative_final_holdout
    else:
        fold_train_fn = train_fold
        aggregate_fn = aggregate_fold_results
        holdout_train_fn = train_final_holdout

    for fold_index, fold in enumerate(folds):
        result = fold_train_fn(
            prepared,
            fold,
            architecture,
            build_model,
            training_config=training_config,
            metric_config=active_metric_config,
            constraints=constraints,
            seed=split_config.random_seed + fold_index,
            budget_seconds=fold_budget,
        )
        fold_results.append(result)
        if num_params is None:
            num_params = result.num_params

    if num_params is None:
        raise RuntimeError("Experiment produced no fold results.")

    aggregate = aggregate_fn(fold_results, metric_config=active_metric_config)
    holdout_metrics = None
    if training_config.evaluate_test_split and prepared.test_genes and final_budget > 0:
        holdout_metrics = holdout_train_fn(
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
        active_run_dir = _make_run_dir(RUNS_DIR)
    else:
        active_run_dir.mkdir(parents=True, exist_ok=True)
    summary = ExperimentSummary(
        primary_metric_name=str(aggregate["primary_metric_name"]),
        primary_metric_value=float(aggregate["primary_metric_value"]),
        metric_direction=active_metric_config.primary_metric_direction,
        improvement_epsilon=active_metric_config.improvement_epsilon,
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
        feature_dim=prepared.feature_dim,
        num_rows=len(prepared.target),
        cv_folds=len(folds),
        train_genes=prepared.train_genes,
        test_genes=prepared.test_genes,
        cv_genes=prepared.cv_genes,
        run_dir=str(active_run_dir),
        experiment_mode=prepared.experiment_mode,
        label_threshold_lower=(
            active_metric_config.comparative_no_effect_lower
            if prepared.experiment_mode == "comparative"
            else None
        ),
        label_threshold_upper=(
            active_metric_config.comparative_no_effect_upper
            if prepared.experiment_mode == "comparative"
            else None
        ),
        weighted_cv_overall_auc=(
            None
            if aggregate.get("weighted_cv_overall_auc") is None
            else float(aggregate["weighted_cv_overall_auc"])
        ),
        weighted_cv_auc_class_neg1=(
            None
            if aggregate.get("weighted_cv_auc_class_neg1") is None
            else float(aggregate["weighted_cv_auc_class_neg1"])
        ),
        weighted_cv_auc_class_0=(
            None
            if aggregate.get("weighted_cv_auc_class_0") is None
            else float(aggregate["weighted_cv_auc_class_0"])
        ),
        weighted_cv_auc_class_pos1=(
            None
            if aggregate.get("weighted_cv_auc_class_pos1") is None
            else float(aggregate["weighted_cv_auc_class_pos1"])
        ),
        weighted_cv_auc_pos_vs_neg=(
            None
            if aggregate.get("weighted_cv_auc_pos_vs_neg") is None
            else float(aggregate["weighted_cv_auc_pos_vs_neg"])
        ),
        test_overall_auc=(
            None
            if holdout_metrics is None or holdout_metrics.overall_auc is None
            else float(holdout_metrics.overall_auc)
        ),
        test_auc_class_neg1=(
            None
            if holdout_metrics is None or holdout_metrics.auc_class_neg1 is None
            else float(holdout_metrics.auc_class_neg1)
        ),
        test_auc_class_0=(
            None
            if holdout_metrics is None or holdout_metrics.auc_class_0 is None
            else float(holdout_metrics.auc_class_0)
        ),
        test_auc_class_pos1=(
            None
            if holdout_metrics is None or holdout_metrics.auc_class_pos1 is None
            else float(holdout_metrics.auc_class_pos1)
        ),
        test_auc_pos_vs_neg=(
            None
            if holdout_metrics is None or holdout_metrics.auc_pos_vs_neg is None
            else float(holdout_metrics.auc_pos_vs_neg)
        ),
    )
    save_run_summary(
        summary,
        fold_results,
        architecture,
        run_dir=active_run_dir,
        latest_summary_path=latest_summary_path,
        dataset_config=dataset_config,
        split_config=split_config,
        training_config=training_config,
        metric_config=active_metric_config,
        constraints=constraints,
    )
    print_experiment_summary(summary)
    return summary
