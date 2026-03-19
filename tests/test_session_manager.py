from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

import session_manager
from autoresirch.session_manager.shared import orchestration as session_orchestration_module
from autoresirch.session_manager.shared import storage as session_storage_module
from autoresirch.prepare import DatasetConfig, PreparedDataset
from prepare import ExperimentSummary


def _session_prepared_dataset(*, raw_data_path: Path, experiment_mode: str) -> PreparedDataset:
    feature_name = "delta::feature_score" if experiment_mode == "comparative" else "feature_score"
    return PreparedDataset(
        flat_features=pd.DataFrame({feature_name: [1.0, 2.0]}),
        sequence_features=None,
        target=np.array([0.1, 0.2], dtype=np.float32),
        genes=np.array(["GENE1", "GENE2"], dtype=object),
        row_ids=np.array([0, 1], dtype=np.int64),
        numeric_feature_columns=(feature_name,),
        categorical_feature_columns=(),
        train_genes=("GENE1", "GENE2"),
        test_genes=(),
        cv_genes=("GENE1", "GENE2"),
        source_path=str(raw_data_path),
        experiment_mode=experiment_mode,
    )


@pytest.fixture()
def session_env(tmp_path: Path, monkeypatch):
    train_path = tmp_path / "train.py"
    program_path = tmp_path / "program.md"
    run_log_path = tmp_path / "run.log"
    train_path.write_text("BASELINE = 'baseline'\n", encoding="utf-8")
    program_path.write_text("# Program\n", encoding="utf-8")

    monkeypatch.setattr(session_manager, "SESSIONS_DIR", tmp_path / "sessions")
    monkeypatch.setattr(session_storage_module, "SESSIONS_DIR", tmp_path / "sessions")
    monkeypatch.setattr(session_manager, "EDITABLE_TRAIN_FILE", train_path)
    monkeypatch.setattr(session_storage_module, "EDITABLE_TRAIN_FILE", train_path)
    monkeypatch.setattr(session_orchestration_module, "EDITABLE_TRAIN_FILE", train_path)
    monkeypatch.setattr(session_manager, "PROGRAM_FILE", program_path)
    monkeypatch.setattr(session_storage_module, "PROGRAM_FILE", program_path)
    monkeypatch.setattr(session_orchestration_module, "PROGRAM_FILE", program_path)
    monkeypatch.setattr(session_manager, "RUN_LOG", run_log_path)
    monkeypatch.setattr(session_orchestration_module, "RUN_LOG", run_log_path)
    monkeypatch.setattr(session_storage_module, "_collect_git_metadata", lambda: ("abc123", "main", False))
    monkeypatch.setattr(
        session_storage_module,
        "prepare_dataset",
        lambda *, dataset_config, artifact_root=None, **_kwargs: _session_prepared_dataset(
            raw_data_path=Path(dataset_config.raw_data_path),
            experiment_mode=dataset_config.experiment_mode,
        ),
    )
    monkeypatch.setattr(
        session_storage_module,
        "_load_architecture_metadata",
        lambda: (
            "arch123",
            {
                "family": "mlp",
                "hidden_dims": [16],
                "activation": "silu",
                "dropout": 0.0,
                "normalization": "none",
                "use_bias": True,
            },
        ),
    )
    monkeypatch.setattr(session_orchestration_module, "_session_manager_package", lambda: session_manager)
    monkeypatch.setattr(session_storage_module, "_session_manager_package", lambda: session_manager)
    return {
        "tmp_path": tmp_path,
        "train_path": train_path,
        "program_path": program_path,
        "run_log_path": run_log_path,
    }


def _write_fake_summary(
    run_dir: Path,
    metric_value: float,
    *,
    experiment_mode: str = "standard",
    primary_metric_name: str | None = None,
    weighted_cv_rmse_mean: float = 0.31,
    cv_rmse_std: float = 0.02,
    weighted_cv_pearson_r_mean: float | None = 0.2,
    weighted_cv_spearman_r_mean: float | None = 0.25,
    weighted_cv_overall_auc: float | None = None,
    weighted_cv_auc_class_neg1: float | None = None,
    weighted_cv_auc_class_0: float | None = None,
    weighted_cv_auc_class_pos1: float | None = None,
    weighted_cv_auc_pos_vs_neg: float | None = None,
    num_params: int = 123,
    train_seconds: float = 1.5,
    diagnostics: dict[str, Any] | None = None,
    architecture: dict[str, Any] | None = None,
) -> ExperimentSummary:
    resolved_primary_metric_name = primary_metric_name
    if resolved_primary_metric_name is None:
        resolved_primary_metric_name = (
            "weighted_cv_overall_auc" if experiment_mode == "comparative" else "weighted_cv_auc"
        )
    if experiment_mode == "comparative" and weighted_cv_overall_auc is None:
        weighted_cv_overall_auc = metric_value
    summary = ExperimentSummary(
        primary_metric_name=resolved_primary_metric_name,
        primary_metric_value=metric_value,
        metric_direction="higher_is_better",
        improvement_epsilon=1e-4,
        weighted_cv_rmse_mean=weighted_cv_rmse_mean,
        cv_rmse_mean=weighted_cv_rmse_mean,
        cv_rmse_std=cv_rmse_std,
        weighted_cv_mae_mean=0.22,
        weighted_cv_r2_mean=0.1,
        weighted_cv_auc_mean=metric_value if experiment_mode == "standard" else None,
        weighted_cv_pearson_r_mean=weighted_cv_pearson_r_mean,
        weighted_cv_spearman_r_mean=weighted_cv_spearman_r_mean,
        pooled_cv_rmse=weighted_cv_rmse_mean,
        test_rmse=None,
        test_mae=None,
        test_r2=None,
        test_auc=None,
        test_pearson_r=None,
        test_spearman_r=None,
        num_params=num_params,
        train_seconds=train_seconds,
        feature_dim=3,
        num_rows=6,
        cv_folds=2,
        train_genes=("GENE1", "GENE2"),
        test_genes=(),
        cv_genes=("GENE1", "GENE2"),
        run_dir=str(run_dir),
        experiment_mode=experiment_mode,
        label_threshold_lower=-0.2 if experiment_mode == "comparative" else None,
        label_threshold_upper=0.2 if experiment_mode == "comparative" else None,
        weighted_cv_overall_auc=weighted_cv_overall_auc,
        weighted_cv_auc_class_neg1=weighted_cv_auc_class_neg1,
        weighted_cv_auc_class_0=weighted_cv_auc_class_0,
        weighted_cv_auc_class_pos1=weighted_cv_auc_class_pos1,
        weighted_cv_auc_pos_vs_neg=weighted_cv_auc_pos_vs_neg,
    )
    architecture_payload = {
        "family": "mlp",
        "hidden_dims": [16],
        "activation": "silu",
        "dropout": 0.0,
        "normalization": "none",
        "use_bias": True,
    }
    if architecture is not None:
        architecture_payload = architecture

    payload = {
        "summary": asdict(summary),
        "architecture": architecture_payload,
        "diagnostics": {
            "fold_count": 2,
            "nan_metric_counts": {"auc": 0, "pearson_r": 0, "spearman_r": 0},
            "best_auc_fold": {"gene": "GENE1", "count": 3, "auc": metric_value},
            "worst_auc_fold": {"gene": "GENE2", "count": 3, "auc": metric_value - 0.05},
            "best_rmse_fold": {"gene": "GENE1", "count": 3, "rmse": 0.2},
            "worst_rmse_fold": {"gene": "GENE2", "count": 3, "rmse": 0.4},
        },
        "fold_results": [],
        "constraints": {},
        "training_config": {},
        "dataset_config": {},
        "split_config": {},
        "metric_config": {},
    }
    if diagnostics is not None:
        payload["diagnostics"] = diagnostics
    run_dir.joinpath("summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def _fake_run_experiment_factory(outcomes: list[float | Exception | dict[str, Any]]):
    outcome_iter = iter(outcomes)

    def fake_run_experiment(*, run_dir=None, latest_summary_path=None, **kwargs):
        outcome = next(outcome_iter)
        if isinstance(outcome, Exception):
            raise outcome
        assert run_dir is not None
        run_dir.mkdir(parents=True, exist_ok=True)
        if isinstance(outcome, dict):
            outcome_payload = dict(outcome)
            metric_value = float(outcome_payload.pop("metric_value"))
            return _write_fake_summary(run_dir, metric_value, **outcome_payload)
        return _write_fake_summary(run_dir, outcome)

    return fake_run_experiment


def _start_session(
    session_id: str,
    *,
    experiment_mode: str = "standard",
    objective: str | None = None,
    raw_data_path: Path | None = None,
) -> None:
    argv = [
        "start",
        "--session-id",
        session_id,
        "--initiated-by",
        "agent",
        "--experiment-mode",
        experiment_mode,
    ]
    if raw_data_path is not None:
        argv.extend(["--raw-data-path", str(raw_data_path)])
    if objective is not None:
        argv.extend(["--objective", objective])
    assert session_manager.main(argv) == 0


def _run_base(session_id: str) -> int:
    return session_manager.main(
        [
            "run",
            "--session-id",
            session_id,
            "--run-role",
            "base",
            "--hypothesis",
            "Establish the baseline.",
            "--mutation-summary",
            "Unmodified baseline.",
            "--description",
            "Base run",
        ]
    )


def _run_candidate(
    session_id: str,
    *,
    hypothesis: str = "Try a candidate.",
    mutation_summary: str = "Candidate mutation.",
    description: str = "Candidate run",
) -> int:
    return session_manager.main(
        [
            "run",
            "--session-id",
            session_id,
            "--run-role",
            "candidate",
            "--hypothesis",
            hypothesis,
            "--mutation-summary",
            mutation_summary,
            "--description",
            description,
        ]
    )


def test_start_creates_session_artifacts(session_env) -> None:
    session_id = "session-001"
    _start_session(session_id)

    session_dir = session_manager.SESSIONS_DIR / session_id
    assert session_dir.exists()
    assert session_dir.joinpath("results.tsv").exists()
    assert session_dir.joinpath("session_context.json").exists()
    assert session_dir.joinpath("session_state.json").exists()


def test_start_persists_comparative_experiment_mode(session_env) -> None:
    session_id = "session-mode-001"
    _start_session(session_id, experiment_mode="comparative")

    context = session_manager.load_session_context(session_id)

    assert context.experiment_mode == "comparative"
    assert context.objective == "Maximize weighted_cv_overall_auc"
    assert context.feature_names == ("delta::feature_score",)


def test_start_persists_raw_data_path_and_session_prepared_cache(session_env, monkeypatch) -> None:
    session_id = "session-raw-data-001"
    raw_data_path = session_env["tmp_path"] / "comparative.csv"
    prepared_cache_roots: list[Path | None] = []

    def fake_prepare_dataset(*, dataset_config, artifact_root=None, **_kwargs):
        prepared_cache_roots.append(artifact_root)
        if artifact_root is not None:
            artifact_root.mkdir(parents=True, exist_ok=True)
            artifact_root.joinpath("prepared_dataset_comparative.pkl").write_bytes(b"prepared")
            artifact_root.joinpath("prepared_dataset_metadata_comparative.json").write_text(
                "{}",
                encoding="utf-8",
            )
        return _session_prepared_dataset(
            raw_data_path=Path(dataset_config.raw_data_path),
            experiment_mode=dataset_config.experiment_mode,
        )

    monkeypatch.setattr(session_storage_module, "prepare_dataset", fake_prepare_dataset)

    _start_session(
        session_id,
        experiment_mode="comparative",
        raw_data_path=raw_data_path,
    )

    context = session_manager.load_session_context(session_id)

    assert context.raw_data_path == raw_data_path
    assert context.feature_names == ("delta::feature_score",)
    assert Path(context.dataset_config_payload["raw_data_path"]) == raw_data_path
    assert context.dataset_config_payload["experiment_mode"] == "comparative"
    assert context.prepared_dataset_cache_dir == session_manager.SESSIONS_DIR / session_id / "prepared_data"
    assert prepared_cache_roots == [context.prepared_dataset_cache_dir]
    assert context.prepared_dataset_cache_dir.joinpath("prepared_dataset_comparative.pkl").exists()


def test_base_run_creates_expected_artifacts(session_env, monkeypatch) -> None:
    session_id = "session-002"
    _start_session(session_id)
    monkeypatch.setattr(session_manager, "run_experiment", _fake_run_experiment_factory([0.65]))

    assert _run_base(session_id) == 0

    state = session_manager.load_session_state(session_id)
    run_id = state.ordered_run_ids[0]
    run_dir = Path(state.run_dirs[run_id])
    assert run_dir.joinpath("train_snapshot.py").exists()
    assert run_dir.joinpath("run_context.json").exists()
    assert run_dir.joinpath("summary.json").exists()
    assert run_dir.joinpath("decision.json").exists()
    assert run_dir.joinpath("analysis_input.json").exists()
    assert run_dir.joinpath("synopsis.md").exists()
    results_lines = session_manager._session_results_path(session_id).read_text(encoding="utf-8").strip().splitlines()
    assert len(results_lines) == 2


def test_candidate_run_writes_analysis_input_metric_bundle(session_env, monkeypatch) -> None:
    session_id = "session-analysis-input-001"
    _start_session(session_id)
    monkeypatch.setattr(
        session_manager,
        "run_experiment",
        _fake_run_experiment_factory(
            [
                {"metric_value": 0.65, "weighted_cv_rmse_mean": 0.31, "weighted_cv_pearson_r_mean": 0.20, "weighted_cv_spearman_r_mean": 0.25},
                {"metric_value": 0.67, "weighted_cv_rmse_mean": 0.30, "weighted_cv_pearson_r_mean": 0.23, "weighted_cv_spearman_r_mean": 0.27},
            ]
        ),
    )

    assert _run_base(session_id) == 0
    assert _run_candidate(
        session_id,
        hypothesis="Try a broader-based improvement.",
        mutation_summary="Candidate analysis-input mutation.",
        description="Candidate analysis-input run",
    ) == 0

    state = session_manager.load_session_state(session_id)
    run_dir = Path(state.run_dirs[state.ordered_run_ids[-1]])
    analysis_input = json.loads(run_dir.joinpath("analysis_input.json").read_text(encoding="utf-8"))

    assert analysis_input["analysis_mode"] == "metric_comparison"
    assert analysis_input["decision_status"] == "keep"
    assert analysis_input["metrics"]["weighted_cv_auc"]["delta_vs_compared"] == pytest.approx(0.02)
    assert analysis_input["metrics"]["weighted_cv_rmse_mean"]["compared_label"] == "better"
    assert analysis_input["rule_based_interpretation"]
    assert analysis_input["analysis_constraints"]["do_not_override_decision_rule"] is True


def test_analyze_run_writes_agent_analysis_and_refreshes_synopsis(
    session_env,
    monkeypatch,
) -> None:
    session_id = "session-agent-analysis-001"
    _start_session(session_id)
    monkeypatch.setattr(session_manager, "run_experiment", _fake_run_experiment_factory([0.65]))

    assert _run_base(session_id) == 0
    state = session_manager.load_session_state(session_id)
    run_id = state.ordered_run_ids[0]
    run_dir = Path(state.run_dirs[run_id])

    assert session_manager.main(
        [
            "analyze-run",
            "--session-id",
            session_id,
            "--run-id",
            run_id,
            "--summary-label",
            "baseline reference",
            "--freeform-analysis",
            "This baseline establishes the first trustworthy reference point and shows stable secondary metrics, so future mutations should stay local and easy to attribute.",
            "--likely-helped",
            "balanced starting width",
            "--likely-hurt",
            "limited depth capacity",
            "--confidence",
            "0.72",
            "--next-step-reasoning",
            "Test one nearby width or dropout change so the next run stays attributable to a single architectural axis.",
        ]
    ) == 0

    agent_analysis = json.loads(run_dir.joinpath("agent_analysis.json").read_text(encoding="utf-8"))
    synopsis = run_dir.joinpath("synopsis.md").read_text(encoding="utf-8")

    assert agent_analysis["summary_label"] == "baseline reference"
    assert agent_analysis["analysis_mode"] == "metric_comparison"
    assert "### Agent Analysis" in synopsis
    assert "Summary label: `baseline reference`" in synopsis
    assert "future mutations should stay local and easy to attribute." in synopsis


def test_record_agent_analysis_validates_word_limits(session_env, monkeypatch) -> None:
    session_id = "session-agent-analysis-002"
    _start_session(session_id)
    monkeypatch.setattr(session_manager, "run_experiment", _fake_run_experiment_factory([0.65]))

    assert _run_base(session_id) == 0
    state = session_manager.load_session_state(session_id)
    run_id = state.ordered_run_ids[0]

    with pytest.raises(ValueError, match="freeform_analysis"):
        session_manager.record_agent_analysis(
            session_id=session_id,
            run_id=run_id,
            summary_label="too short",
            freeform_analysis="Too short.",
            likely_helped=["balanced starting width"],
            likely_hurt=[],
            confidence=0.5,
            next_step_reasoning="Try one nearby width change next.",
        )


def test_mixed_signal_synopsis_marks_partial_support_and_base_context(
    session_env,
    monkeypatch,
) -> None:
    session_id = "session-interpret-001"
    _start_session(session_id)
    monkeypatch.setattr(
        session_manager,
        "run_experiment",
        _fake_run_experiment_factory(
            [
                {"metric_value": 0.65, "weighted_cv_rmse_mean": 0.31, "weighted_cv_pearson_r_mean": 0.20, "weighted_cv_spearman_r_mean": 0.25},
                {"metric_value": 0.70, "weighted_cv_rmse_mean": 0.30, "weighted_cv_pearson_r_mean": 0.21, "weighted_cv_spearman_r_mean": 0.26},
                {"metric_value": 0.6985, "weighted_cv_rmse_mean": 0.295, "weighted_cv_pearson_r_mean": 0.24, "weighted_cv_spearman_r_mean": 0.29},
            ]
        ),
    )

    assert _run_base(session_id) == 0
    assert _run_candidate(
        session_id,
        hypothesis="Try the first improving candidate.",
        mutation_summary="Keep candidate mutation.",
        description="Keep candidate",
    ) == 0
    assert _run_candidate(
        session_id,
        hypothesis="Probe a mixed-signal mutation.",
        mutation_summary="Mixed-signal candidate mutation.",
        description="Mixed-signal candidate",
    ) == 0

    state = session_manager.load_session_state(session_id)
    candidate_run_dir = Path(state.run_dirs[state.ordered_run_ids[-1]])
    decision_payload = json.loads(candidate_run_dir.joinpath("decision.json").read_text(encoding="utf-8"))
    synopsis = candidate_run_dir.joinpath("synopsis.md").read_text(encoding="utf-8")

    assert decision_payload["hypothesis_result"] == "partially_supported"
    assert "Metric tradeoff: this was a `near miss` and a `mixed signal`;" in synopsis
    assert "Hypothesis assessment: `partially_supported`" in synopsis
    assert "Session context: despite losing to the compared run, this candidate still outperformed the session base on `weighted_cv_auc`." in synopsis
    assert "| vs base " in synopsis


def test_keep_synopsis_calls_out_threshold_separation_tradeoff(
    session_env,
    monkeypatch,
) -> None:
    session_id = "session-interpret-002"
    _start_session(session_id)
    monkeypatch.setattr(
        session_manager,
        "run_experiment",
        _fake_run_experiment_factory(
            [
                {"metric_value": 0.65, "weighted_cv_rmse_mean": 0.31, "weighted_cv_pearson_r_mean": 0.20, "weighted_cv_spearman_r_mean": 0.25},
                {"metric_value": 0.661, "weighted_cv_rmse_mean": 0.34, "weighted_cv_pearson_r_mean": 0.19, "weighted_cv_spearman_r_mean": 0.24},
            ]
        ),
    )

    assert _run_base(session_id) == 0
    assert _run_candidate(
        session_id,
        hypothesis="Try a brittle AUC win.",
        mutation_summary="Brittle keep candidate mutation.",
        description="Brittle keep candidate",
    ) == 0

    state = session_manager.load_session_state(session_id)
    kept_run_dir = Path(state.run_dirs[state.ordered_run_ids[-1]])
    decision_payload = json.loads(kept_run_dir.joinpath("decision.json").read_text(encoding="utf-8"))
    synopsis = kept_run_dir.joinpath("synopsis.md").read_text(encoding="utf-8")

    assert decision_payload["hypothesis_result"] == "supported"
    assert "Metric tradeoff: this was a `brittle gain`; `weighted_cv_auc` improved while `weighted_cv_rmse_mean` worsened." in synopsis
    assert "Metric tradeoff: this looks like a `threshold-separation gain` rather than a broad regression improvement." in synopsis


def test_synopsis_flags_robustness_and_metric_coverage_regressions(
    session_env,
    monkeypatch,
) -> None:
    session_id = "session-interpret-003"
    _start_session(session_id)
    monkeypatch.setattr(
        session_manager,
        "run_experiment",
        _fake_run_experiment_factory(
            [
                {
                    "metric_value": 0.65,
                    "weighted_cv_rmse_mean": 0.31,
                    "cv_rmse_std": 0.02,
                    "diagnostics": {
                        "fold_count": 2,
                        "nan_metric_counts": {"auc": 0, "pearson_r": 0, "spearman_r": 0},
                        "best_auc_fold": {"gene": "GENE1", "count": 3, "auc": 0.70},
                        "worst_auc_fold": {"gene": "GENE2", "count": 3, "auc": 0.68},
                        "best_rmse_fold": {"gene": "GENE1", "count": 3, "rmse": 0.20},
                        "worst_rmse_fold": {"gene": "GENE2", "count": 3, "rmse": 0.23},
                    },
                },
                {
                    "metric_value": 0.64,
                    "weighted_cv_rmse_mean": 0.32,
                    "cv_rmse_std": 0.05,
                    "diagnostics": {
                        "fold_count": 2,
                        "nan_metric_counts": {"auc": 2, "pearson_r": 1, "spearman_r": 1},
                        "best_auc_fold": {"gene": "GENE1", "count": 3, "auc": 0.95},
                        "worst_auc_fold": {"gene": "GENE2", "count": 3, "auc": 0.10},
                        "best_rmse_fold": {"gene": "GENE1", "count": 3, "rmse": 0.10},
                        "worst_rmse_fold": {"gene": "GENE2", "count": 3, "rmse": 0.45},
                    },
                },
            ]
        ),
    )

    assert _run_base(session_id) == 0
    assert _run_candidate(
        session_id,
        hypothesis="Trigger robustness regressions.",
        mutation_summary="Robustness regression candidate mutation.",
        description="Robustness regression candidate",
    ) == 0

    state = session_manager.load_session_state(session_id)
    candidate_run_dir = Path(state.run_dirs[state.ordered_run_ids[-1]])
    synopsis = candidate_run_dir.joinpath("synopsis.md").read_text(encoding="utf-8")

    assert "metric coverage became less informative." in synopsis
    assert "fold variability worsened relative to the compared run." in synopsis


def test_kept_run_updates_incumbent(session_env, monkeypatch) -> None:
    session_id = "session-003"
    _start_session(session_id)
    monkeypatch.setattr(session_manager, "run_experiment", _fake_run_experiment_factory([0.65, 0.70]))

    assert _run_base(session_id) == 0
    session_env["train_path"].write_text("BASELINE = 'candidate-keep'\n", encoding="utf-8")
    assert session_manager.main(
        [
            "run",
            "--session-id",
            session_id,
            "--run-role",
            "candidate",
            "--hypothesis",
            "Increase the primary metric.",
            "--mutation-summary",
            "Candidate keep mutation.",
            "--description",
            "Keep candidate",
        ]
    ) == 0

    state = session_manager.load_session_state(session_id)
    assert state.keep_count == 2
    assert state.incumbent_run_id == state.ordered_run_ids[-1]
    assert state.run_statuses[state.incumbent_run_id] == "keep"


def test_discarded_run_does_not_advance_incumbent_and_sync_restores_train(
    session_env,
    monkeypatch,
) -> None:
    session_id = "session-004"
    _start_session(session_id)
    monkeypatch.setattr(session_manager, "run_experiment", _fake_run_experiment_factory([0.65, 0.60]))

    baseline_source = session_env["train_path"].read_text(encoding="utf-8")
    assert _run_base(session_id) == 0
    session_env["train_path"].write_text("BASELINE = 'candidate-discard'\n", encoding="utf-8")
    assert session_manager.main(
        [
            "run",
            "--session-id",
            session_id,
            "--run-role",
            "candidate",
            "--hypothesis",
            "Try a weaker candidate.",
            "--mutation-summary",
            "Candidate discard mutation.",
            "--description",
            "Discard candidate",
        ]
    ) == 0

    state = session_manager.load_session_state(session_id)
    assert state.keep_count == 1
    assert state.discard_count == 1
    assert state.incumbent_run_id == state.ordered_run_ids[0]
    assert session_manager.main(["sync-incumbent", "--session-id", session_id]) == 0
    assert session_env["train_path"].read_text(encoding="utf-8") == baseline_source


def test_within_epsilon_primary_delta_uses_pearson_tiebreak_to_keep(
    session_env,
    monkeypatch,
) -> None:
    session_id = "session-pearson-keep"
    _start_session(session_id)
    monkeypatch.setattr(
        session_manager,
        "run_experiment",
        _fake_run_experiment_factory(
            [
                {"metric_value": 0.65, "weighted_cv_pearson_r_mean": 0.20},
                {"metric_value": 0.64995, "weighted_cv_pearson_r_mean": 0.25},
            ]
        ),
    )

    assert _run_base(session_id) == 0
    assert _run_candidate(
        session_id,
        hypothesis="Try a near-tied candidate with stronger correlation.",
        mutation_summary="Near-epsilon Pearson gain.",
        description="Pearson tiebreak keep",
    ) == 0

    state = session_manager.load_session_state(session_id)
    kept_run_id = state.ordered_run_ids[-1]
    kept_run_dir = Path(state.run_dirs[kept_run_id])
    decision_payload = json.loads(kept_run_dir.joinpath("decision.json").read_text(encoding="utf-8"))

    assert state.incumbent_run_id == kept_run_id
    assert decision_payload["decision_status"] == "keep"
    assert "broke the tie" in decision_payload["decision_reason"]


def test_within_epsilon_primary_delta_without_pearson_gain_discards(
    session_env,
    monkeypatch,
) -> None:
    session_id = "session-pearson-discard"
    _start_session(session_id)
    monkeypatch.setattr(
        session_manager,
        "run_experiment",
        _fake_run_experiment_factory(
            [
                {"metric_value": 0.65, "weighted_cv_pearson_r_mean": 0.20},
                {"metric_value": 0.65005, "weighted_cv_pearson_r_mean": 0.19},
            ]
        ),
    )

    assert _run_base(session_id) == 0
    assert _run_candidate(
        session_id,
        hypothesis="Try a near-tied candidate with weaker correlation.",
        mutation_summary="Near-epsilon Pearson loss.",
        description="Pearson tiebreak discard",
    ) == 0

    state = session_manager.load_session_state(session_id)
    discarded_run_id = state.ordered_run_ids[-1]
    discarded_run_dir = Path(state.run_dirs[discarded_run_id])
    decision_payload = json.loads(discarded_run_dir.joinpath("decision.json").read_text(encoding="utf-8"))

    assert state.incumbent_run_id == state.ordered_run_ids[0]
    assert decision_payload["decision_status"] == "discard"
    assert "did not improve over the incumbent" in decision_payload["decision_reason"]


def test_crashed_run_records_failure(session_env, monkeypatch) -> None:
    session_id = "session-005"
    _start_session(session_id)
    monkeypatch.setattr(session_manager, "run_experiment", _fake_run_experiment_factory([0.65, RuntimeError("boom")]))

    assert _run_base(session_id) == 0
    session_env["train_path"].write_text("BASELINE = 'candidate-crash'\n", encoding="utf-8")
    assert session_manager.main(
        [
            "run",
            "--session-id",
            session_id,
            "--run-role",
            "candidate",
            "--hypothesis",
            "Trigger a crash.",
            "--mutation-summary",
            "Crash candidate mutation.",
            "--description",
            "Crash candidate",
        ]
    ) == 0

    state = session_manager.load_session_state(session_id)
    crashed_run_id = state.ordered_run_ids[-1]
    crashed_run_dir = Path(state.run_dirs[crashed_run_id])
    assert state.crash_count == 1
    assert state.run_statuses[crashed_run_id] == "crash"
    assert crashed_run_dir.joinpath("failure.json").exists()
    analysis_input = json.loads(crashed_run_dir.joinpath("analysis_input.json").read_text(encoding="utf-8"))
    crash_synopsis = crashed_run_dir.joinpath("synopsis.md").read_text(encoding="utf-8")
    assert analysis_input["analysis_mode"] == "failure_review"
    assert analysis_input["failure"]["error_type"] == "RuntimeError"
    assert "This run crashed before a summary was written." in crash_synopsis
    assert "Metric tradeoff:" not in crash_synopsis
    assert "Hypothesis assessment:" not in crash_synopsis
    results_lines = session_manager._session_results_path(session_id).read_text(encoding="utf-8").strip().splitlines()
    assert len(results_lines) == 3


def test_finalize_writes_session_summary(session_env, monkeypatch) -> None:
    session_id = "session-006"
    _start_session(session_id)
    monkeypatch.setattr(session_manager, "run_experiment", _fake_run_experiment_factory([0.65]))

    assert _run_base(session_id) == 0
    assert session_manager.main(
        [
            "finalize",
            "--session-id",
            session_id,
            "--status",
            "completed",
            "--end-reason",
            "Reached stopping condition.",
        ]
    ) == 0

    state = session_manager.load_session_state(session_id)
    assert state.status == "completed"
    assert session_manager._session_summary_json_path(session_id).exists()
    assert session_manager._session_summary_md_path(session_id).exists()


def test_finalize_handles_hybrid_runs_with_empty_hidden_dims(session_env, monkeypatch) -> None:
    session_id = "session-007"
    _start_session(session_id)
    hybrid_architecture = {
        "family": "hybrid_cnn_mlp",
        "hidden_dims": [],
        "flat_hidden_dims": [32],
        "fusion_hidden_dims": [16],
        "activation": "relu",
        "dropout": 0.1,
        "normalization": "layernorm",
        "use_bias": True,
    }
    monkeypatch.setattr(
        session_manager,
        "run_experiment",
        _fake_run_experiment_factory(
            [
                {"metric_value": 0.65, "architecture": hybrid_architecture},
                {"metric_value": 0.66, "architecture": hybrid_architecture},
            ]
        ),
    )

    assert _run_base(session_id) == 0
    assert _run_candidate(session_id) == 0
    assert (
        session_manager.main(
            [
                "finalize",
                "--session-id",
                session_id,
                "--status",
                "completed",
                "--end-reason",
                "Hybrid search finished.",
            ]
        )
        == 0
    )

    summary_payload = json.loads(
        session_manager._session_summary_json_path(session_id).read_text(encoding="utf-8")
    )
    coverage = summary_payload["search_space_coverage"]
    assert coverage["families_tried"] == ["hybrid_cnn_mlp"]
    assert coverage["depth_values_tried"] == [0]
    assert coverage["max_width_values_tried"] == []


def test_candidate_run_comparative_analysis_includes_class_support(session_env, monkeypatch) -> None:
    session_id = "session-comparative-001"
    _start_session(session_id, experiment_mode="comparative")
    monkeypatch.setattr(
        session_manager,
        "run_experiment",
        _fake_run_experiment_factory(
            [
                {
                    "metric_value": 0.70,
                    "experiment_mode": "comparative",
                    "weighted_cv_overall_auc": 0.70,
                    "weighted_cv_auc_class_neg1": 0.72,
                    "weighted_cv_auc_class_0": 0.68,
                    "weighted_cv_auc_class_pos1": 0.71,
                    "weighted_cv_auc_pos_vs_neg": 0.74,
                    "diagnostics": {
                        "fold_count": 2,
                        "nan_metric_counts": {"auc": 0, "pearson_r": 0, "spearman_r": 0},
                        "undefined_metric_counts": {
                            "overall_auc": 0,
                            "auc_class_neg1": 0,
                            "auc_class_0": 1,
                            "auc_class_pos1": 0,
                            "auc_pos_vs_neg": 0,
                        },
                        "best_auc_fold": {"gene": "GENE1", "count": 3, "overall_auc": 0.74},
                        "worst_auc_fold": {"gene": "GENE2", "count": 3, "overall_auc": 0.66},
                        "best_rmse_fold": {"gene": "GENE1", "count": 3, "rmse": 0.20},
                        "worst_rmse_fold": {"gene": "GENE2", "count": 3, "rmse": 0.32},
                        "class_support": {"folds_missing_neg1": 0, "folds_missing_0": 1, "folds_missing_pos1": 0},
                    },
                },
                {
                    "metric_value": 0.73,
                    "experiment_mode": "comparative",
                    "weighted_cv_overall_auc": 0.73,
                    "weighted_cv_auc_class_neg1": 0.75,
                    "weighted_cv_auc_class_0": 0.69,
                    "weighted_cv_auc_class_pos1": 0.76,
                    "weighted_cv_auc_pos_vs_neg": 0.80,
                    "diagnostics": {
                        "fold_count": 2,
                        "nan_metric_counts": {"auc": 0, "pearson_r": 0, "spearman_r": 0},
                        "undefined_metric_counts": {
                            "overall_auc": 0,
                            "auc_class_neg1": 0,
                            "auc_class_0": 1,
                            "auc_class_pos1": 0,
                            "auc_pos_vs_neg": 0,
                        },
                        "best_auc_fold": {"gene": "GENE1", "count": 3, "overall_auc": 0.77},
                        "worst_auc_fold": {"gene": "GENE2", "count": 3, "overall_auc": 0.68},
                        "best_rmse_fold": {"gene": "GENE1", "count": 3, "rmse": 0.18},
                        "worst_rmse_fold": {"gene": "GENE2", "count": 3, "rmse": 0.30},
                        "class_support": {"folds_missing_neg1": 0, "folds_missing_0": 1, "folds_missing_pos1": 0},
                    },
                },
            ]
        ),
    )

    assert _run_base(session_id) == 0
    assert _run_candidate(
        session_id,
        hypothesis="Improve comparative AUC.",
        mutation_summary="Comparative candidate mutation.",
        description="Comparative candidate run",
    ) == 0

    state = session_manager.load_session_state(session_id)
    run_dir = Path(state.run_dirs[state.ordered_run_ids[-1]])
    analysis_input = json.loads(run_dir.joinpath("analysis_input.json").read_text(encoding="utf-8"))
    synopsis = run_dir.joinpath("synopsis.md").read_text(encoding="utf-8")

    assert analysis_input["metrics"]["weighted_cv_overall_auc"]["delta_vs_compared"] == pytest.approx(0.03)
    assert analysis_input["metrics"]["weighted_cv_auc_class_0"]["current_value"] == pytest.approx(0.69)
    assert analysis_input["diagnostics"]["undefined_metric_counts"]["auc_class_0"] == 1
    assert analysis_input["diagnostics"]["class_support"]["folds_missing_0"] == 1
    assert "Comparative class support" in synopsis


def test_comparative_session_passes_mode_specific_dataset_config(session_env, monkeypatch) -> None:
    session_id = "session-comparative-002"
    raw_data_path = session_env["tmp_path"] / "comparative.csv"
    _start_session(session_id, experiment_mode="comparative", raw_data_path=raw_data_path)
    captured_dataset_configs: list[DatasetConfig] = []
    captured_cache_dirs: list[Path] = []

    def fake_run_experiment(
        *,
        dataset_config=None,
        prepared_dataset_cache_dir=None,
        run_dir=None,
        latest_summary_path=None,
        **kwargs,
    ):
        assert dataset_config is not None
        captured_dataset_configs.append(dataset_config)
        assert prepared_dataset_cache_dir is not None
        captured_cache_dirs.append(prepared_dataset_cache_dir)
        return _fake_run_experiment_factory(
            [
                {
                    "metric_value": 0.70,
                    "experiment_mode": "comparative",
                    "weighted_cv_overall_auc": 0.70,
                }
            ]
        )(dataset_config=dataset_config, run_dir=run_dir, latest_summary_path=latest_summary_path, **kwargs)

    monkeypatch.setattr(session_manager, "run_experiment", fake_run_experiment)

    assert _run_base(session_id) == 0

    assert captured_dataset_configs
    assert captured_dataset_configs[0].experiment_mode == "comparative"
    assert captured_dataset_configs[0].raw_data_path == raw_data_path
    assert captured_cache_dirs == [session_manager.SESSIONS_DIR / session_id / "prepared_data"]
