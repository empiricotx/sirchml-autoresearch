from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pytest

import session_manager
from prepare import ExperimentSummary


@pytest.fixture()
def session_env(tmp_path: Path, monkeypatch):
    train_path = tmp_path / "train.py"
    program_path = tmp_path / "program.md"
    run_log_path = tmp_path / "run.log"
    train_path.write_text("BASELINE = 'baseline'\n", encoding="utf-8")
    program_path.write_text("# Program\n", encoding="utf-8")

    monkeypatch.setattr(session_manager, "SESSIONS_DIR", tmp_path / "sessions")
    monkeypatch.setattr(session_manager, "EDITABLE_TRAIN_FILE", train_path)
    monkeypatch.setattr(session_manager, "PROGRAM_FILE", program_path)
    monkeypatch.setattr(session_manager, "RUN_LOG", run_log_path)
    monkeypatch.setattr(session_manager, "_collect_git_metadata", lambda: ("abc123", "main", False))
    monkeypatch.setattr(
        session_manager,
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
    return {
        "tmp_path": tmp_path,
        "train_path": train_path,
        "program_path": program_path,
        "run_log_path": run_log_path,
    }


def _write_fake_summary(run_dir: Path, metric_value: float, *, num_params: int = 123) -> ExperimentSummary:
    summary = ExperimentSummary(
        primary_metric_name="weighted_cv_auc",
        primary_metric_value=metric_value,
        metric_direction="higher_is_better",
        improvement_epsilon=1e-4,
        weighted_cv_rmse_mean=0.31,
        cv_rmse_mean=0.31,
        cv_rmse_std=0.02,
        weighted_cv_mae_mean=0.22,
        weighted_cv_r2_mean=0.1,
        weighted_cv_auc_mean=metric_value,
        weighted_cv_pearson_r_mean=0.2,
        weighted_cv_spearman_r_mean=0.25,
        pooled_cv_rmse=0.30,
        test_rmse=None,
        test_mae=None,
        test_r2=None,
        test_auc=None,
        test_pearson_r=None,
        test_spearman_r=None,
        num_params=num_params,
        train_seconds=1.5,
        feature_dim=3,
        num_rows=6,
        cv_folds=2,
        train_genes=("GENE1", "GENE2"),
        test_genes=(),
        cv_genes=("GENE1", "GENE2"),
        run_dir=str(run_dir),
    )
    payload = {
        "summary": asdict(summary),
        "architecture": {
            "family": "mlp",
            "hidden_dims": [16],
            "activation": "silu",
            "dropout": 0.0,
            "normalization": "none",
            "use_bias": True,
        },
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
    run_dir.joinpath("summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def _fake_run_experiment_factory(outcomes: list[float | Exception]):
    outcome_iter = iter(outcomes)

    def fake_run_experiment(*, run_dir=None, latest_summary_path=None, **kwargs):
        outcome = next(outcome_iter)
        if isinstance(outcome, Exception):
            raise outcome
        assert run_dir is not None
        run_dir.mkdir(parents=True, exist_ok=True)
        return _write_fake_summary(run_dir, outcome)

    return fake_run_experiment


def _start_session(session_id: str) -> None:
    assert session_manager.main(
        [
            "start",
            "--session-id",
            session_id,
            "--objective",
            "Maximize weighted_cv_auc",
            "--initiated-by",
            "agent",
        ]
    ) == 0


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


def test_start_creates_session_artifacts(session_env) -> None:
    session_id = "session-001"
    _start_session(session_id)

    session_dir = session_manager.SESSIONS_DIR / session_id
    assert session_dir.exists()
    assert session_dir.joinpath("results.tsv").exists()
    assert session_dir.joinpath("session_context.json").exists()
    assert session_dir.joinpath("session_state.json").exists()


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
    assert run_dir.joinpath("synopsis.md").exists()
    results_lines = session_manager._session_results_path(session_id).read_text(encoding="utf-8").strip().splitlines()
    assert len(results_lines) == 2


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
