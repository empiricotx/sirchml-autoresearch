from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

from autoresirch.prepare import DATASET_CONFIG, ExperimentMode, METRIC_CONFIG
from autoresirch.prepare.shared.utils import resolve_primary_metric_name
from autoresirch.session_manager.shared.analysis import record_agent_analysis
from autoresirch.session_manager.shared.orchestration import (
    create_session,
    finalize_session,
    run_session_experiment,
    sync_train_to_incumbent,
)
from autoresirch.session_manager.schemas import RunIntent
from autoresirch.session_manager.shared.storage import load_session_context, load_session_state

EXPERIMENT_MODE_CHOICES: tuple[ExperimentMode, ...] = ("standard", "comparative")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Manage autonomous search sessions and run artifacts.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    start_parser = subparsers.add_parser("start", help="Create a new search session.")
    start_parser.add_argument("--session-id", default=None)
    start_parser.add_argument("--objective", default=None)
    start_parser.add_argument("--initiated-by", default="agent")
    start_parser.add_argument("--raw-data-path", type=Path, default=None)
    start_parser.add_argument(
        "--experiment-mode",
        choices=EXPERIMENT_MODE_CHOICES,
        default=DATASET_CONFIG.experiment_mode,
    )

    run_parser = subparsers.add_parser("run", help="Execute one run inside an existing session.")
    run_parser.add_argument("--session-id", required=True)
    run_parser.add_argument(
        "--run-role",
        required=True,
        choices=("base", "candidate", "rerun", "recovery"),
    )
    run_parser.add_argument("--parent-run-id", default=None)
    run_parser.add_argument("--compared-against-run-id", default=None)
    run_parser.add_argument("--hypothesis", default="")
    run_parser.add_argument("--mutation-summary", default="")
    run_parser.add_argument("--description", default="")

    analyze_parser = subparsers.add_parser(
        "analyze-run",
        help="Record agent-authored analysis for a run and refresh synopsis.md.",
    )
    analyze_parser.add_argument("--session-id", required=True)
    analyze_parser.add_argument("--run-id", required=True)
    analyze_parser.add_argument("--summary-label", required=True)
    analyze_parser.add_argument("--freeform-analysis", required=True)
    analyze_parser.add_argument("--likely-helped", action="append", default=[])
    analyze_parser.add_argument("--likely-hurt", action="append", default=[])
    analyze_parser.add_argument("--confidence", type=float, required=True)
    analyze_parser.add_argument("--next-step-reasoning", required=True)

    sync_parser = subparsers.add_parser(
        "sync-incumbent",
        help="Restore autoresirch/train.py from the incumbent run.",
    )
    sync_parser.add_argument("--session-id", required=True)

    status_parser = subparsers.add_parser("status", help="Show current session status.")
    status_parser.add_argument("--session-id", required=True)

    finalize_parser = subparsers.add_parser("finalize", help="Finalize a session and write its summary.")
    finalize_parser.add_argument("--session-id", required=True)
    finalize_parser.add_argument("--status", default="completed", choices=("completed", "aborted"))
    finalize_parser.add_argument("--end-reason", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "start":
        objective = args.objective
        if objective is None:
            objective = f"Maximize {resolve_primary_metric_name(args.experiment_mode, METRIC_CONFIG)}"
        context = create_session(
            session_id=args.session_id,
            objective=objective,
            initiated_by=args.initiated_by,
            experiment_mode=args.experiment_mode,
            raw_data_path=args.raw_data_path,
        )
        print(context.session_id)
        return 0

    if args.command == "run":
        decision = run_session_experiment(
            RunIntent(
                session_id=args.session_id,
                run_role=args.run_role,
                parent_run_id=args.parent_run_id,
                compared_against_run_id=args.compared_against_run_id,
                hypothesis=args.hypothesis,
                mutation_summary=args.mutation_summary,
                description=args.description,
            )
        )
        print(json.dumps(asdict(decision), indent=2, sort_keys=True))
        return 0

    if args.command == "analyze-run":
        agent_analysis = record_agent_analysis(
            session_id=args.session_id,
            run_id=args.run_id,
            summary_label=args.summary_label,
            freeform_analysis=args.freeform_analysis,
            likely_helped=args.likely_helped,
            likely_hurt=args.likely_hurt,
            confidence=args.confidence,
            next_step_reasoning=args.next_step_reasoning,
        )
        print(json.dumps(asdict(agent_analysis), indent=2, sort_keys=True))
        return 0

    if args.command == "sync-incumbent":
        snapshot_path = sync_train_to_incumbent(args.session_id)
        print(snapshot_path)
        return 0

    if args.command == "status":
        context = load_session_context(args.session_id)
        state = load_session_state(args.session_id)
        payload = {
            "session_id": state.session_id,
            "status": state.status,
            "experiment_mode": context.experiment_mode,
            "raw_data_path": str(context.raw_data_path),
            "latest_run_id": state.latest_run_id,
            "incumbent_run_id": state.incumbent_run_id,
            "incumbent_primary_metric_value": state.incumbent_primary_metric_value,
            "keep_count": state.keep_count,
            "discard_count": state.discard_count,
            "crash_count": state.crash_count,
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    if args.command == "finalize":
        summary = finalize_session(
            args.session_id,
            status=args.status,
            end_reason=args.end_reason,
        )
        print(json.dumps(asdict(summary), indent=2, sort_keys=True))
        return 0

    raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
