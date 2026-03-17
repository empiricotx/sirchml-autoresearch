from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class SessionContext:
    session_id: str
    base_run_id: str
    started_at: str
    objective: str
    initiated_by: str
    dataset_config_fingerprint: str
    split_config_fingerprint: str
    training_config_fingerprint: str
    metric_config_fingerprint: str
    constraints_fingerprint: str
    program_md_sha256: str


@dataclass
class SessionState:
    session_id: str
    status: str
    started_at: str
    objective: str
    initiated_by: str
    base_run_id: str | None = None
    base_architecture_fingerprint: str | None = None
    base_primary_metric_value: float | None = None
    completed_at: str | None = None
    duration_seconds: float | None = None
    end_reason: str | None = None
    ordered_run_ids: list[str] = field(default_factory=list)
    latest_run_id: str | None = None
    incumbent_run_id: str | None = None
    incumbent_primary_metric_value: float | None = None
    final_incumbent_run_id: str | None = None
    incumbent_progression: list[dict[str, Any]] = field(default_factory=list)
    keep_count: int = 0
    discard_count: int = 0
    crash_count: int = 0
    rerun_count: int = 0
    run_dirs: dict[str, str] = field(default_factory=dict)
    run_roles: dict[str, str] = field(default_factory=dict)
    run_statuses: dict[str, str] = field(default_factory=dict)
    run_primary_metric_values: dict[str, float | None] = field(default_factory=dict)


@dataclass(frozen=True)
class RunIntent:
    session_id: str
    run_role: str
    parent_run_id: str | None
    compared_against_run_id: str | None
    hypothesis: str
    mutation_summary: str
    description: str


@dataclass
class RunContext:
    session_id: str
    run_id: str
    session_run_index: int
    run_role: str
    is_base_run: bool
    parent_run_id: str | None
    compared_against_run_id: str | None
    best_known_run_id_at_start: str | None
    hypothesis: str
    mutation_summary: str
    description: str
    started_at: str
    completed_at: str | None
    duration_seconds: float | None
    run_dir: str
    git_commit: str | None
    git_branch: str | None
    git_is_dirty: bool | None
    train_py_sha256: str
    architecture_fingerprint: str | None
    program_md_sha256: str
    architecture_spec: dict[str, Any] | None = None
    summary_path: str | None = None
    failure_path: str | None = None
    hypothesis_result: str | None = None


@dataclass(frozen=True)
class DecisionRecord:
    session_id: str
    run_id: str
    decision_status: str
    decision_metric_name: str
    decision_metric_value: float | None
    decision_baseline_run_id: str | None
    decision_baseline_value: float | None
    decision_delta: float | None
    decision_epsilon: float | None
    decision_reason: str
    incumbent_before_run_id: str | None
    incumbent_after_run_id: str | None
    compared_against_run_id: str | None
    hypothesis_result: str | None


@dataclass(frozen=True)
class AnalysisInputRecord:
    schema_version: int
    analysis_mode: str
    session_id: str
    run_id: str
    session_run_index: int
    run_role: str
    decision_status: str
    compared_against_run_id: str | None
    base_run_id: str | None
    best_known_run_id_at_start: str | None
    hypothesis: str
    mutation_summary: str
    description: str
    architecture: dict[str, Any] | None
    decision: dict[str, Any]
    metrics: dict[str, dict[str, Any]]
    diagnostics: dict[str, Any]
    rule_based_interpretation: list[str]
    failure: dict[str, Any] | None
    analysis_constraints: dict[str, Any]


@dataclass(frozen=True)
class AgentAnalysisRecord:
    schema_version: int
    session_id: str
    run_id: str
    analysis_mode: str
    created_at: str
    analysis_input_sha256: str
    decision_status: str
    summary_label: str
    freeform_analysis: str
    likely_helped: list[str]
    likely_hurt: list[str]
    confidence: float
    next_step_reasoning: str


@dataclass(frozen=True)
class SessionSummaryRecord:
    session_id: str
    status: str
    started_at: str
    completed_at: str | None
    duration_seconds: float | None
    objective: str
    initiated_by: str
    end_reason: str | None
    base_run_id: str | None
    best_run_id: str | None
    final_incumbent_run_id: str | None
    total_runs: int
    keep_count: int
    discard_count: int
    crash_count: int
    rerun_count: int
    best_primary_metric_value: float | None
    delta_from_base: float | None
    best_secondary_metric_snapshot: dict[str, float | None]
    most_promising_near_misses: list[dict[str, Any]]
    search_space_coverage: dict[str, Any]
    hypotheses_tested: list[str]
    hypotheses_supported: list[str]
    hypotheses_unsupported: list[str]
    patterns_that_helped: list[str]
    patterns_that_hurt: list[str]
    instability_patterns: list[str]
    unresolved_questions: list[str]
    recommended_next_starting_run_id: str | None
    recommended_next_mutations: list[str]


@dataclass(frozen=True)
class InterpretationMetricSpec:
    summary_attr: str
    display_name: str
    direction: str
    flat_threshold: float


@dataclass(frozen=True)
class MetricDeltaView:
    metric_name: str
    current_value: float | int | None
    delta_vs_compared: float | None
    delta_vs_base: float | None
    compared_label: str
    base_label: str

