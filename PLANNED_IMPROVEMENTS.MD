# Planned Improvement: Richer Run And Session Metadata

## Objective Analysis

**What:** Expand the experiment record so both each run and each multi-run search session preserve enough structured and human-readable metadata for the next agent to make better decisions.

**Why:** The current harness already stores strong factual run data in `summary.json`, but there is no first-class concept of a session rooted at a base model. Important information is split across per-run JSON, `results.tsv`, and the agent's transient context.

**Success looks like:** After an agent finishes a session, another agent should be able to open the session directory and answer:

- what base model the session started from
- what runs were tried, in what order, and why
- when the incumbent changed and by how much
- how the non-primary metrics behaved across the session
- why the session ended
- what the next session should try

## Current State

The current setup already has a good run-level core:

- `prepare.py` writes `summary.json` with aggregate metrics, fold metrics, `ArchitectureSpec`, and fixed configs
- `results.tsv` stores keep/discard/crash plus a short description and a subset of aggregate metrics
- each run currently lands in a top-level `runs/<timestamp>/` directory

The main gaps are:

- no first-class session rooted at a base run
- no structured link between a `results.tsv` row and a specific run directory
- no snapshot of the exact `train.py` code that produced a run
- no run-local record of the keep/discard decision
- no session-local record of incumbent progression
- no session-level summary of what helped, what hurt, and what to try next

## Session Model

A session should be the canonical unit of search, and a run should be the canonical unit of evidence.

Definition:

- a session starts when the agent chooses a base model and runs it as the first run of the session
- the base model is always `session_run_index = 0`
- every later candidate, rerun, or recovery run belongs to the same `session_id`
- the session ends when the agent reaches a stopping condition such as budget exhaustion, plateau, crash exhaustion, or a human stop
- at the end of the session, the agent writes a final session summary

This gives you a clean hierarchy:

- session = one bounded search effort with a base model and an ending summary
- run = one concrete experiment inside that effort

## Recommended Canonical Storage Layout

`sessions/` should become the canonical store.

```text
sessions/
  <session_id>/
    results.tsv
    session_context.json
    session_state.json
    session_summary.json
    session_summary.md
    runs/
      000_base__<run_id>/
        summary.json
        run_context.json
        decision.json
        synopsis.md
        train_snapshot.py
      001_candidate__<run_id>/
        ...
      002_candidate__<run_id>/
        ...
```

Recommended behavior:

- `sessions/<session_id>/results.tsv` is the canonical append-only tabular ledger for that session
- `sessions/<session_id>/runs/` is the source of truth for per-run artifacts
- `session_context.json` is written once at session start and should be treated as immutable
- `session_state.json` is updated after each run as the live machine-readable ledger
- `session_summary.json` and `session_summary.md` are written when the session ends

Compatibility note:

- a repo-root `results.tsv` may still exist as a derived cross-session index if you want a global rollup
- `runs/latest_summary.json` can remain as a convenience mirror if needed
- neither should be the canonical historical store once `sessions/` exists

## Recommended Metadata To Track Per Run

### 1. Run Identity And Session Linkage

Track these fields in each run directory:

- `run_id`
- `session_id`
- `session_run_index`
- `run_role`
- `is_base_run`
- `parent_run_id`
- `compared_against_run_id`
- `best_known_run_id_at_start`
- `started_at`
- `completed_at`
- `duration_seconds`
- `run_dir`

Recommended `run_role` values:

- `base`
- `candidate`
- `rerun`
- `recovery`

Why this matters:

- the next agent needs to know where the run sits in the session trajectory
- base runs need to be distinguished from mutations of earlier runs
- comparing against the parent run and comparing against the current incumbent are not always the same thing

### 2. Provenance And Code Snapshot

Track:

- `git_commit`
- `git_branch`
- `git_is_dirty`
- `train_py_sha256`
- `architecture_fingerprint`
- `program_md_sha256`
- `train_snapshot.py`

Why this matters:

- `ArchitectureSpec` does not fully describe the run because `train.py` can contain extra architectural behavior outside the spec
- if the autonomous loop or instructions change, the session should preserve that provenance too
- a future agent should not need git archaeology to reconstruct a run

### 3. Search Change Record

Track:

- `mutation_summary`
- `changed_fields`
- `hypothesis`
- `hypothesis_result`

Examples:

- `mutation_summary: "Reduced the first layer from 256 to 192 while keeping depth and dropout fixed"`
- `changed_fields: {"hidden_dims": [[256, 128, 64, 32, 16], [192, 128, 64, 32, 16]]}`
- `hypothesis: "A narrower input projection may reduce overcapacity and improve generalization"`
- `hypothesis_result: "supported" | "unsupported" | "inconclusive"`

Why this matters:

- this is the most useful metadata for next-run iteration
- the next agent learns much more from "what changed and why" than from the score alone

### 4. Decision Metadata

Keep the current keep/discard rule exactly as-is, but record it inside the run directory:

- `decision_status`
- `decision_metric_name`
- `decision_metric_value`
- `decision_baseline_value`
- `decision_delta`
- `decision_epsilon`
- `decision_reason`

Why this matters:

- the decision should live both in the run directory and in the session-local `results.tsv`
- the run record should contain both what happened and what the agent did with it

### 5. Aggregate Metric Bundle

Continue storing the current summary metrics, but make the role of each metric explicit.

Primary decision metric:

- `weighted_cv_auc`

Secondary interpretation metrics:

- `weighted_cv_rmse_mean`
- `cv_rmse_mean`
- `cv_rmse_std`
- `weighted_cv_mae_mean`
- `weighted_cv_r2_mean`
- `weighted_cv_pearson_r_mean`
- `weighted_cv_spearman_r_mean`
- `pooled_cv_rmse`
- any test-split metrics when enabled

Why this matters:

- keep/discard should still depend only on `weighted_cv_auc`
- the secondary metrics still matter for interpreting whether the run improved ranking, absolute error, robustness, or only one narrow aspect of behavior

### 6. Derived Run Diagnostics

Your current `fold_results` are already useful. Add compact derived summaries so the next agent does not need to mentally recompute them.

Fold robustness:

- `fold_count`
- `fold_size_min`
- `fold_size_median`
- `fold_size_max`
- `auc_defined_fold_count`
- `auc_nan_fold_count`
- `pearson_nan_fold_count`
- `spearman_nan_fold_count`
- `best_auc_fold`
- `worst_auc_fold`
- `best_rmse_fold`
- `worst_rmse_fold`
- `largest_gene_fold`
- `smallest_gene_fold`

Metric disagreement:

- `auc_vs_rmse_tradeoff`
- `auc_vs_pearson_tradeoff`
- `auc_vs_spearman_tradeoff`
- `broad_improvement_flag`
- `outlier_fold_flag`

Prediction behavior:

- fraction of scaled predictions clipped to `0.0`
- fraction of scaled predictions clipped to `1.0`
- scaled prediction mean
- scaled prediction standard deviation
- per-fold effective positive rate for the AUC label

Training dynamics:

- `epoch_count_min`
- `epoch_count_median`
- `epoch_count_max`
- `best_epoch_min`
- `best_epoch_median`
- `best_epoch_max`
- `best_epoch_ratio_mean`
- `seconds_per_fold_mean`

Why this matters:

- aggregate metrics can hide instability across genes
- undefined AUC folds should be easy to interpret instead of discovered indirectly through `NaN`
- clipping and label-balance diagnostics explain why some folds produce odd AUC behavior

## Recommended Metadata To Track Per Session

### 1. Session Identity And Purpose

Track:

- `session_id`
- `status`
- `started_at`
- `completed_at`
- `duration_seconds`
- `objective`
- `initiated_by`
- `end_reason`

Recommended `status` values:

- `active`
- `completed`
- `aborted`

Why this matters:

- a session is not just a folder of runs, it is a bounded search effort with a purpose and a stopping reason
- the next agent needs to know whether a session ended because it succeeded, plateaued, was interrupted, or ran into instability

### 2. Base Model And Fixed Experiment Boundary

Track:

- `base_run_id`
- `base_architecture_fingerprint`
- `base_primary_metric_value`
- `dataset_config_fingerprint`
- `training_config_fingerprint`
- `metric_config_fingerprint`
- `constraints_fingerprint`
- `program_md_sha256`

Why this matters:

- the session baseline is the anchor for all later comparisons
- if the evaluation boundary changed, the session should not be treated as directly comparable to earlier sessions

### 3. Ordered Run Ledger And Incumbent History

Track:

- `ordered_run_ids`
- `latest_run_id`
- `incumbent_progression`
- `keep_count`
- `discard_count`
- `crash_count`
- `rerun_count`
- `final_incumbent_run_id`

`incumbent_progression` should capture each point where the best-known run changed, including:

- `after_run_id`
- `new_incumbent_run_id`
- `primary_metric_value`
- `delta_vs_previous_incumbent`

Why this matters:

- this is the session-level equivalent of lineage tracking
- the next agent should be able to see the sequence of improvements, not just the final winner

### 4. Search Space Coverage

Track:

- `families_tried`
- `activations_tried`
- `normalizations_tried`
- `dropout_values_tried`
- `depth_values_tried`
- `max_width_values_tried`
- `param_count_min`
- `param_count_max`
- `unique_architecture_count`
- `repeated_architecture_count`

Why this matters:

- the next session should know what parts of the search space were already explored
- this prevents repetitive local search with poor memory of what was already ruled out

### 5. Session Outcome Summary

Track:

- `total_runs`
- `total_train_seconds`
- `best_run_id`
- `best_primary_metric_value`
- `delta_from_base`
- `best_secondary_metric_snapshot`
- `most_promising_near_misses`

Why this matters:

- a session should leave behind a compact answer to "what did this search effort actually accomplish?"
- near misses often make the best starting points for the next session even when they were discarded

### 6. Learning And Handoff Notes

Track:

- `hypotheses_tested`
- `hypotheses_supported`
- `hypotheses_unsupported`
- `patterns_that_helped`
- `patterns_that_hurt`
- `instability_patterns`
- `unresolved_questions`
- `recommended_next_starting_run_id`
- `recommended_next_mutations`

Why this matters:

- the session summary should not just say which run won
- it should preserve what the session learned about the local search landscape

## Recommended Session Artifacts

### `results.tsv`

Write once at session start with the header, then append one row per run.

Purpose:

- provide a simple session-local ledger that is easy for both agents and humans to scan
- preserve run ordering, statuses, and key metrics without opening every run directory
- keep the canonical tabular history scoped to the session instead of mixing unrelated search efforts together

### `session_context.json`

Write once at session start.

Purpose:

- identify the session
- record the base run
- freeze the evaluation boundary and search objective

### `session_state.json`

Update after each run.

Purpose:

- act as the live machine-readable ledger for the session
- track ordered runs, counts, incumbent progression, and current status

### `session_summary.json`

Write once at the end.

Purpose:

- provide a structured final handoff for tools and future automation

### `session_summary.md`

Write once at the end.

Purpose:

- provide the human-readable narrative of what happened in the session and what to do next

## Proposed `synopsis.md` Contents

This file should be short, opinionated, and optimized for the next iteration inside the same session.

Suggested outline:

1. Run header
   - `session_id`
   - `run_id`
   - `session_run_index`
   - keep/discard/crash
   - primary metric and delta vs comparison run

2. Architecture synopsis
   - one short paragraph describing the architecture and the hypothesis behind it

3. Performance overview
   - primary metric result
   - secondary metric movement
   - parameter count and runtime context

4. Robustness notes
   - whether gains or losses were broad-based
   - best and worst folds
   - count of undefined AUC or correlation folds
   - any obvious instability or metric disagreement

5. Interpretation
   - what likely helped
   - what likely hurt
   - whether the result supported the hypothesis

6. Next-run guidance
   - 2 or 3 concrete suggested follow-up mutations

Suggested frontmatter:

```yaml
session_id: 20260313-s01
run_id: 20260313-091140
session_run_index: 4
run_role: candidate
parent_run_id: 20260313-090943
decision_status: keep
decision_metric_name: weighted_cv_auc
decision_metric_value: 0.707199
decision_baseline_value: 0.697144
decision_delta: 0.010055
hypothesis: Narrower first layer may improve generalization.
mutation_summary: Reduced hidden_dims from [256, 128, 64, 32, 16] to [192, 128, 64, 32, 16].
```

Tone and rules:

- keep the keep/discard rule tied only to `weighted_cv_auc`
- explicitly discuss secondary metrics without overriding the decision rule
- call out tradeoffs instead of hiding them
- mention whether the result appears robust across folds or concentrated in a few genes
- suggest the next architectural move, not just summarize the past run

## Proposed `session_summary.md` Contents

This file should be the end-of-session handoff artifact.

Suggested outline:

1. Session header
   - `session_id`
   - start and end times
   - status
   - end reason
   - total runs

2. Starting point
   - base run id
   - base architecture
   - base primary metric

3. Champion progression
   - each time the incumbent changed
   - the metric delta at that point
   - the mutation that caused the improvement

4. Search space explored
   - architectures tried
   - families, activations, normalizations, dropout range, parameter range
   - what areas were underexplored

5. Session-wide findings
   - patterns that helped
   - patterns that hurt
   - metric tradeoffs that repeated across runs
   - instability or failure modes

6. Final recommendation
   - best run to continue from
   - whether to continue this line of search or start a fresh session
   - 3 concrete next-session mutations

Suggested frontmatter:

```yaml
session_id: 20260313-s01
status: completed
base_run_id: 20260313-085354
best_run_id: 20260313-091140
final_incumbent_run_id: 20260313-091140
total_runs: 14
keep_count: 4
discard_count: 9
crash_count: 1
best_primary_metric_value: 0.707199
delta_from_base: 0.022226
end_reason: Plateau after repeated non-improving width variants.
```

Tone and rules:

- summarize the session honestly, not just the winning run
- mention discarded runs that still taught something useful
- make the final recommendation concrete enough that the next agent can start immediately

## Minimum Viable Version

If you want the highest return for the least code, implement these first:

1. Create `sessions/<session_id>/runs/` and make the base model the first run of the session.
2. Create `sessions/<session_id>/results.tsv` at session start with the session-local header.
3. Write `session_context.json` at session start.
4. Write all per-run artifacts inside the session run directory:
   - `summary.json`
   - `run_context.json`
   - `decision.json`
   - `synopsis.md`
   - `train_snapshot.py`
5. Append one row to the session-local `results.tsv` after each run.
6. Update `session_state.json` after each run with ordered runs, counts, and incumbent progression.
7. Write `session_summary.md` and `session_summary.json` at session end.

This would already give the next agent much better memory across both runs and whole search sessions without changing the evaluation rule.

## Nice-To-Have Version

After the minimum version works, add:

1. derived run robustness summaries
2. prediction clipping and label-balance diagnostics
3. structured search-space coverage summaries at the session level
4. a hypothesis ledger that tracks which ideas were supported or unsupported
5. automatic generation of "recommended next session seeds" from the best run and best near-miss runs

## Implementation Plan

### Task 1: Introduce Session Structure

Scope:

- add `sessions/` as the canonical artifact root
- define `session_id`
- define base-run semantics
- create a session-local `results.tsv`
- nest runs under `sessions/<session_id>/runs/`

Acceptance criteria:

- every run belongs to exactly one session
- every session begins with a clearly identified base run
- every session has its own `results.tsv`

### Task 2: Add Immutable Run Provenance Inside Sessions

Scope:

- write `train_snapshot.py`
- write `run_context.json`
- include git metadata, hashes, timestamps, session linkage, and architecture fingerprint

Acceptance criteria:

- every new run directory contains enough information to reconstruct the exact model code used
- every run contains machine-readable provenance and lineage fields

### Task 3: Add Run Decision Artifacts

Scope:

- append one row to the session-local `results.tsv` after each run
- write `decision.json` from the autonomous loop
- write `synopsis.md`
- link each decision to the comparison run and the current session incumbent

Acceptance criteria:

- each session has a complete tabular ledger of its own runs
- keep/discard/crash is visible inside each run directory
- the run synopsis preserves secondary-metric interpretation without changing the decision rule

### Task 4: Add Live Session State

Scope:

- write `session_state.json`
- update it after each run
- track ordered runs, counts, and incumbent progression

Acceptance criteria:

- an agent can inspect an active session and immediately understand the search trajectory so far

### Task 5: Add Final Session Summary

Scope:

- write `session_summary.json`
- write `session_summary.md`
- summarize base model, improvements, explored search space, lessons, and next moves

Acceptance criteria:

- the next agent can start a new session or continue from the current best run without reconstructing the prior search manually

### Task 6: Add Derived Diagnostics

Scope:

- summarize fold robustness
- summarize metric disagreement
- summarize session-level search-space coverage
- optionally add prediction clipping and label-balance behavior

Acceptance criteria:

- both run and session summaries can reference compact diagnostics instead of re-deriving everything from raw data

## Recommendation

The most important design principle is this:

**Treat the session as the primary handoff artifact, and each run as supporting evidence inside that session.**

That means each session should preserve:

- where it started
- what it tried
- when the incumbent changed
- why it stopped
- what the next session should do

If you do only one session-level thing beyond writing a final session summary, I would make it **incumbent progression tracking**. The next agent will improve faster if it can see not just the final winner, but the sequence of mutations that actually changed the best-known model during the session.

## Discrete Implementation Plan

This section is the exact implementation plan for turning the design above into working code and instructions.

### Implementation Strategy

Use a small orchestration layer to manage sessions and artifacts, instead of asking the agent to hand-edit session files correctly every time.

The separation of responsibilities should be:

- `prepare.py`: fixed evaluation harness that trains, evaluates, and writes factual run outputs
- `session_manager.py`: session orchestration layer that creates session directories, allocates run ids, writes session metadata, computes keep/discard/crash, updates session state, and generates markdown summaries
- `program.md`: strict runbook that tells the agent to use the orchestration commands and forbids manual editing of generated session state files
- `train.py`: still the only code file the agent edits for architecture changes

This keeps the evaluation boundary clean while making the metadata workflow reliable.

## File-By-File Change Plan

### 1. New File: `session_manager.py`

Create a new top-level file [session_manager.py](/Users/lucasplatter/sirchml-autoresearch/session_manager.py).

This file should be both:

- a small library for session metadata and artifact management
- a CLI entrypoint driven by `argparse`

It should own the canonical session workflow.

#### Data Structures To Add

Define dataclasses or typed dictionaries for:

- `SessionContext`
- `SessionState`
- `RunIntent`
- `RunContext`
- `DecisionRecord`
- `SessionSummaryRecord`

These should be the only schemas used for session JSON files.

#### Core Functions To Add

Add functions for:

- `create_session(...)`
- `load_session_context(...)`
- `load_session_state(...)`
- `save_session_context(...)`
- `save_session_state(...)`
- `allocate_run_dir(...)`
- `write_run_context(...)`
- `write_train_snapshot(...)`
- `append_session_results_row(...)`
- `write_decision_record(...)`
- `write_run_synopsis(...)`
- `finalize_session(...)`
- `write_session_summary(...)`
- `sync_train_to_incumbent(...)`

#### CLI Subcommands To Add

Implement these subcommands:

1. `start`
   - creates `sessions/<session_id>/`
   - creates `sessions/<session_id>/results.tsv`
   - writes `session_context.json`
   - writes initial `session_state.json`

2. `run`
   - loads the active session
   - allocates the next run directory under `sessions/<session_id>/runs/`
   - snapshots `train.py`
   - writes `run_context.json`
   - invokes `prepare.run_experiment(...)`
   - writes `decision.json`
   - writes `synopsis.md`
   - appends the session-local `results.tsv`
   - updates `session_state.json`

3. `sync-incumbent`
   - copies the incumbent run's `train_snapshot.py` back to repo-root `train.py`
   - used after discarded or crashed runs so the working file always matches the best known model in the session

4. `status`
   - prints the current session id, incumbent run id, best metric, latest run id, and counts
   - used by the agent before deciding the next mutation

5. `finalize`
   - writes `session_summary.json`
   - writes `session_summary.md`
   - marks the session as completed or aborted in `session_state.json`

#### Exact Behavior Of `run`

The `run` subcommand should accept at least:

- `--session-id`
- `--run-role`
- `--parent-run-id`
- `--compared-against-run-id`
- `--hypothesis`
- `--mutation-summary`
- `--description`

Flow:

1. Load `session_context.json` and `session_state.json`.
2. Determine `session_run_index`.
3. Create `run_id` and run directory name `NNN_<role>__<run_id>`.
4. Compute provenance fields:
   - git commit
   - git branch
   - dirty flag
   - `train.py` hash
   - `program.md` hash
5. Write `run_context.json` before running training.
6. Copy `train.py` to `train_snapshot.py` before running training.
7. Call `prepare.run_experiment(...)` with the allocated run directory.
8. On success:
   - load `summary.json`
   - compare `summary.primary_metric_value` against the current incumbent using `improvement_epsilon`
   - determine `keep` or `discard`
   - write `decision.json`
   - write `synopsis.md`
   - append the session-local `results.tsv`
   - update `session_state.json`
9. On failure:
   - write a crash record
   - append a `crash` row to the session-local `results.tsv`
   - update crash counts in `session_state.json`
   - preserve traceback or error text in the run directory

#### Additional Crash Artifact

Add one crash-specific file:

- `failure.json` or `error.txt`

This should contain the exception type, message, timestamp, and traceback.

That file is needed because a crashed run will not have a complete `summary.json`.

### 2. Update `prepare.py`

Modify [prepare.py](/Users/lucasplatter/sirchml-autoresearch/prepare.py), but keep it as the fixed evaluation harness.

#### Path And Runtime Changes

Current code hard-codes top-level run output through:

- `RUNS_DIR` at [prepare.py#L32](/Users/lucasplatter/sirchml-autoresearch/prepare.py#L32)
- `_make_run_dir()` at [prepare.py#L406](/Users/lucasplatter/sirchml-autoresearch/prepare.py#L406)
- `save_run_summary()` at [prepare.py#L1314](/Users/lucasplatter/sirchml-autoresearch/prepare.py#L1314)
- `run_experiment()` at [prepare.py#L1413](/Users/lucasplatter/sirchml-autoresearch/prepare.py#L1413)

Change this so `run_experiment()` can write into a caller-supplied run directory.

#### Exact API Changes

Add optional parameters to `run_experiment(...)`:

- `run_dir: Path | None = None`
- `latest_summary_path: Path | None = None`

Behavior:

- if `run_dir` is passed, write artifacts there
- if `run_dir` is `None`, preserve the current one-off behavior and create a top-level run directory under `runs/`
- if `latest_summary_path` is passed, write the latest-summary mirror there
- if `latest_summary_path` is `None`, preserve the current top-level mirror behavior

#### Exact Helper Changes

Refactor:

- `_make_run_dir()` so it can accept a root path
- `save_run_summary()` so it optionally writes `latest_summary.json` to a caller-provided path
- `ensure_runtime_dirs()` so it does not treat repo-root `results.tsv` as mandatory for canonical workflows

The repo-root `results.tsv` should become optional compatibility output, not something the harness always initializes.

#### Add Derived Diagnostics To `summary.json`

Add a helper like:

- `build_run_diagnostics(fold_results: list[FoldResult]) -> dict[str, Any]`

Include at least:

- fold size summary
- counts of undefined AUC, Pearson, and Spearman folds
- best and worst fold ids by AUC and RMSE
- epoch/best_epoch aggregate summaries

Write the diagnostics block into `summary.json`.

#### Optional But Recommended `FoldResult` Extension

To support clipping and label-balance diagnostics, extend the fold-level return path to retain:

- scaled prediction mean
- scaled prediction standard deviation
- clipped-to-zero fraction
- clipped-to-one fraction
- effective positive rate

This will require either:

- extending `FoldResult`, or
- adding a nested `diagnostics` field under each fold payload

#### Important Constraint

Do not move keep/discard logic into `prepare.py`.

`prepare.py` should still report facts, not agent decisions.
The decision logic belongs in `session_manager.py`.

### 3. Update `program.md`

Rewrite [program.md](/Users/lucasplatter/sirchml-autoresearch/program.md) so it becomes a strict session runbook rather than a loose experiment note.

#### Core Rule Changes

Keep the core rule that the agent may edit only `train.py` among code files.

Add an explicit allowance that the agent may also create and update generated artifacts under:

- `sessions/`
- `run.log`

Add an explicit prohibition:

- do not manually edit `session_state.json`
- do not manually edit `results.tsv`
- do not manually edit `decision.json`
- do not manually edit `session_summary.json`

Those files must be produced through `session_manager.py`.

#### Replace “Experiment Protocol” With “Session Protocol”

Replace the current protocol in [program.md#L32](/Users/lucasplatter/sirchml-autoresearch/program.md#L32) with a session-aware protocol:

1. Read `README.md`, `prepare.py`, `train.py`, and `PLANNED_IMPROVEMENTS.md`.
2. Confirm the cached dataset exists. If not, tell the human to run `uv run prepare.py`.
3. Start a session with `uv run python session_manager.py start ...`.
4. Run the unmodified baseline as the base run with `uv run python session_manager.py run --run-role base ...`.
5. Before each candidate run, define:
   - parent run id
   - compared-against run id
   - hypothesis
   - mutation summary
   - description
6. Run the candidate through `session_manager.py run`.
7. Inspect the reported decision.
8. Run `uv run python session_manager.py sync-incumbent --session-id <id>` after every run so `train.py` always matches the current incumbent.
9. Continue until a stopping condition is met.
10. Finalize the session with `uv run python session_manager.py finalize --session-id <id> --end-reason "..."`

#### Result Logging Section Changes

Replace the existing repo-root `results.tsv` instructions with:

- use `sessions/<session_id>/results.tsv` as the canonical ledger
- never append rows manually
- let `session_manager.py run` append the row after each run

The header should be expanded to include session-specific columns.

Recommended columns:

- `session_id`
- `session_run_index`
- `run_id`
- `run_role`
- `parent_run_id`
- `compared_against_run_id`
- `commit`
- `weighted_cv_rmse_mean`
- `cv_rmse_std`
- `weighted_cv_auc`
- `weighted_cv_pearson_r`
- `weighted_cv_spearman_r`
- `status`
- `num_params`
- `train_seconds`
- `hypothesis`
- `mutation_summary`
- `description`
- `run_dir`

#### Crash Handling Changes

Update the crash section so that:

- `session_manager.py run` records crash metadata automatically
- the agent uses `sync-incumbent` after a crash
- the agent finalizes the session if crash frequency or unrecoverable failures make further search unproductive

#### Simplicity Rule Changes

Keep the simplicity rule, but add:

- if two candidates are close on `weighted_cv_auc`, the session summary should explicitly prefer the simpler incumbent
- session summaries should call out when a more complex model did not justify its added parameters

### 4. Update `README.md`

Modify [README.md](/Users/lucasplatter/sirchml-autoresearch/README.md) so it documents the new canonical workflow.

#### Exact README Changes

Update:

- “How the repo works”
- “Quick start”
- “Repo structure”

New points to add:

- `uv run train.py` remains available for one-off manual runs
- `uv run python session_manager.py ...` is the canonical session workflow for autonomous search
- session artifacts now live under `sessions/`
- the per-session `results.tsv` is the canonical tabular run ledger

Add the new repo structure entries:

- `session_manager.py`
- `sessions/`

### 5. Update `.gitignore`

Modify [.gitignore](/Users/lucasplatter/sirchml-autoresearch/.gitignore).

Add:

- `sessions/`

Keep:

- `runs/`
- `results.tsv`

Reason:

- session artifacts are generated search outputs and should not be checked in by default

### 6. No Persistent Logic Change To `train.py`

Do not make any architectural or metadata-tracking logic part of [train.py](/Users/lucasplatter/sirchml-autoresearch/train.py).

The only standing expectation for `train.py` should remain:

- define `ARCHITECTURE`
- define `build_model(context)`
- optionally contain future architecture edits by the agent

This avoids mixing search bookkeeping into the editable model file.

### 7. Update Tests

#### Existing Test File: `tests/test_prepare.py`

Extend [tests/test_prepare.py](/Users/lucasplatter/sirchml-autoresearch/tests/test_prepare.py) to cover the new `prepare.py` behavior.

Add tests for:

- `run_experiment(..., run_dir=custom_dir)` writes `summary.json` to the provided path
- custom latest-summary path behavior
- derived diagnostics block presence in `summary.json`
- compatibility behavior when `run_experiment()` is called without a custom run directory

#### New Test File: `tests/test_session_manager.py`

Create [tests/test_session_manager.py](/Users/lucasplatter/sirchml-autoresearch/tests/test_session_manager.py).

Add tests for:

1. `start` creates:
   - session directory
   - session-local `results.tsv`
   - `session_context.json`
   - initial `session_state.json`

2. `run --run-role base` creates:
   - run directory
   - `train_snapshot.py`
   - `run_context.json`
   - `summary.json`
   - `decision.json`
   - `synopsis.md`
   - appended results row

3. a kept run:
   - updates incumbent in `session_state.json`
   - appends a `keep` row

4. a discarded run:
   - does not advance the incumbent
   - appends a `discard` row
   - works with `sync-incumbent`

5. a crashed run:
   - appends a `crash` row
   - increments crash counts
   - writes `failure.json` or `error.txt`

6. `finalize` writes:
   - `session_summary.json`
   - `session_summary.md`
   - final status and end reason

#### Test Strategy

Do not require the full real dataset for session-manager tests.

Use either:

- monkeypatched fake `run_experiment()` outputs, or
- a tiny synthetic dataset fixture and temporary directories

The tests should verify metadata behavior, not expensive model training.

### 8. Optional `pyproject.toml` Change

No dependency change is required.

Leave [pyproject.toml](/Users/lucasplatter/sirchml-autoresearch/pyproject.toml) unchanged unless you want ergonomic console scripts such as:

- `uv run session-manager start`
- `uv run session-manager run`

If you want that convenience, add a `[project.scripts]` entry. Otherwise `uv run python session_manager.py ...` is sufficient.

## Ordered Execution Plan

This is the recommended implementation order.

### Step 1: Refactor `prepare.py` For External Run Directories

Why first:

- `session_manager.py` depends on being able to tell the harness where to write run artifacts

Done when:

- `run_experiment()` accepts a caller-supplied output directory
- one-off direct runs still work

### Step 2: Add `session_manager.py` Session Creation And Run Lifecycle

Why second:

- this creates the canonical session structure and per-run artifact flow

Done when:

- `start`, `run`, `sync-incumbent`, `status`, and `finalize` all exist

### Step 3: Add Decision And Summary Generation

Why third:

- the raw run lifecycle has to exist before the decision and markdown layers can be generated

Done when:

- each run produces `decision.json` and `synopsis.md`
- each session produces `session_summary.json` and `session_summary.md`

### Step 4: Rewrite `program.md`

Why fourth:

- the workflow should not be documented until the commands it references exist

Done when:

- the agent instructions reference only the supported session workflow
- the allowed writes and forbidden manual edits are explicit

### Step 5: Update README And Ignore Rules

Why fifth:

- these are follow-on docs once the behavior is defined

Done when:

- the README matches the canonical session workflow
- generated session artifacts are ignored

### Step 6: Add And Run Tests

Why last:

- the final APIs and file layout must be stable before writing focused tests

Done when:

- new tests pass for both harness compatibility and session metadata behavior

## Acceptance Criteria For The Whole Change

The change is complete only when all of the following are true:

- a new session can be started without hand-creating files
- the base run is recorded as `session_run_index = 0`
- every run is written under `sessions/<session_id>/runs/`
- every session has its own `results.tsv`
- every run has `summary.json`, `run_context.json`, `train_snapshot.py`, `decision.json`, and `synopsis.md`
- discarded and crashed runs do not leave `train.py` drifted away from the incumbent after `sync-incumbent`
- each session ends with `session_summary.json` and `session_summary.md`
- `program.md` tells the agent exactly how to use the session workflow
- `README.md` documents the new canonical workflow
- tests cover both the harness changes and the session orchestration layer
