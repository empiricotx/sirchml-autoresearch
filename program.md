# Regression Autoresearch Program

This repo is an autonomous search loop for **siRNA regression architectures**.

## Core Rule

You may edit **only** `train.py` among source-code files.

You may also create and update generated artifacts under:

- `sessions/`
- `run.log`

Do not modify:

- `prepare.py`
- `session_manager.py`
- `pyproject.toml`
- dataset files
- tests
- evaluation logic
- train/test split logic
- CV logic
- optimizer settings
- time budgets
- metric definitions

Those are fixed by the human and define the experiment boundary.

## Generated Artifact Rule

Do not manually edit generated session artifacts.

These must be created or updated only through `session_manager.py`:

- `session_state.json`
- `results.tsv`
- `decision.json`
- `session_summary.json`
- `session_summary.md`
- `synopsis.md`
- `run_context.json`
- `analysis_input.json`
- `agent_analysis.json`

## Objective

Maximize `weighted_cv_auc`.

Higher is better.

The metric is the weighted mean of per-gene validation AUC values, weighted by the number of evaluation sequences in each held-out gene fold.

## Session Protocol

1. Read `README.md`, `prepare.py`, `train.py`, and `PLANNED_IMPROVEMENTS.md`.
2. Confirm the cached dataset exists. If not, tell the human to run `uv run prepare.py`.
3. Start a session:

```bash
uv run python session_manager.py start --objective "Maximize weighted_cv_auc" --initiated-by agent
```

4. Run the unmodified baseline as the base run:

```bash
uv run python session_manager.py run \
  --session-id <session_id> \
  --run-role base \
  --hypothesis "Establish the baseline incumbent for this session." \
  --mutation-summary "Unmodified baseline architecture." \
  --description "Base run"
```

5. Before each candidate run, define:
   - `parent_run_id`
   - `compared_against_run_id`
   - `hypothesis`
   - `mutation_summary`
   - `description`

6. Edit only `train.py`.
7. Run exactly one candidate at a time through `session_manager.py`:

```bash
uv run python session_manager.py run \
  --session-id <session_id> \
  --run-role candidate \
  --parent-run-id <parent_run_id> \
  --compared-against-run-id <compared_against_run_id> \
  --hypothesis "<hypothesis>" \
  --mutation-summary "<mutation_summary>" \
  --description "<description>"
```

8. Inspect the reported decision and the generated `analysis_input.json` for that run.
9. Record agent analysis through `session_manager.py` before syncing the incumbent:

```bash
uv run python session_manager.py analyze-run \
  --session-id <session_id> \
  --run-id <run_id> \
  --summary-label "<short label>" \
  --freeform-analysis "<brief analysis that references concrete metric movement>" \
  --likely-helped "<factor that may have helped>" \
  --likely-helped "<optional second factor>" \
  --likely-hurt "<factor that may have hurt>" \
  --confidence <0.0_to_1.0> \
  --next-step-reasoning "<1-2 concrete next-step sentences>"
```

Rules for `analyze-run`:

- do not override the AUC-only keep/discard decision
- reference concrete metric deltas from `analysis_input.json`
- keep the free-form analysis concise
- suggest only one or two follow-up ideas
- do not edit `agent_analysis.json` or `synopsis.md` by hand

10. After recording agent analysis, restore `train.py` to the current incumbent:

```bash
uv run python session_manager.py sync-incumbent --session-id <session_id>
```

11. Use session status when needed:

```bash
uv run python session_manager.py status --session-id <session_id>
```

12. Continue until a stopping condition is met.
13. Finalize the session:

```bash
uv run python session_manager.py finalize \
  --session-id <session_id> \
  --status completed \
  --end-reason "<reason>"
```

## Allowed Architecture Changes

Only architecture:

- hidden layer widths
- depth
- residual vs plain MLP family
- activation choice
- normalization choice
- dropout amount
- module structure inside `build_model`

## Forbidden Changes

You must not change:

- feature engineering
- preprocessing
- gene grouping
- held-out folds
- weighting logic
- train/test split
- optimizer
- loss
- learning rate
- weight decay
- batch size
- time budget
- metric definitions
- architecture constraints in `prepare.py`
- generated session metadata by hand

## Result Logging

Use `sessions/<session_id>/results.tsv` as the canonical run ledger.

Never append rows manually.

Let `session_manager.py run` append one row after each run.

The session-local header is:

```text
session_id	session_run_index	run_id	run_role	parent_run_id	compared_against_run_id	commit	weighted_cv_rmse_mean	cv_rmse_std	weighted_cv_auc	weighted_cv_pearson_r	weighted_cv_spearman_r	status	num_params	train_seconds	decision_baseline_value	decision_delta	hypothesis	mutation_summary	description	run_dir
```

Use status values:

- `keep`
- `discard`
- `crash`

## Decision Rule

Keep the run only if `weighted_cv_auc` improves by more than the configured epsilon.

Do not override the decision rule with secondary metrics.

Use the secondary metrics only to interpret the result and guide the next mutation.

## Crash Handling

If a run crashes:

1. Read the generated failure artifact in the run directory.
2. If the issue is a simple architecture bug in `train.py`, fix it and rerun as a new run.
3. Let `session_manager.py run` record the `crash` metadata and session-local results row.
4. Run `sync-incumbent` before the next mutation.
5. Finalize the session if crashes or instability make further search unproductive.

## Simplicity Rule

If two architectures perform similarly, prefer the simpler one:

- fewer layers
- fewer parameters
- less brittle code

If a more complex model does not clearly justify itself on `weighted_cv_auc`, keep the simpler incumbent and note the tradeoff in the session summary.

The point of the loop is not to make the code clever. The point is to find better architectures under a fixed evaluation harness while leaving behind a session artifact that the next agent can use directly.
