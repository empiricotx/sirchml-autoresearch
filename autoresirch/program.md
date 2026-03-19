# Autoresearch Program

This repo runs an autonomous architecture search over fixed siRNA experiment harnesses.

The harness supports two experiment modes:

- `standard`: sequence-level regression with binary AUC-derived reporting
- `comparative`: within-gene pairwise regression with multiclass AUC-centered reporting

## Core Rule

Edit only [autoresirch/train.py](/Users/lucasplatter/sirchml-autoresearch/autoresirch/train.py) among source files during a search run.

You may also create or update generated artifacts under:

- `sessions/`
- `run.log`

Do not modify:

- [autoresirch/prepare](/Users/lucasplatter/sirchml-autoresearch/autoresirch/prepare)
- [autoresirch/session_manager](/Users/lucasplatter/sirchml-autoresearch/autoresirch/session_manager)
- [pyproject.toml](/Users/lucasplatter/sirchml-autoresearch/pyproject.toml)
- dataset files
- tests
- feature engineering
- preprocessing
- split logic
- CV logic
- optimizer settings
- time budgets
- metric definitions

Those define the experiment boundary.

## Generated Artifacts

Do not edit generated session artifacts by hand.

Use `autoresirch.session_manager` to create or update:

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

Optimize the run's configured primary metric:

- `standard`: `weighted_cv_auc`
- `comparative`: `weighted_cv_overall_auc`

Higher is better.

Do not infer the active mode. Read the current run summary, `analysis_input.json`, or `session_context.json` first.

## Session Protocol

1. Read [README.md](/Users/lucasplatter/sirchml-autoresearch/README.md), [autoresirch/program.md](/Users/lucasplatter/sirchml-autoresearch/autoresirch/program.md), and [autoresirch/train.py](/Users/lucasplatter/sirchml-autoresearch/autoresirch/train.py).
2. Confirm the cached dataset exists. If not, tell the human to run:

```bash
uv run python -m autoresirch.prepare.cli
```

3. Start a session:

```bash
uv run python -m autoresirch.session_manager.cli start \
  --experiment-mode <standard|comparative> \
  --initiated-by agent
```

The session manager stores this mode in `session_context.json`. All runs in the session must use that stored mode.

4. Run the unmodified baseline as the base run:

```bash
uv run python -m autoresirch.session_manager.cli run \
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
6. Edit only [autoresirch/train.py](/Users/lucasplatter/sirchml-autoresearch/autoresirch/train.py).
7. Run one candidate at a time:

```bash
uv run python -m autoresirch.session_manager.cli run \
  --session-id <session_id> \
  --run-role candidate \
  --parent-run-id <parent_run_id> \
  --compared-against-run-id <compared_against_run_id> \
  --hypothesis "<hypothesis>" \
  --mutation-summary "<mutation_summary>" \
  --description "<description>"
```

8. Inspect the decision plus the generated `analysis_input.json`.
9. Record agent analysis before syncing the incumbent:

```bash
uv run python -m autoresirch.session_manager.cli analyze-run \
  --session-id <session_id> \
  --run-id <run_id> \
  --summary-label "<short label>" \
  --freeform-analysis "<brief metric-grounded analysis>" \
  --likely-helped "<factor that may have helped>" \
  --likely-hurt "<factor that may have hurt>" \
  --confidence <0.0_to_1.0> \
  --next-step-reasoning "<concise next-step guidance>"
```

Rules for `analyze-run`:

- do not override the primary-metric keep/discard decision
- reference concrete metric deltas from `analysis_input.json`
- keep analysis short
- suggest at most one or two follow-up ideas

10. Restore [autoresirch/train.py](/Users/lucasplatter/sirchml-autoresearch/autoresirch/train.py) to the incumbent:

```bash
uv run python -m autoresirch.session_manager.cli sync-incumbent --session-id <session_id>
```

11. Use status when needed:

```bash
uv run python -m autoresirch.session_manager.cli status --session-id <session_id>
```

`status` reports the persisted session `experiment_mode`.

12. Finalize when done:

```bash
uv run python -m autoresirch.session_manager.cli finalize \
  --session-id <session_id> \
  --status completed \
  --end-reason "<reason>"
```

## Allowed Changes

Change only architecture behavior inside `build_model` and the declared architecture spec in [autoresirch/train.py](/Users/lucasplatter/sirchml-autoresearch/autoresirch/train.py).

Allowed examples:

- hidden widths
- depth
- residual vs plain MLP family
- activation
- normalization
- dropout
- module structure
- valid architecture-family choices allowed by the harness

## Forbidden Changes

Do not change:

- feature construction
- pair generation
- label thresholds
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
- architecture constraints
- generated session metadata by hand

## Results and Decisions

`sessions/<session_id>/results.tsv` is the canonical session ledger. Never append rows manually.

Keep a run only if its primary metric improves by more than the configured epsilon.

Use secondary metrics only to interpret the run and choose the next mutation.

Important:

- `standard` decisions are based on `weighted_cv_auc`
- `comparative` decisions are based on `weighted_cv_overall_auc`
- comparative class-specific AUCs may be undefined on some folds because class support can be absent

## Crash Handling

If a run crashes:

1. Read the generated failure artifact.
2. If the issue is a simple architecture bug in [autoresirch/train.py](/Users/lucasplatter/sirchml-autoresearch/autoresirch/train.py), fix it and rerun as a new run.
3. Let `autoresirch.session_manager` record the crash metadata.
4. Run `sync-incumbent` before the next mutation.
5. Stop the session if crashes or instability make search unproductive.

## Simplicity Rule

If two architectures perform similarly, prefer the simpler one:

- fewer parameters
- less brittle code
- fewer moving parts

Do not keep complexity that is not justified by the active mode's primary metric.
