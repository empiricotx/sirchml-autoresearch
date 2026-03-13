# Regression Autoresearch Program

This repo is an autonomous search loop for **siRNA regression architectures**.

## Core rule

You may edit **only** `train.py`.

Do not modify:

- `prepare.py`
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

## Objective

Minimize `weighted_cv_rmse_mean`.

Lower is better.

The metric is the weighted mean of per-gene validation RMSE values, weighted by the number of evaluation sequences in each held-out gene fold.

## Experiment protocol

1. Read `README.md`, `prepare.py`, and `train.py`.
2. Confirm the cached dataset exists. If not, tell the human to run `uv run prepare.py`.
3. Initialize `results.tsv` if it is missing.
4. Run the unmodified baseline:

```bash
uv run train.py > run.log 2>&1
```

5. Extract the metric from `run.log`.
6. Log the result in `results.tsv`.
7. Edit only `train.py`.
8. Run one experiment at a time.
9. Keep the commit only if `weighted_cv_rmse_mean` improves by more than the configured epsilon.
10. If the score is worse or unchanged, revert the `train.py` change.

## What you are allowed to change

Only architecture:

- hidden layer widths
- depth
- residual vs plain MLP family
- activation choice
- normalization choice
- dropout amount
- module structure inside `build_model`

## What you are not allowed to change

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

## Result logging

Use `results.tsv` with the header:

```text
commit	weighted_cv_rmse_mean	cv_rmse_std	weighted_cv_auc	status	num_params	train_seconds	description
```

Use status values:

- `keep`
- `discard`
- `crash`

## Crash handling

If the run crashes:

1. Read `run.log`.
2. If the issue is a simple architecture bug in `train.py`, fix it and rerun.
3. If the idea itself is broken, log a `crash` result and move on.

## Simplicity rule

If two architectures perform similarly, prefer the simpler one:

- fewer layers
- fewer parameters
- less brittle code

The point of the loop is not to make the code clever. The point is to find better architectures under a fixed evaluation harness.
