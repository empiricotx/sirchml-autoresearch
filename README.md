# sirna-architecture-search

This repo is a small autonomous experimentation harness for **tabular siRNA regression**.

The design goal is strict separation of responsibilities:

- `prepare.py` is the fixed evaluation harness.
- `train.py` is the only file the agent edits.
- `program.md` is the runbook for the autonomous loop.

The agent is constrained to model architecture only. Dataset loading, sequence featurization, gene-based folds, optimization, metric definitions, time budgets, and keep/discard rules all live in `prepare.py`.

## How the repo works

1. `uv run prepare.py`
   - loads the raw siRNA dataset from `data/feature_subset_df.csv` by default
   - validates the required schema
   - uses the provided feature columns directly
   - excludes `antisense_strand_seq` from the feature matrix
   - uses `transcript_gene` for deterministic gene-held-out CV folds
   - caches the prepared dataset under `.cache/sirna_regression/`

2. `uv run train.py`
   - loads the architecture from `train.py`
   - validates that `train.py` only uses allowed imports and patterns
   - runs fixed-budget cross-validation
   - reports the primary metric `weighted_cv_rmse_mean`
   - optionally trains once on the full train split and reports a final holdout test metric

3. The autonomous agent loop in `program.md`
   - edits only `train.py`
   - runs `uv run train.py`
   - compares `weighted_cv_rmse_mean`
   - keeps the change only if the metric improves

## Primary metric

The primary metric is:

- `weighted_cv_rmse_mean`

It is computed as the weighted average of per-fold RMSE values, with each fold weighted by the number of evaluation sequences in that held-out gene.

Before metrics are computed, raw regression predictions are rescaled and clipped into `[0, 1]` using the fixed rule in [prepare.py](/Users/lucasplatter/sirchml-autoresearch/prepare.py):

```python
scaled_preds = np.clip((y_pred - 0.45) / (0.9 - 0.45), 0, 1)
```

The harness also tracks AUC using `rel_exp_individual < 0.4` as the effective class, while still selecting models by `weighted_cv_rmse_mean`.

This is intentionally **not** the same as an unweighted mean over genes.

## Required dataset columns

By default, `prepare.py` expects:

- `transcript_gene`: gene identifier used for held-out folds
- `rel_exp_individual`: regression target
- `antisense_strand_seq`: row identifier that is excluded from features

Gene identifiers are normalized to uppercase by default before split generation, so labels like `CPN1` and `Cpn1` are treated as the same held-out gene.

You can change these names in `DATASET_CONFIG` near the top of [prepare.py](/Users/lucasplatter/sirchml-autoresearch/prepare.py).

Optional columns:

- additional numeric feature columns
- additional categorical feature columns
- optional sequence columns if you later choose to engineer sequence-derived features

If `numeric_columns` or `categorical_columns` are not specified explicitly, `prepare.py` infers them from the input frame after excluding the target, gene, optional sequence, and dropped columns.

## Fixed constraints

The following are fixed in `prepare.py` unless a human changes them:

- raw dataset path and schema
- sequence featurization
- optional train/test gene split
- gene CV fold selection
- optimizer and loss
- batch size and wall-clock time budget
- primary metric and improvement threshold
- allowed architecture families and parameter limits

The agent should only search model architecture.

## Architecture contract

`train.py` must export:

```python
ARCHITECTURE = ArchitectureSpec(...)

def build_model(context: ArchitectureContext) -> nn.Module:
    ...
```

`context.input_dim` is the fixed feature dimension after fold-local preprocessing. The model must return shape `[batch]` or `[batch, 1]`.

## Quick start

```bash
# 1. Install dependencies
uv sync

# 2. Put your raw dataset at data/feature_subset_df.csv

# 3. Build and cache the prepared dataset
uv run prepare.py

# 4. Run a single fixed-budget experiment
uv run train.py

# 5. Run tests
pytest
```

## Repo structure

```text
prepare.py      fixed dataset prep, CV, metrics, constraints, training harness
train.py        editable architecture module and run entrypoint
program.md      autonomous agent instructions
data/README.md  expected raw dataset schema
tests/          focused tests for the fixed harness
```

## Notes

- Each CV fold is one held-out gene based on `transcript_gene`.
- By default there is no extra outer test split; all genes participate in gene-held-out CV.
- The selection metric is weighted cross-validation, not a random row split.
