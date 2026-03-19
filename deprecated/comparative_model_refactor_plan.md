# Comparative Model Refactor Plan

## Objective Analysis

**What:** Add a new comparative modeling mode that converts per-sequence rows into within-gene pairwise comparisons, trains with the existing leave-one-gene-out cross-validation pattern, and reports comparative classification metrics centered on AUC.

**Why:** The current harness only supports single-sequence regression. Comparative training should let the model learn relative activity differences between two sequences for the same gene, which better matches ranking-style experimental questions.

**Success looks like:** The codebase supports both standard and comparative experiments behind a shared interface, comparative data generation is deterministic and leak-free, session management continues to work, and run summaries expose overall AUC, class-specific AUC, and positive-vs-negative AUC for comparative runs.

## Current State Summary

The current package is organized around one experimental path:

- `autoresirch.prepare` owns dataset preparation, fold generation, preprocessing, training, metrics, and run summaries.
- `autoresirch.session_manager` assumes a single primary metric bundle built around `weighted_cv_auc`.
- `PreparedDataset` represents one row per input sequence.
- `build_cv_folds()` creates one validation fold per held-out gene.
- `train_fold()` optimizes regression with a single scalar target and derives binary AUC from scaled regression predictions.

This design is clean, but most types and functions still assume:

- one row equals one sequence
- one target equals absolute activity
- one evaluation recipe fits every experiment mode
- one metrics vocabulary is shared by all runs

Comparative support requires generalizing all four assumptions.

## Desired Comparative Semantics

For each gene:

1. Group all sequences belonging to that gene.
2. Generate all unique unordered pairs within the gene.
3. Pick a canonical ordering for each pair so only one example is emitted.
4. Represent the comparison as `seq1 - seq2` for every numeric feature.
5. Represent the regression target as `target(seq1) - target(seq2)`.
6. Convert that regression delta into a 3-class label:
   - `-1` when delta `< -0.2`
   - `0` when `-0.2 <= delta <= 0.2`
   - `1` when delta `> 0.2`

Cross-validation should still leave one gene out at a time. All comparison rows derived from the held-out gene must remain entirely in that fold's validation split.

## Recommended Top-Level Refactor

Restructure both `prepare` and `session_manager` into mode-aware packages:

```text
autoresirch/
  prepare/
    shared/
    standard/
      mlp/
      cnn/
    comparative/
      mlp/
      cnn/
  session_manager/
    shared/
    standard/
    comparative/
```

Recommended interpretation of these layers:

- `shared`: config types, common utilities, generic orchestration, persistence helpers, and dispatch code.
- `standard`: the existing absolute-activity workflow, moved with minimal behavior change.
- `comparative`: new pairwise data generation, targets, metrics, and summaries.
- `mlp` / `cnn`: model-family-specific builders, validators, and context helpers.

This should be treated as a packaging refactor first, not a behavior rewrite first. Preserve existing standard behavior while introducing the new mode behind explicit dispatch.

## Target Architecture

### 1. Shared Experiment Mode Layer

Introduce an explicit experiment-mode concept in shared schemas/config:

- `experiment_mode: Literal["standard", "comparative"]`
- optional `task_type` or `label_mode` if needed later
- mode-aware metric selection
- mode-aware summary formatting

Recommended new shared abstractions:

- `PreparedExampleSet`: common protocol/base type for any prepared dataset
- `FoldSpec`: keep shared, because leave-one-gene-out still applies
- `MetricBundle`: mode-specific metric output normalized into a shared summary format
- `ExperimentDriver`: dispatches to standard or comparative implementations

The key design rule is that orchestration should not know how rows are built. It should only know how to ask a mode-specific module for:

- prepared data
- folds
- training/evaluation behavior
- aggregate metrics
- summary payload

### 2. Standard Mode Migration

Move the current logic into `standard` modules with behavior preserved:

- current dataset preparation becomes `prepare/standard/...`
- current fold preprocessing stays largely unchanged
- current regression metrics remain intact
- current summary fields remain valid for backward compatibility

This isolates risk. Standard mode should remain the control path used to verify the refactor did not regress existing behavior.

### 3. Comparative Mode Data Pipeline

Build a comparative dataset constructor that runs after raw sequence-level data loading and gene normalization.

Recommended flow:

1. Load and validate the raw per-sequence frame exactly as standard mode does.
2. Infer/select usable source features before pair generation.
3. Partition genes into train/test/cv at the sequence level, not after pairing.
4. For each gene independently, generate pairwise comparison rows.
5. Concatenate all gene-local comparison rows into a `ComparativePreparedDataset`.

Recommended canonical pair generation:

- For a gene with rows indexed `[i0, i1, i2, ...]`, emit only pairs where `left_index < right_index`.
- Persist source row ids for traceability, for example:
  - `left_row_id`
  - `right_row_id`
  - `pair_id`
  - `gene`

This prevents duplicate comparisons and keeps the rule deterministic.

### 4. Comparative Feature Construction

Numeric/tabular features:

- Build comparison features as `left_feature - right_feature`.
- Keep feature names explicit, for example `delta::<feature_name>`.

Categorical features:

- Do not silently subtract encoded categories after fold preprocessing without a design choice.
- Recommended approach: one-hot encode both sides fold-locally, then subtract the encoded vectors to produce `{-1, 0, 1}` deltas.
- Alternative: drop categorical columns in comparative mode initially if they add too much complexity, but only if documented explicitly.

Sequence-derived features:

- For engineered sequence tabular features, the same subtraction rule applies after feature generation.
- For CNN or hybrid comparative models, decide whether the model consumes:
  - only delta features/tensors, or
  - both left/right branches with shared encoders and a subtraction layer

Recommended model-path rule:

- `comparative/mlp`: operate on precomputed delta tabular features.
- `comparative/cnn`: use siamese/shared encoders over left and right sequence tensors, then subtract latent representations before the prediction head.
- `comparative/hybrid`: fuse delta flat features with `left-right` sequence encodings.

Even if the first implementation only lands MLP cleanly, the module boundaries should reserve this path now.

### 5. Comparative Target and Labels

Store both:

- `target_delta`: continuous `left_target - right_target`
- `target_class`: discrete label in `{-1, 0, 1}`

Recommended threshold config:

- `no_effect_lower = -0.2`
- `no_effect_upper = 0.2`

Centralize this in comparative metric config so the same thresholds are used in:

- dataset construction
- training labels for evaluation
- holdout/test summaries
- session analysis copy

## Comparative Training Strategy

### Model Output

Use the same training pattern as today for the first version:

- predict one scalar regression delta
- optimize MSE on `target_delta`
- derive classes from prediction deltas using the same thresholds as the ground truth

This keeps training close to the current harness while enabling richer comparative evaluation.

Future extension:

- allow direct 3-class classification heads
- allow joint regression + classification loss

That should not be part of the first refactor unless implementation remains small.

### Fold Construction

Keep leave-one-gene-out CV exactly aligned with current semantics:

- each validation fold contains all comparison rows for one held-out gene
- training folds contain comparison rows from all remaining train genes
- no pair may combine rows from different genes

This means the comparative dataset builder should preserve a one-to-many mapping from gene to emitted pair rows so fold creation stays trivial.

## Comparative Metrics and Reporting

Comparative runs need a new metric vocabulary. Recommended metric bundle:

- `weighted_cv_overall_auc`
- `weighted_cv_auc_class_neg1`
- `weighted_cv_auc_class_0`
- `weighted_cv_auc_class_pos1`
- `weighted_cv_auc_pos_vs_neg`
- `weighted_cv_rmse_mean`
- `weighted_cv_mae_mean`
- `weighted_cv_pearson_r_mean`
- `weighted_cv_spearman_r_mean`

`weighted_cv_overall_auc` should be the primary metric for comparative mode.

Definitions:

- `overall_auc`: multiclass AUC over `{-1, 0, 1}` using one-vs-rest or one-vs-one, chosen explicitly and documented.
- `class_specific_auc`: one-vs-rest AUC for each class.
- `pos_vs_neg_auc`: binary AUC computed after excluding class `0`.

Recommended implementation choice:

- use one-vs-rest for class-specific AUCs
- use macro-average one-vs-rest multiclass AUC for overall AUC
- compute `pos_vs_neg_auc` on the filtered subset where true class is `-1` or `1`

Important edge-case handling:

- Some folds may not contain all three classes.
- Some held-out genes may have only near-zero comparisons or only one signed direction.
- AUC values should become `None` when the required class support is absent.
- Weighted aggregation should ignore undefined fold metrics and report how many folds were undefined.

## Summary and Session-Manager Changes

The current session layer is tightly coupled to standard metric names. Comparative mode needs explicit mode-aware session analysis.

### Required schema changes

Update run/session records to capture at least:

- `experiment_mode`
- `primary_metric_name`
- comparative metric fields in summaries
- possibly `label_thresholds` for reproducibility

### Required session-manager refactor

Split `session_manager` into:

- shared persistence/orchestration
- standard analysis/reporting
- comparative analysis/reporting

Specific impact areas:

- `SESSION_RESULTS_HEADER` is currently standard-specific and should become mode-aware.
- `INTERPRETATION_METRIC_SPECS` and metric ordering need separate standard vs comparative definitions.
- decision logic should compare whichever metric is primary for the active mode.
- `analysis_input.json` should include comparative metrics and class-support diagnostics.
- `synopsis.md` generation should explain undefined AUCs caused by class absence.

Recommended primary metric for comparative mode:

- `weighted_cv_overall_auc`

Keep this fixed as the comparative incumbent-selection metric so session decisions remain consistent across runs.

## Proposed Module Breakdown

### `autoresirch.prepare.shared`

Responsibility:

- shared config and schema definitions
- raw dataframe reading
- gene normalization and gene splitting
- common fold utilities
- orchestration dispatch
- generic run summary persistence

Likely modules:

- `schemas.py`
- `config.py`
- `splits.py`
- `orchestration.py`
- `summary.py`
- `metrics_common.py`

### `autoresirch.prepare.standard`

Responsibility:

- current dataset preparation flow
- current fold preprocessing
- current regression evaluation
- standard train/final-fit paths

Likely modules:

- `dataset.py`
- `preprocessing.py`
- `metrics.py`
- `training.py`
- `families/mlp.py`
- `families/cnn.py`

### `autoresirch.prepare.comparative`

Responsibility:

- pair generation
- comparative feature encoding
- comparative target + class assignment
- comparative metrics
- comparative training/evaluation

Likely modules:

- `dataset.py`
- `pair_generation.py`
- `preprocessing.py`
- `metrics.py`
- `training.py`
- `families/mlp.py`
- `families/cnn.py`

### `autoresirch.session_manager.shared`

Responsibility:

- session lifecycle
- storage
- common decision recording
- common run context and summary loading

### `autoresirch.session_manager.standard`

Responsibility:

- existing metric interpretation and synopsis generation

### `autoresirch.session_manager.comparative`

Responsibility:

- comparative metric interpretation
- class-support diagnostics
- comparative synopsis text

## Implementation Phases

### Phase 1: Introduce Mode-Aware Shared Contracts

Scope:

- add experiment-mode config and dispatch
- move existing standard logic under `standard`
- keep root-level imports compatible

Acceptance criteria:

- `prepare.py` and existing imports still work
- standard tests still pass unchanged
- no behavior change for existing runs

### Phase 2: Build Comparative Dataset Generation

Scope:

- comparative prepared dataset type
- unique within-gene pair generation
- delta feature construction
- delta target and 3-class label assignment

Acceptance criteria:

- each gene emits `n * (n - 1) / 2` rows for `n` sequences
- no duplicate inverse pairs are emitted
- target/class thresholds are deterministic and configurable
- pair lineage is preserved in the prepared dataset

### Phase 3: Add Comparative Folding, Training, and Metrics

Scope:

- comparative fold builder or shared fold builder reuse
- regression-on-delta training path
- multiclass and filtered binary AUC computation
- fold aggregation and summary generation

Acceptance criteria:

- held-out gene folds contain only that gene's comparison rows
- comparative summaries report overall/class-specific/pos-vs-neg AUCs
- comparative summaries report weighted Pearson and Spearman correlation metrics
- undefined AUC cases are handled without crashing

### Phase 4: Refactor Session Manager for Mode-Aware Analysis

Scope:

- mode-aware result headers
- comparative decision/reporting payloads
- comparative synopsis generation

Acceptance criteria:

- session runs can record comparative summaries without schema hacks
- incumbent decisions use the configured comparative primary metric
- analysis artifacts remain readable for both modes

### Phase 5: Expand Architecture Family Support

Scope:

- standard/comparative family-specific modules
- consistent builder interfaces across MLP/CNN/hybrid

Acceptance criteria:

- architecture validation is mode-aware
- comparative MLP path is fully functional
- comparative CNN/hybrid interfaces are scaffolded, even if some implementations are deferred

## Testing Plan

Add tests in layers.

### Unit tests

- gene-local pair generation count and uniqueness
- canonical pair ordering
- delta target calculation
- 3-class threshold mapping
- comparative fold generation
- one-vs-rest AUC helpers
- positive-vs-negative filtering behavior

### Integration tests

- comparative prepared dataset from a small synthetic frame
- one complete comparative CV run with deterministic mock architecture
- summary serialization/deserialization
- session-manager run artifacts for comparative mode

### Regression tests

- current standard `tests/test_prepare.py` still passing
- current standard `tests/test_session_manager.py` still passing
- backward-compatible imports from `prepare` and `session_manager`
- comparative metric tests verify Pearson and Spearman are always computed and serialized when fold support exists

## Data and Metric Edge Cases

These need explicit handling in code and tests:

- genes with fewer than 2 sequences produce zero pairs
- genes with only 2 sequences produce exactly 1 pair
- folds with only one present class make some AUCs undefined
- filtered `pos_vs_neg` subsets can be empty
- delta feature subtraction can amplify missing-data problems if imputations are not fold-local
- categorical feature subtraction must not leak validation-only categories into training encodings

Recommended rule for low-pair genes:

- either drop genes with fewer than 2 sequences during dataset build and report them
- or fail fast if any configured CV/test gene yields zero comparisons

The safer first implementation is to fail fast with a clear message.

## Risks and Mitigations

### Risk: Pair-count explosion

Genes with many sequences create quadratic growth in rows.

Mitigations:

- log per-gene pair counts in dataset summaries
- add optional caps or sampling later, but not in the first implementation
- review training budget defaults after comparative mode lands

### Risk: Metric instability on sparse classes

Comparative thresholds can create folds where one or more classes disappear.

Mitigations:

- treat unsupported AUCs as undefined, not zero
- track class counts per fold in diagnostics
- surface undefined metric counts in summaries and session analysis

### Risk: Over-coupled family abstractions

Trying to fully generalize MLP, CNN, and hybrid models before comparative data is stable will slow delivery.

Mitigations:

- make comparative MLP the first production path
- scaffold CNN/hybrid interfaces behind the same context contract
- defer full comparative sequence-branch training until the delta-data path is verified

## Recommended Delivery Order

1. Refactor package layout while preserving current standard behavior.
2. Add experiment-mode config and dispatch plumbing.
3. Implement comparative pair generation and dataset schemas.
4. Land comparative MLP training and metrics end to end.
5. Update session-manager schemas and comparative reporting.
6. Add CNN/hybrid comparative support on top of the stabilized shared contracts.

## Concrete Deliverables

The refactor should produce:

- a mode-aware `prepare` package with `shared`, `standard`, and `comparative`
- a mode-aware `session_manager` package with `shared`, `standard`, and `comparative`
- comparative dataset and metric schemas
- comparative run summaries with overall/class-specific/pos-vs-neg AUC plus required Pearson/Spearman metrics
- tests proving pair uniqueness, target subtraction, thresholding, fold isolation, and session artifact compatibility
- updated README and operator docs once implementation starts

## Open Design Decisions To Resolve During Implementation

- whether categorical features are supported in v1 comparative mode or deferred
- whether comparative CNN input should use raw left/right branches or precomputed delta sequence features
- whether genes with fewer than 2 sequences should fail fast or be dropped with reporting
- whether comparative mode should expose holdout-test metrics immediately or only after CV is stable

## Recommended Default Decisions

If you want the implementation to move quickly with minimal ambiguity, use these defaults:

- primary metric: `weighted_cv_overall_auc`
- training objective: regression on `target_delta`
- class derivation: threshold predicted delta at `[-0.2, 0.2]`
- v1 architecture path: comparative MLP first, comparative CNN/hybrid scaffolded but optional
- genes with fewer than 2 sequences: fail fast
- categorical features: support through fold-local one-hot subtraction if straightforward, otherwise defer explicitly rather than handling them implicitly
- Pearson and Spearman correlation: required comparative evaluation metrics alongside AUC and error metrics

## Structural Outline

### Target top-level repo structure

```text
prepare.py
session_manager.py
autoresirch/
  train.py
  program.md
  prepare/
    __init__.py
    cli.py
    shared/
      __init__.py
      schemas.py
      config.py
      dataframe.py
      splits.py
      summary.py
      orchestration.py
      metrics_common.py
      runtime.py
    standard/
      __init__.py
      dataset.py
      preprocessing.py
      metrics.py
      training.py
      mlp/
        __init__.py
        builders.py
      cnn/
        __init__.py
        builders.py
    comparative/
      __init__.py
      dataset.py
      pair_generation.py
      preprocessing.py
      metrics.py
      training.py
      mlp/
        __init__.py
        builders.py
      cnn/
        __init__.py
        builders.py
  session_manager/
    __init__.py
    cli.py
    shared/
      __init__.py
      schemas.py
      constants.py
      storage.py
      orchestration.py
      analysis_io.py
    standard/
      __init__.py
      analysis.py
      reporting.py
    comparative/
      __init__.py
      analysis.py
      reporting.py
tests/
  test_prepare_standard.py
  test_prepare_comparative.py
  test_session_manager_standard.py
  test_session_manager_comparative.py
documentation/
  comparative_model_refactor_plan.md
```

### File-by-file target responsibilities

`prepare.py`

- Thin compatibility shim.
- Re-export the public `autoresirch.prepare` API.

`session_manager.py`

- Thin compatibility shim.
- Re-export the public `autoresirch.session_manager` API.

`autoresirch/train.py`

- User-editable architecture entrypoint.
- Exports architecture specs/builders chosen by the active experiment mode and family.

`autoresirch/program.md`

- Agent workflow instructions.
- Updated to mention mode-aware experiment execution and comparative metric interpretation.

`autoresirch/prepare/__init__.py`

- Public exports for shared, standard, and comparative prepare APIs.
- Mode-dispatch helpers exposed at package level.

`autoresirch/prepare/cli.py`

- CLI entrypoint.
- Parses experiment mode, family, data path overrides, and run options.
- Dispatches into shared orchestration.

`autoresirch/prepare/shared/schemas.py`

- Core dataclasses and protocols used by both modes.
- `DatasetConfig`, `SplitConfig`, `TrainingConfig`, metric config types, architecture context/spec types, prepared dataset base types, fold types, summary types.
- Shared enum/literal values like `experiment_mode`.

`autoresirch/prepare/shared/config.py`

- Default config instances.
- Standard and comparative metric defaults.
- Comparative thresholds including `[-0.2, 0.2]`.

`autoresirch/prepare/shared/dataframe.py`

- Raw dataframe reading and schema validation.
- Gene normalization and generic feature-column inference utilities.

`autoresirch/prepare/shared/splits.py`

- Train/test gene selection.
- Leave-one-gene-out fold construction for any prepared dataset carrying gene labels.

`autoresirch/prepare/shared/summary.py`

- Run-summary serialization.
- Mode-aware summary payload building.
- Shared printing helpers.

`autoresirch/prepare/shared/orchestration.py`

- Main experiment driver.
- Loads architecture, resolves mode and family, prepares data, builds folds, trains/evaluates each fold, aggregates metrics, writes summaries.

`autoresirch/prepare/shared/metrics_common.py`

- Generic math helpers used by both modes.
- RMSE, MAE, R2, Pearson, Spearman, weighting helpers, safe undefined-metric handling.

`autoresirch/prepare/shared/runtime.py`

- Runtime directories, caching helpers, config fingerprints, generic utility functions now spread across `utils.py`.

`autoresirch/prepare/standard/__init__.py`

- Standard-mode exports.

`autoresirch/prepare/standard/dataset.py`

- Current single-row-per-sequence preparation path.
- Builds the standard prepared dataset without behavioral change.

`autoresirch/prepare/standard/preprocessing.py`

- Fold-local numeric imputation, scaling, categorical encoding.
- Existing target scaling for standard regression.

`autoresirch/prepare/standard/metrics.py`

- Current standard regression metrics.
- Binary AUC derived from scaled regression predictions.

`autoresirch/prepare/standard/training.py`

- Standard fold training, final-fit evaluation, dataloaders, model invocation, aggregation.

`autoresirch/prepare/standard/mlp/builders.py`

- Standard MLP-family model builders and any family-specific validation helpers.

`autoresirch/prepare/standard/cnn/builders.py`

- Standard CNN and hybrid sequence-capable builders.

`autoresirch/prepare/comparative/__init__.py`

- Comparative-mode exports.

`autoresirch/prepare/comparative/dataset.py`

- Builds the comparative prepared dataset from raw per-sequence rows.
- Stores pair ids, left/right row ids, gene ids, delta targets, and class labels.

`autoresirch/prepare/comparative/pair_generation.py`

- Unique within-gene pair enumeration.
- Canonical left/right ordering.
- Pair-count validation and low-support checks.

`autoresirch/prepare/comparative/preprocessing.py`

- Fold-local comparative feature building.
- Standardizes or encodes source features, then forms comparative deltas.
- Supports categorical one-hot subtraction if retained in v1.

`autoresirch/prepare/comparative/metrics.py`

- Comparative regression and classification evaluation.
- Required metrics:
  - `weighted_cv_overall_auc`
  - `weighted_cv_auc_class_neg1`
  - `weighted_cv_auc_class_0`
  - `weighted_cv_auc_class_pos1`
  - `weighted_cv_auc_pos_vs_neg`
  - `weighted_cv_rmse_mean`
  - `weighted_cv_mae_mean`
  - `weighted_cv_pearson_r_mean`
  - `weighted_cv_spearman_r_mean`
- Handles undefined class-support cases cleanly.

`autoresirch/prepare/comparative/training.py`

- Comparative dataloaders and model execution.
- Predicts continuous delta targets.
- Converts predictions to classes using shared comparative thresholds.
- Aggregates fold metrics with `weighted_cv_overall_auc` as the primary metric.

`autoresirch/prepare/comparative/mlp/builders.py`

- Comparative MLP builders operating on delta flat features.

`autoresirch/prepare/comparative/cnn/builders.py`

- Comparative CNN/hybrid builders using left/right branches and subtraction in latent space.
- May be scaffold-first if not fully implemented in v1.

`autoresirch/session_manager/__init__.py`

- Public exports for session manager shared/standard/comparative APIs.

`autoresirch/session_manager/cli.py`

- Session CLI entrypoint.
- Accepts experiment mode and routes analysis/reporting to the correct implementation.

`autoresirch/session_manager/shared/schemas.py`

- Session context, run context, decision records, analysis payloads, and summaries.
- Adds `experiment_mode`, mode-aware primary metric fields, and comparative metric bundles.

`autoresirch/session_manager/shared/constants.py`

- Shared paths, artifact names, and generic constants.
- Keeps standard and comparative metric-order definitions separate.

`autoresirch/session_manager/shared/storage.py`

- Read/write JSON, TSV, snapshots, session directories, and summary files.
- Handles mode-aware result ledgers.

`autoresirch/session_manager/shared/orchestration.py`

- Session lifecycle and run execution.
- Calls `prepare` orchestration, records decisions, updates incumbent state using the active mode's primary metric.

`autoresirch/session_manager/shared/analysis_io.py`

- Shared helpers for constructing analysis input payloads and reading summary artifacts.

`autoresirch/session_manager/standard/analysis.py`

- Existing metric comparison logic for standard runs.

`autoresirch/session_manager/standard/reporting.py`

- Standard synopsis and session summary rendering.

`autoresirch/session_manager/comparative/analysis.py`

- Comparative metric interpretation.
- Builds analysis bundles including overall AUC, class-specific AUCs, pos-vs-neg AUC, Pearson, and Spearman.
- Explains undefined metrics caused by missing class support.

`autoresirch/session_manager/comparative/reporting.py`

- Comparative run synopsis and session summary rendering.
- Highlights whether changes improved overall relative ranking quality and signed-difference fidelity.

`tests/test_prepare_standard.py`

- Preserved current standard prepare coverage.

`tests/test_prepare_comparative.py`

- Comparative dataset, pair generation, fold isolation, thresholding, metrics, and summary tests.

`tests/test_session_manager_standard.py`

- Preserved current standard session-manager coverage.

`tests/test_session_manager_comparative.py`

- Comparative session artifacts, comparative incumbent selection, and analysis/reporting tests.

### Planned workflow

1. `uv run prepare.py --mode standard|comparative --family mlp|cnn|hybrid`
   - CLI parses mode/family.
   - Shared orchestration loads `train.py` architecture definitions.
   - Shared orchestration dispatches to `standard` or `comparative` dataset and training modules.

2. Data preparation
   - `shared/dataframe.py` loads raw data and normalizes genes.
   - `shared/splits.py` determines train/test/CV genes.
   - `standard/dataset.py` builds sequence-level examples or `comparative/dataset.py` builds within-gene pairwise examples.

3. Fold-local preprocessing
   - Standard mode preprocesses rows directly.
   - Comparative mode encodes source rows fold-locally, then subtracts left/right feature blocks into delta inputs.

4. Model training and evaluation
   - Family-specific builders create the requested model.
   - Standard mode predicts absolute activity.
   - Comparative mode predicts activity deltas and derives 3-class labels from those predictions.

5. Metric aggregation
   - Standard mode reports its current regression/AUC bundle.
   - Comparative mode reports overall AUC, class-specific AUCs, pos-vs-neg AUC, RMSE, MAE, Pearson, and Spearman.
   - `weighted_cv_overall_auc` is the comparative primary metric.

6. Summary persistence
   - Shared summary code writes `summary.json` and latest-summary artifacts with mode-aware payloads.

7. Session execution
   - `uv run python session_manager.py run ... --mode standard|comparative`
   - Shared session orchestration executes the run, loads the correct metric bundle, and compares against the incumbent using the active primary metric.

8. Analysis and reporting
   - Standard runs use `session_manager/standard/*`.
   - Comparative runs use `session_manager/comparative/*`.
   - Comparative reports explain movement in overall AUC, class-specific separability, pos-vs-neg ranking, and Pearson/Spearman correlation.
