# CNN And Multimodal Refactor Plan

## Objective

Extend the fixed `autoresirch.prepare` harness so it can support:

- flat/tabular models
- convolutional models over sequence-like tensor inputs
- hybrid models that consume both tensor inputs and flat features
- optional RNA-FM embeddings extracted from `antisense_strand_seq`

without forking the entire experiment system.

The goal is not to create separate end-to-end pipelines for MLP and CNN training. The goal is to
create one shared experiment harness with modality-specific data preparation, preprocessing, and
model branches.

For the first hybrid-sequence feature, the system should support a prepare-time submodule that
extracts RNA-FM embeddings from `antisense_strand_seq`, persists them as a reusable feature
artifact, and allows each run to opt in or out of using those embeddings alongside the existing
flat feature set from the `data/` folder.

## Design Principle

Duplicate only the parts that are genuinely modality-specific:

- flat feature extraction
- tensor/sequence feature extraction
- flat preprocessing
- tensor preprocessing
- model implementations

Keep shared the parts that define the experiment boundary:

- gene-based folds
- target scaling
- training budgets
- optimizer/loss policy
- metrics
- aggregation
- run/session artifact writing

If the implementation duplicates the full training harness or orchestration layer, the system will
drift into two experiment frameworks instead of one framework with multiple input modes.

## Current Constraints

The existing code is flat-feature centric.

Current assumptions:

- [autoresirch/prepare/schemas.py](/Users/lucasplatter/sirchml-autoresearch/autoresirch/prepare/schemas.py)
  defines `PreparedDataset` with a single `features: pd.DataFrame`
- [autoresirch/prepare/fold_preprocessor.py](/Users/lucasplatter/sirchml-autoresearch/autoresirch/prepare/fold_preprocessor.py)
  assumes flat numeric/categorical preprocessing only
- [autoresirch/prepare/training_harness.py](/Users/lucasplatter/sirchml-autoresearch/autoresirch/prepare/training_harness.py)
  assumes the model receives a single tensor shaped like `[batch, input_dim]`
- [autoresirch/prepare/dataset_preparation.py](/Users/lucasplatter/sirchml-autoresearch/autoresirch/prepare/dataset_preparation.py)
  converts sequence-like columns into flattened tabular features rather than CNN-ready tensors
- [autoresirch/prepare/schemas.py](/Users/lucasplatter/sirchml-autoresearch/autoresirch/prepare/schemas.py)
  only allows `mlp` and `residual_mlp`
- there is no prepare submodule dedicated to generating reusable RNA-FM embeddings from
  `antisense_strand_seq`
- there is no run-level configuration switch that says whether RNA-FM embeddings should be included
  in the feature set for a specific experiment

These assumptions need to be relaxed, but only at the data-contract and batch-construction layers.

## What Should Be Duplicated

These should become separate submodules because the behavior is materially different across data
types.

### 1. Feature Builders

Create separate data builders for:

- flat/tabular features
- tensor/sequence features
- multimodal assembly
- RNA-FM embedding extraction from `antisense_strand_seq`

Suggested modules:

- `autoresirch/prepare/data/shared.py`
- `autoresirch/prepare/data/flat.py`
- `autoresirch/prepare/data/sequence.py`
- `autoresirch/prepare/data/rnafm.py`
- `autoresirch/prepare/data/multimodal.py`

Why duplication is appropriate:

- flat features need numeric coercion, categorical inference, one-hot expansion
- tensor features need either tokenized sequence handling or pretrained embedding extraction
- RNA-FM extraction has its own dependency, caching, and shape contract that should not be mixed
  into tabular builders
- hybrid mode needs aligned assembly, not a third independent representation

RNA-FM-specific expectations:

- source column is `antisense_strand_seq`
- output is one tensor block per row with a stable embedding shape
- extracted embeddings should be stored as a reusable prepared artifact so repeated runs do not
  recompute them unnecessarily
- multimodal assembly should include the RNA-FM block only when the run configuration enables it

### 2. Preprocessors

Split preprocessing into:

- `FlatFoldPreprocessor`
- `SequenceFoldPreprocessor` or `SequenceBatchEncoder`
- `MultiModalPreprocessor`

Suggested modules:

- `autoresirch/prepare/preprocessing/flat.py`
- `autoresirch/prepare/preprocessing/sequence.py`
- `autoresirch/prepare/preprocessing/multimodal.py`

Why duplication is appropriate:

- tabular preprocessing requires imputation, standardization, categorical level fitting
- tensor preprocessing requires token/channel mapping, fixed-length shaping, possibly masks
- hybrid preprocessing should combine already-specialized branches rather than reimplement them

### 3. Model Implementations

Create separate model modules for:

- flat models
- convolutional models
- hybrid models

Suggested modules:

- `autoresirch/prepare/models/flat.py`
- `autoresirch/prepare/models/conv.py`
- `autoresirch/prepare/models/hybrid.py`

Why duplication is appropriate:

- the internal architecture is materially different
- the forward inputs differ
- hybrid fusion logic should be explicit rather than hidden inside a giant conditional model

## What Should Be Adapted To Work For Both

These parts should not be duplicated. They should be generalized so they can support flat-only,
tensor-only, and mixed-input workflows.

### 1. `ArchitectureSpec`

Current issue:

- [autoresirch/prepare/schemas.py](/Users/lucasplatter/sirchml-autoresearch/autoresirch/prepare/schemas.py)
  defines an MLP-oriented spec with `hidden_dims`, activation, normalization, and dropout only

Required adaptation:

- support multiple model families such as:
  - `mlp`
  - `residual_mlp`
  - `cnn`
  - `hybrid_cnn_mlp`
- add a run-level feature toggle such as `use_rnafm_embeddings: bool`
- add family-specific optional fields such as:
  - `conv_channels`
  - `kernel_sizes`
  - `pooling`
  - `sequence_encoder_dim`
  - `flat_hidden_dims`
  - `fusion_hidden_dims`
- add RNA-FM-specific optional fields such as:
  - `sequence_feature_source`
  - `rnafm_embedding_dim`
  - `rnafm_pooling_strategy`

Important constraint:

- do not force CNN configuration into `hidden_dims`
- family-specific fields should be explicit
- the decision to include RNA-FM embeddings should be explicit at run start, not inferred from
  whether an embedding artifact happens to exist

### 2. `ArchitectureContext`

Current issue:

- it only exposes `input_dim`, `output_dim`, `train_size`, `feature_names`, and `device`

Required adaptation:

- support modality-aware context, such as:
  - `flat_input_dim: int | None`
  - `sequence_channels: int | None`
  - `sequence_length: int | None`
  - `sequence_embedding_dim: int | None`
  - `sequence_source_name: str | None`
  - `has_flat_features: bool`
  - `has_sequence_features: bool`
  - `flat_feature_names: tuple[str, ...]`

Reason:

- model construction should not need to infer whether the run is flat-only, sequence-only, or
  hybrid by guessing from a single scalar input dimension

### 3. `PreparedDataset`

Current issue:

- [autoresirch/prepare/schemas.py](/Users/lucasplatter/sirchml-autoresearch/autoresirch/prepare/schemas.py)
  assumes one flat feature dataframe

Required adaptation:

- refactor the contract to hold:
  - shared row metadata
  - optional flat feature payload
  - optional sequence/tensor payload

A good direction is:

- shared:
  - `target`
  - `genes`
  - `row_ids`
  - split metadata
- optional flat block:
  - flat dataframe or array source
  - flat feature names
- optional sequence block:
  - encoded tensor-ready source
  - RNA-FM embedding source metadata when enabled
  - tensor shape metadata

Reason:

- the dataset abstraction itself must become multimodal before the training harness can
  responsibly support CNNs and hybrids

### 4. `create_dataloader`

Current issue:

- [autoresirch/prepare/training_harness.py](/Users/lucasplatter/sirchml-autoresearch/autoresirch/prepare/training_harness.py)
  builds a `TensorDataset(feature_tensor, target_tensor)` from one flat feature matrix

Required adaptation:

- move to a custom dataset or structured batch representation
- batches should carry:
  - flat features if present
  - sequence tensors if present
  - target

Suggested batch shape:

```python
{
    "flat": ...,
    "sequence": ...,
    "target": ...,
}
```

Reason:

- structured batches allow one shared train loop to support all input modes

### 5. `predict_regression` and `_train_epoch`

Current issue:

- the model is always called like `model(batch_features)`

Required adaptation:

- support structured input:
  - `model(batch)`
  - or `model(flat=batch["flat"], sequence=batch["sequence"])`

Reason:

- this is the correct shared adaptation point for mixed-feature training
- do not duplicate the training loop just because the model input is no longer a single tensor

### 6. `instantiate_model`

Current issue:

- model validation uses a dummy tensor shaped like `[2, context.input_dim]`

Required adaptation:

- create family-aware dummy inputs:
  - flat dummy for flat models
  - sequence dummy for conv models
  - both for hybrid models

Reason:

- model contract validation should remain shared, but the dummy input structure must follow the
  context

### 7. Architecture Validation

Current issue:

- [autoresirch/prepare/architecture_loading.py](/Users/lucasplatter/sirchml-autoresearch/autoresirch/prepare/architecture_loading.py)
  validates only MLP-style constraints and allowed families

Required adaptation:

- support family-specific validation
- add CNN-specific checks, such as:
  - valid kernel sizes
  - valid channel counts
  - supported pooling choices
  - consistency between fusion fields and hybrid family

Reason:

- validation should prevent malformed hybrid/CNN specs before training starts

## What Should Remain Shared

These parts should remain shared across all model families and data modalities.

### 1. Split Logic

Keep shared:

- gene normalization
- train/test gene selection
- CV fold generation

Reason:

- split behavior is part of the experiment boundary and is independent of feature modality

### 2. Target Scaling

Keep shared:

- `TargetScaler`

Reason:

- target scaling depends only on the regression target

### 3. Metric Computation

Keep shared:

- prediction scaling
- AUC/RMSE/correlation computation
- fold diagnostics
- aggregation

Reason:

- metrics operate on targets and predictions, not on whether the model consumed flat or tensor
  inputs

### 4. Experiment Orchestration

Keep shared:

- [autoresirch/prepare/orchestration.py](/Users/lucasplatter/sirchml-autoresearch/autoresirch/prepare/orchestration.py)
- [autoresirch/session_manager](/Users/lucasplatter/sirchml-autoresearch/autoresirch/session_manager)

Reason:

- session artifacts, run summaries, decision logic, and budgeting should not diverge by model type

### 5. Training Policy

Keep shared:

- optimizer selection
- loss
- clipping
- budget validation

Reason:

- this is part of the fixed experiment boundary unless intentionally redefined by a human

## Hybrid Flat + Tensor Training Design

The clean multimodal flow is:

1. Read the dataframe once.
2. Build aligned row metadata once.
3. Build optional flat feature sources.
4. Build optional tensor/sequence sources.
5. Extract or load cached RNA-FM embeddings from `antisense_strand_seq` when the run enables them.
6. Store both under one `PreparedDataset`.
7. Use one shared fold index definition.
8. Preprocess each modality on the training slice only.
9. Return train/validation batches containing both modalities.
10. Let hybrid models fuse the encoded branches inside the model.

The important boundary is:

- preprocessing assembles aligned modality-specific inputs
- the model owns fusion

For the first hybrid implementation:

- RNA-FM embeddings are the input to the convolutional head
- existing flat features from the `data/` folder are the input to the flat MLP head
- the hybrid model fuses the outputs of those two heads after branch-specific encoding

Do not fuse features by flattening tensor inputs back into tabular form just to reuse the current
MLP path. That would defeat the purpose of CNN support.

## Recommended Module Layout

Suggested package structure:

```text
autoresirch/prepare/
  schemas.py
  architecture_loading.py
  orchestration.py
  training_harness.py
  data/
    shared.py
    flat.py
    sequence.py
    rnafm.py
    multimodal.py
  preprocessing/
    flat.py
    sequence.py
    multimodal.py
  models/
    flat.py
    conv.py
    hybrid.py
```

Role of each:

- `data/shared.py`
  - read raw dataframe
  - normalize genes
  - choose test genes
  - choose CV genes
- `data/flat.py`
  - infer numeric/categorical columns
  - build flat feature sources
- `data/sequence.py`
  - encode sequence inputs into tensor-ready format
- `data/rnafm.py`
  - extract or load cached RNA-FM embeddings from `antisense_strand_seq`
  - validate embedding shape and row alignment
- `data/multimodal.py`
  - assemble `PreparedDataset`
  - honor the run-level switch for including RNA-FM embeddings
- `preprocessing/flat.py`
  - fit/transform flat features
- `preprocessing/sequence.py`
  - encode/train-val transform tensor features
- `preprocessing/multimodal.py`
  - combine branch outputs into structured batches
- `models/flat.py`
  - MLP and residual MLP
- `models/conv.py`
  - CNN encoders
- `models/hybrid.py`
  - tensor encoder + flat encoder + fusion

## Places Where Duplication Would Be A Mistake

Do not duplicate:

- full training harnesses for flat and conv
- full orchestration layers for flat and conv
- metric code
- session-manager code
- fold-generation code

If those are duplicated, changes to one modality will eventually stop matching the other modality,
and results will no longer be comparable.

## File-Level Change Plan

### `autoresirch/prepare/schemas.py`

Needs adaptation:

- extend `ArchitectureSpec` to support CNN/hybrid fields
- extend `ArchitectureContext` to include modality-aware input metadata
- refactor `PreparedDataset` into a multimodal container
- widen `ArchitectureConstraints.allowed_families`

### `autoresirch/prepare/dataset_preparation.py`

Should be split:

- shared raw-data/split logic stays conceptually shared
- flat feature extraction moves to flat-specific module
- tensor feature extraction moves to sequence-specific module
- RNA-FM extraction/loading moves to its own submodule
- multimodal assembly becomes explicit

### `autoresirch/prepare/fold_preprocessor.py`

Should be split:

- current `FoldPreprocessor` becomes flat-specific
- add sequence/tensor preprocessor
- add multimodal coordinator

### `autoresirch/prepare/training_harness.py`

Should be adapted, not duplicated:

- support structured batches
- support family-aware dummy inputs
- support multimodal prediction path
- keep metric logic and budgeting shared

### `autoresirch/prepare/architecture_loading.py`

Should be adapted:

- validate new families
- validate family-specific fields
- keep load-and-contract flow shared

### `autoresirch/prepare/orchestration.py`

Should remain mostly shared:

- use new dataset contracts
- pass the run-level choice about whether RNA-FM embeddings are enabled
- keep run summaries, budgets, fold execution, and metrics common

### `autoresirch/train.py`

Will need to be simplified into a model-definition entrypoint that can instantiate:

- flat models
- CNN models
- hybrid models

depending on `ARCHITECTURE.family`

## Recommended Implementation Order

### Phase 1. Contract Refactor

Done when:

- `PreparedDataset` is multimodal-capable
- `ArchitectureContext` is modality-aware
- `ArchitectureSpec` can represent CNN and hybrid families

### Phase 2. Data Builder Split

Done when:

- flat-only dataset prep still works
- sequence/tensor builder exists independently
- RNA-FM embeddings can be extracted from `antisense_strand_seq` through a dedicated prepare
  submodule
- RNA-FM embeddings can be reused from artifacts rather than recomputed for every run
- multimodal assembly aligns rows and targets correctly

### Phase 3. Preprocessing Split

Done when:

- flat preprocessing remains stable
- sequence preprocessing produces tensor-ready batches
- multimodal preprocessing can emit structured train/validation inputs

### Phase 4. Training Harness Generalization

Done when:

- the shared train loop can train flat-only models
- the same loop can train tensor-only models
- the same loop can train hybrid models

### Phase 5. CNN And Hybrid Family Support

Done when:

- `cnn` and `hybrid_cnn_mlp` are valid families
- model validation handles them cleanly
- the run configuration can explicitly enable or disable RNA-FM embeddings
- the hybrid architecture routes RNA-FM embeddings to the convolutional branch and existing flat
  features to the MLP branch
- `autoresirch/train.py` can define and run them

### Phase 6. Test Coverage

Add tests for:

- flat-only prepared dataset
- tensor-only prepared dataset
- hybrid prepared dataset
- RNA-FM embedding extraction from `antisense_strand_seq`
- RNA-FM artifact reuse and row alignment
- run-level enabling/disabling of RNA-FM features
- row alignment across flat/tensor branches
- multimodal dataloader batch structure
- CNN family validation
- hybrid family validation
- flat path regression protection

## Acceptance Criteria

- Flat-only MLP training still produces the same type of summaries and metrics as before.
- The shared fold logic is unchanged across flat, CNN, and hybrid runs.
- The training harness accepts structured multimodal batches without duplicating the train loop.
- CNN models can consume tensor features without flattening them into tabular inputs.
- Hybrid models can consume both tensor inputs and flat features in the same batch.
- A run can explicitly opt in or out of RNA-FM embeddings at the start of execution.
- When enabled, RNA-FM embeddings are extracted from `antisense_strand_seq` and fed to the
  convolutional branch rather than flattened into tabular inputs.
- The hybrid model uses existing `data/` folder features as the flat MLP branch input.
- Session artifacts and summary files remain unchanged in structure unless a deliberate schema
  extension is required.
- Metric calculation and keep/discard logic remain shared across all model families.

## Main Risk

The largest architectural risk is trying to preserve the current `PreparedDataset.features` model as
the central abstraction and then layering CNN inputs around it. That will keep the codebase
flat-centric and force awkward special cases throughout the training stack.

The correct pivot is:

- make the dataset contract multimodal first
- make the batch contract structured second
- keep the experiment harness shared around those generalized contracts

Once those two contracts are correct, separate flat, conv, and hybrid model code becomes easy to
add without duplicating the whole system.
