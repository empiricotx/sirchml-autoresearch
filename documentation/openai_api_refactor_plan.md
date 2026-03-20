# OpenAI API Session Refactor Plan

## Objective Analysis

**What:** Refactor the autonomous workflow so session execution is launched from the command line, data preparation remains local and deterministic, and OpenAI API calls replace Codex/Claude as the model-generation and run-interpretation layer.

**Why:** The current design assumes an external coding agent edits `autoresirch/train.py` under `program.md` constraints. That makes orchestration depend on a tool-specific interactive loop. Moving to explicit OpenAI API calls should make sessions reproducible, scriptable, and easier to extend with multiple search strategies.

**Who:** The primary user is the researcher launching search sessions from the CLI and reviewing session artifacts afterward.

**Success looks like:** A user can start a session with explicit launch parameters, the system prepares data once, the orchestration layer uses OpenAI API calls to propose model definitions and generate interpretation artifacts, and the session can run in one of three agent modes: `search`, `refinement`, or `sandbox`.

## Current State Summary

The current codebase is already close to the desired separation, but one layer is still agent-centric:

- `autoresirch.prepare` owns dataset preparation, caching, fold generation, training, and metric computation.
- `autoresirch.session_manager` owns session state, run artifacts, keep/discard logic, and session summaries.
- `autoresirch/train.py` is the single mutable architecture module.
- `autoresirch/program.md` assumes an autonomous coding agent edits `train.py`, runs experiments, records analysis, and syncs the incumbent.

Important existing constraints:

- `session_manager start` already persists `experiment_mode`, `raw_data_path`, dataset fingerprints, and prepared-data paths.
- RNA-FM support already exists in the preparation path and architecture contract.
- Feature engineering is explicitly forbidden by the current agent program, so `sandbox` mode will require a deliberate expansion of the experiment boundary.

## Desired User Experience

### CLI launch contract

The user should be able to start a session with one command that fully defines the run envelope:

- session mode
- dataset to use
- whether RNA-FM embeddings are enabled
- number of iterations
- model family or model-type constraint
- agent mode: `search`, `refinement`, or `sandbox`

Recommended launch shape:

```bash
uv run python -m autoresirch.session_manager.cli start \
  --experiment-mode <standard|comparative> \
  --dataset <dataset_name_or_path> \
  --use-rnafm <true|false> \
  --iterations <n> \
  --model-type <mlp|residual_mlp|cnn|hybrid_cnn_mlp|auto> \
  --agent-mode <search|refinement|sandbox>
```

### Session lifecycle

1. User launches a session from the CLI.
2. The system resolves the dataset and runs local preparation.
3. The session manager writes a fully specified session context.
4. For each iteration, the orchestration layer builds an API prompt from the session context plus run history.
5. OpenAI returns either a proposed architecture definition, a structured mutation, or both.
6. The local harness runs training and evaluation exactly as it does today.
7. OpenAI is called again to generate the interpretation artifact for the run.
8. The session manager applies the existing metric-based keep/discard rule and records all artifacts.

The API should not become the execution engine for training, dataset handling, or decision rules. Those should remain local.

## Core Design Principles

1. Keep `prepare` local, deterministic, and human-auditable.
2. Keep metric calculation and keep/discard decisions local.
3. Use OpenAI only for proposal generation and interpretation.
4. Make agent behavior explicit through typed session configuration, not hidden prompt variations.
5. Preserve backward compatibility where possible so current standard and comparative harnesses still work during the migration.
6. Treat agents as decision-makers, not as the place where business logic lives.
7. Treat tools as deterministic executors with narrow contracts.
8. Prefer explicit workflow orchestration over giant prompts that implicitly encode control flow.
9. Compose multi-agent behavior from small reusable patterns such as agents-as-tools, routers, and parallel workers.
10. Separate reasoning, execution, and coordination into different modules and different persisted artifacts.

## Layer Model

The refactor should be organized around three explicit layers:

### Reasoning layer

This is where the OpenAI-powered agents live. Their job is to decide what to try next, not to perform file I/O, training, or state mutation directly.

Recommended responsibilities:

- choose the next candidate architecture or candidate model plan
- interpret prior run results
- choose among approved feature-transform options in `sandbox`
- explain why a proposed mutation is worth trying

### Execution layer

This is where tools live. Tools should be deterministic and side-effectful in well-defined ways.

Recommended responsibilities:

- prepare datasets
- validate candidate payloads
- materialize `train_candidate.py`
- execute training and evaluation
- write session artifacts
- compute summaries and decisions

### Coordination layer

This is the workflow layer that sequences reasoning and execution.

Recommended responsibilities:

- create sessions
- decide when to call which agent
- route between `search`, `refinement`, and `sandbox` strategies
- fan out or serialize work when parallelism is allowed
- enforce iteration budgets and stopping conditions
- persist workflow state independently from agent responses

## Proposed Top-Level Refactor

### 1. Add a first-class session launch config

Introduce a typed session launch/config model that extends the current `SessionContext`.

Recommended additions:

- `agent_mode: Literal["search", "refinement", "sandbox"]`
- `iteration_budget: int`
- `model_type_constraint: str`
- `include_rnafm_embeddings: bool`
- `dataset_selector: str`
- `openai_model: str`
- `openai_reasoning_model: str | None`
- `feature_engineering_policy: Literal["disabled", "fixed", "agentic"]`

This config should be persisted into `session_context.json` so every run is reproducible from the session artifact alone.

### 2. Replace the editable-agent loop with an explicit workflow stack

The current workflow assumes `autoresirch/train.py` is manually or agentically edited before each run. Replace that with a local proposal engine that uses OpenAI responses to generate run candidates.

Recommended new package:

```text
autoresirch/
  session_manager/
    workflows/
      launch.py
      iteration.py
      routing.py
    agent_modes/
      search.py
      refinement.py
      sandbox.py
    openai/
      client.py
      prompts.py
      schemas.py
      reasoning.py
    tools/
      candidate_materializer.py
      candidate_validator.py
      run_executor.py
      artifact_writer.py
```

Responsibilities:

- `client.py`: authenticated OpenAI API wrapper with retries, logging, and response normalization.
- `prompts.py`: prompt builders for architecture proposals and run interpretation.
- `schemas.py`: strict request/response dataclasses or Pydantic models for proposal payloads.
- `reasoning.py`: thin OpenAI-facing layer for proposal and interpretation calls.
- `agent_modes/*.py`: mode-specific reasoning policies and prompt-shaping logic.
- `workflows/*.py`: explicit orchestration that decides which agent to call, which tools to run, and how state advances.
- `tools/*.py`: deterministic executors that materialize candidates, validate them, run training, and write artifacts.

This layering matters because:

- agents should decide
- tools should execute
- workflows should coordinate

The plan should avoid embedding workflow state transitions inside prompts.

### 3. Replace raw code editing with run-local candidate modules

The current system mutates Python source in `autoresirch/train.py`. That is the weakest part of the present workflow because it couples search to source rewriting and makes iteration state harder to audit.

Recommended target state for all modes:

- the API never edits `train.py` directly
- each run materializes a generated run-local module such as `train_candidate.py`
- the harness executes that generated file for the run
- the exact candidate source is stored with the run artifacts

Mode-specific generation rules:

- `search`: OpenAI returns a structured payload that is effectively limited to `ArchitectureSpec` fields and rationale
- `refinement`: same as `search`, but with smaller, incumbent-aware `ArchitectureSpec` mutations
- `sandbox`: OpenAI may return a richer candidate payload or candidate source plan, and the local materializer may generate a less templated `train_candidate.py` so long as it still satisfies the harness contract

Best near-term implementation:

- keep `ArchitectureSpec` as the core contract for `search` and `refinement`
- generate `train_candidate.py` from stable templates for those two modes
- allow `sandbox` to generate a broader run-local candidate module, but still validate imports, output shapes, and runtime compatibility before execution

This preserves auditability across all modes while still giving `sandbox` much more room to explore.

### 4. Add a feature-engineering module under `prepare`

`sandbox` mode requires a controlled way for the system to propose feature work. That should not be bolted into the agent layer.

Recommended new module structure:

```text
autoresirch/
  prepare/
    feature_engineering/
      __init__.py
      registry.py
      schemas.py
      transforms.py
      sequence.py
      tabular.py
```

Recommended design:

- feature engineering is represented as a structured transform plan, not arbitrary code
- transforms are local, pre-registered, deterministic functions
- session config decides whether feature engineering is unavailable, fixed, or agent-selectable

Example transform families:

- sequence-derived GC/content statistics
- positional nucleotide counts
- k-mer summaries
- transcript-level aggregations if already leak-safe
- feature selection masks
- feature crosses for safe tabular columns

The key constraint is that `sandbox` mode may choose among approved transforms, but should not execute arbitrary user-invisible code generated by the model.

## Agent Mode Semantics

### Search mode

Goal: broad topological exploration of the selected model family.

Behavior:

- sample widely across legal architecture shapes
- prefer diversity over local optimization
- limit per-run changes to `ArchitectureSpec` fields such as family, depth, width, normalization, dropout, and RNA-FM usage within the session constraints
- avoid feature-engineering changes unless explicitly enabled

Recommended search policy:

- start from baseline plus several distant candidates
- penalize repeated architectures
- preserve a simple novelty ledger in session state
- materialize each proposal through a stable `ArchitectureSpec` -> `train_candidate.py` template path
- treat `search` as an agent that can be called by the workflow as a reusable decision tool

### Refinement mode

Goal: narrow in on the best-performing local region.

Behavior:

- anchor search around the incumbent or top-k runs
- make smaller mutations than `search` mode
- bias toward changes with prior support in session history
- explicitly use prior metric deltas and failure patterns in the prompt
- keep proposals constrained to `ArchitectureSpec`-level edits rather than arbitrary Python changes

Recommended refinement policy:

- mutate one or two architectural dimensions at a time
- use tie-break and near-miss information from prior runs
- allow selective reruns when instability is suspected
- materialize each proposal through the same stable template path used by `search`
- allow the workflow to route into `refinement` automatically once the search frontier has narrowed enough

### Sandbox mode

Goal: maximize freedom within the local safety envelope.

Behavior:

- allow the system to choose model family and architecture automatically
- allow selection of approved feature-engineering transforms
- allow compound changes spanning architecture plus features
- require stronger provenance logging because changes are broader
- allow broader candidate-model generation in the run-local `train_candidate.py` file, not just `ArchitectureSpec` mutations, as long as the file still obeys the harness contract and validation rules

Required constraints:

- all transformations must come from local registered modules
- all output must remain reproducible from session artifacts
- keep/discard logic must remain local and metric-based
- sandbox runs should record the exact feature transform plan alongside the architecture payload
- sandbox candidates must still pass local validation for allowed imports, model interface, tensor shapes, and training compatibility before execution
- if `sandbox` later uses multiple reasoning steps, those should still be coordinated by workflows rather than collapsed into one oversized prompt

## Composable Agent Patterns

The plan should explicitly support the following patterns rather than assuming one monolithic agent:

### Agents-as-tools

Each mode-specific reasoner should be invokable by the workflow as a bounded decision service.

Examples:

- `search` agent suggests the next broad candidate
- `refinement` agent suggests a local mutation near the incumbent
- `interpretation` agent summarizes the latest run

### Router agent

The workflow may later add a router that decides whether the next iteration should use `search`, `refinement`, or `sandbox`.

Near-term recommendation:

- keep agent mode user-selected at launch first
- design workflow interfaces so a future router can be inserted without rewriting execution tools

### Parallel agents

Parallel reasoning should be optional and explicit.

Good candidates for future parallelism:

- ask for multiple `search` proposals in parallel, then select one locally
- ask for multiple interpretations or critiques of the same near-miss run
- ask a `sandbox` planner for several candidate directions while the local workflow chooses the safest valid one

Training execution should remain local and controlled even if reasoning becomes parallel.

## Data and Schema Changes

### SessionContext additions

Extend `SessionContext` with fields for:

- user launch parameters
- OpenAI model selection
- agent mode
- iteration budget
- RNA-FM toggle
- dataset selector
- model-type constraint
- feature-engineering policy

### RunContext additions

Extend `RunContext` with fields for:

- `agent_mode`
- `proposal_strategy`
- `proposal_payload_path`
- `interpretation_payload_path`
- `feature_plan`
- `prompt_fingerprint`
- `response_fingerprint`
- `openai_request_id`

### New persisted artifacts per run

Recommended additional run artifacts:

- `proposal_input.json`
- `proposal_output.json`
- `interpretation_input.json`
- `interpretation_output.json`
- `feature_plan.json`
- `candidate_spec.json`

These should sit alongside the existing `run_context.json`, `decision.json`, `summary.json`, and analysis artifacts.

## CLI Refactor Plan

### Start command

Expand `autoresirch.session_manager.cli start` to accept:

- `--dataset`
- `--use-rnafm`
- `--iterations`
- `--model-type`
- `--agent-mode`
- `--openai-model`
- optional `--feature-engineering-policy`

Recommended rule:

- `start` should perform preparation immediately and persist the fully resolved configuration
- validation should reject incompatible combinations early, for example `cnn` without RNA-FM when that remains a hard harness rule

### Run loop command

Add a higher-level command that executes the session loop without requiring an external coding agent:

```bash
uv run python -m autoresirch.session_manager.cli launch --session-id <session_id>
```

Or combine creation plus execution:

```bash
uv run python -m autoresirch.session_manager.cli launch \
  --experiment-mode ... \
  --dataset ... \
  --use-rnafm ... \
  --iterations ... \
  --model-type ... \
  --agent-mode ...
```

Recommended behavior:

1. create or load session context
2. run base experiment
3. iterate for `n` candidate runs
4. request proposal from OpenAI
5. materialize local candidate
6. train and evaluate locally
7. request interpretation from OpenAI
8. persist artifacts and update state
9. finalize session summary

## OpenAI Integration Plan

### Proposal call responsibilities

The proposal API call should receive:

- experiment mode
- model-type constraint
- RNA-FM availability
- current incumbent summary
- top prior runs
- mode-specific strategy instructions
- allowed parameter ranges and architecture constraints
- optional allowed feature transforms for sandbox mode
- current workflow state such as iteration index, incumbent age, and recent failure count

The proposal response should return structured JSON only.

Minimum proposal fields:

- `family`
- `hidden_dims`
- `dropout`
- `normalization`
- `use_rnafm_embeddings`
- `conv_channels`
- `kernel_sizes`
- `rationale`
- `expected_tradeoffs`
- `novelty_tag`

Mode-specific proposal contract:

- `search` and `refinement` should return an `ArchitectureSpec`-shaped payload plus rationale
- `sandbox` may return a richer structured candidate description that the local materializer converts into a broader run-local module
- even in `sandbox`, the API response should remain structured rather than writing raw files directly

The proposal call should not decide:

- whether a run is kept or discarded
- how files are written
- how training is launched
- how session state transitions are persisted

Those remain workflow or tool responsibilities.

### Interpretation call responsibilities

The interpretation API call should receive:

- run summary
- decision record
- deltas versus incumbent and base
- diagnostics
- compact recent run history

The interpretation response should return structured text fields that map cleanly to the current analysis model:

- summary label
- freeform analysis
- likely helped
- likely hurt
- next-step reasoning
- confidence

This allows the current `agent_analysis.json` concept to survive, but with OpenAI rather than a coding agent providing the narrative.

## Recommended Migration Strategy

### Phase 1: Session configuration refactor

Scope:

- add launch parameters to CLI and session schemas
- persist resolved config in session artifacts
- keep the current manual `train.py` workflow working

Acceptance criteria:

- session start accepts and stores the new launch fields
- tests cover config serialization and validation
- existing standard and comparative runs still work

### Phase 2: OpenAI interpretation only

Scope:

- keep candidate generation manual or template-based
- use OpenAI only to generate run interpretations

Why this phase matters:

- it validates API plumbing, schemas, retries, logging, and cost controls without changing the search loop yet

Acceptance criteria:

- each completed run can produce interpretation artifacts through the API
- failures degrade gracefully to no interpretation rather than breaking the run

### Phase 3: Structured proposal engine

Scope:

- add OpenAI proposal generation for new architectures
- materialize candidates locally from structured specs
- remove reliance on ad hoc source editing for normal runs

Acceptance criteria:

- candidate runs can be generated and executed without editing `autoresirch/train.py`
- proposal payloads are persisted per run
- invalid API outputs are rejected before training begins
- workflow code, not prompts, owns iteration sequencing and retry behavior

### Phase 4: Agent-mode strategy layer

Scope:

- implement `search`, `refinement`, and `sandbox` mode policies
- wire mode-aware prompt construction and run selection

Acceptance criteria:

- the same session launch flow can execute in all three modes
- prompts and resulting proposal behavior differ by mode in a controlled, testable way
- mode selection and transitions are visible in workflow state rather than hidden inside prompts

### Phase 5: Feature-engineering submodule

Scope:

- add registered feature transforms
- allow only `sandbox` mode to select among approved transforms at first

Acceptance criteria:

- feature plans are deterministic and persisted
- transformed feature sets remain compatible with existing fold-local training
- leakage checks and regression tests cover new transforms

## Task Breakdown

1. Define new launch and proposal schemas in `session_manager` and `prepare`.
2. Expand the CLI to accept session parameters and validate legal combinations.
3. Persist launch config and feature policy in `session_context.json`.
4. Introduce an OpenAI client wrapper with retry, timeout, and JSON validation.
5. Add structured proposal and interpretation artifact writers.
6. Add a candidate-materialization path that generates run-local `train_candidate.py` files without editing `autoresirch/train.py`.
7. Implement `search` mode over an `ArchitectureSpec`-only proposal engine and stable candidate template.
8. Implement `refinement` mode using incumbent-aware `ArchitectureSpec` mutations and the same template path.
9. Implement the `prepare.feature_engineering` registry and deterministic transforms.
10. Implement `sandbox` mode using approved transforms plus broader run-local candidate generation under the same validation envelope.
11. Update tests for CLI, session persistence, proposal validation, and run-artifact generation.
12. Update `README.md` and replace the current agent-centric `program.md` with OpenAI-session documentation.

## Areas That Are Still Too Vague

These areas need to be made explicit before implementation starts, otherwise the plan still leaves too much to prompt behavior:

### Candidate execution entrypoint

The plan says the harness will execute `train_candidate.py`, but it does not yet specify:

- how the training loader imports a run-local module instead of [autoresirch/train.py](/Users/lucasplatter/sirchml-autoresearch/autoresirch/train.py)
- whether the current architecture-loading code will accept a file path, a module path, or both
- how run-local candidates are validated before import

### Workflow state machine

The plan names session phases but does not yet define a state machine for:

- base run completion
- candidate proposal failure
- candidate validation failure
- training crash
- interpretation failure
- mode-specific retry or fallback behavior

### Router behavior

The plan mentions possible routing patterns but does not yet specify:

- whether mode is always fixed by the user or can be changed mid-session
- what conditions would trigger `search` -> `refinement`
- whether `sandbox` is ever auto-entered or only manually selected

### Sandbox candidate contract

The plan says `sandbox` gets broader freedom, but the exact boundary still needs to be written down:

- which imports are allowed
- whether helper functions or helper classes are allowed in generated candidates
- whether `sandbox` may define custom `nn.Module` subcomponents freely
- whether `sandbox` may alter loss-facing output behavior, as long as the final interface still matches the harness

### Feature-engineering safety model

The plan allows approved transforms, but does not yet specify:

- whether transforms run before or inside fold-local preprocessing
- which transforms are safe at whole-dataset scope versus fold-local scope
- how transform provenance is fingerprinted and persisted
- how transform compatibility with `standard` and `comparative` modes is validated

### Parallel reasoning policy

The plan allows future parallel agents, but does not yet define:

- how many concurrent proposal calls are allowed
- whether parallel proposals are selected by heuristics, a router, or a second agent
- whether parallelism changes reproducibility expectations

## Critical Dependencies

- Session schema changes block almost everything else.
- Structured proposal validation should land before sandbox mode.
- Feature-engineering registry should land before enabling sandbox feature selection.
- Documentation updates should happen after CLI and artifact formats stabilize.

Critical path:

`session config` -> `OpenAI client + schemas` -> `proposal materialization` -> `search/refinement modes` -> `sandbox feature engineering`

## Risks and Mitigations

## Risk: API responses drift outside the architecture contract

**Category:** Technical  
**Likelihood:** Medium  
**Impact:** High

**Mitigation:**

- require structured JSON responses
- validate against local schemas before execution
- reject out-of-bounds architecture values with clear failure artifacts

## Risk: Sandbox mode introduces data leakage or irreproducible features

**Category:** Technical  
**Likelihood:** Medium  
**Impact:** High

**Mitigation:**

- allow only registered local transforms
- persist every transform and parameter in a feature plan
- add fold-local leakage tests before enabling sandbox by default

## Risk: Prompt strategy becomes hard to reason about across modes

**Category:** Scope  
**Likelihood:** Medium  
**Impact:** Medium

**Mitigation:**

- keep mode behavior in dedicated modules
- use shared proposal schemas across all modes
- version prompt templates and fingerprint them in run artifacts

## Risk: Existing workflow breaks before the replacement is complete

**Category:** Timeline  
**Likelihood:** Medium  
**Impact:** High

**Mitigation:**

- preserve the current manual or agentic path until Phase 3 is stable
- gate new flow behind a new `launch` path or an explicit flag

## Open Questions

- Should `session mode` remain the current `experiment_mode` (`standard` vs `comparative`), or do you want another higher-level concept in addition to `agent_mode`?
- Should `dataset` be a named config, a raw CSV path, or both?
- Do you want OpenAI to propose only `ArchitectureSpec`-shaped models, or do you want a richer intermediate DSL for architectures?
- Should `sandbox` be allowed to choose between standard and comparative experiment modes, or only operate within the user-selected experiment mode?
- Do you want the first OpenAI integration to target one official model family for both proposal and interpretation, or separate models for each task?
- Should the first implementation keep mode fixed for the entire session, or should the workflow be allowed to route from `search` into `refinement` automatically?

## Recommended First Implementation Slice

The smallest useful slice is:

1. extend `start` with the new launch parameters
2. persist them into session context
3. add an OpenAI interpretation path for completed runs
4. keep candidate generation local and fixed for that first pass

That sequence gives you immediate value, keeps risk low, and sets up the later proposal-engine migration without forcing the whole workflow to change at once.
