## Next Improvement: Richer `synopsis.md` Interpretation

### Objective Analysis

**What:** Expand the `Interpretation` section of each run-level `synopsis.md` so it explains metric tradeoffs, robustness, and likely next-step implications instead of only restating the decision reason.

**Why:** The current synopsis is useful for factual recall, but it still leaves too much manual inference to the next agent. In particular, mixed-signal runs such as "AUC down, Pearson up, Spearman up" can contain valuable directional information even when the run is correctly discarded by the primary-metric rule.

**Success looks like:** A future agent should be able to open one `synopsis.md` and immediately understand:

- whether the run was a clear win, clear loss, or mixed-signal result
- which secondary metrics moved with or against the primary metric
- whether the run improved ranking structure, absolute error, or only one narrow aspect of behavior
- whether the result looks broad-based or fragile across folds
- whether the same mutation direction is worth exploring again even if this specific run was discarded

### Current Gap

The current implementation writes a short `Interpretation` section containing mainly:

- the decision reason
- the hypothesis result

That is not enough for future iteration because it does not explicitly interpret:

- metric disagreement
- magnitude of secondary-metric changes
- whether the run was broadly robust or only improved a few folds
- whether a discard still provides evidence that the general mutation direction was promising

### Required Expansion Of `synopsis.md`

Keep the existing overall synopsis layout, but make the `Interpretation` section substantially richer.

The revised section should contain five distinct interpretation layers:

1. Decision interpretation
   - one sentence that states the keep/discard outcome and why the primary metric forced that decision

2. Metric tradeoff interpretation
   - one or more bullets that explain how the secondary metrics moved relative to the primary metric
   - these bullets should be explicit, not implicit

3. Robustness interpretation
   - one or two bullets describing whether the result looked broad-based or concentrated in a few folds
   - call out undefined metrics and fold instability directly

4. Hypothesis assessment
   - classify the hypothesis as `supported`, `partially supported`, `unsupported`, or `inconclusive`
   - `partially supported` should be allowed for mixed-signal runs

5. Next-run implication
   - one or two bullets that say whether the next agent should continue in the same direction, reverse the mutation, or isolate a sub-component of the change

### New Interpretation Inputs

The synopsis writer should compare the current run against both:

- the `compared_against_run_id` summary
- the session base run summary

This gives two useful views:

- `delta_vs_compared_run`: what changed against the current incumbent or designated comparison target
- `delta_vs_base_run`: whether the run still moved the session forward overall even if it lost to the incumbent

The `Interpretation` section should use the compared-run deltas for decision commentary and may use base-run deltas for session-context commentary.

### Proposed Additional `Performance Overview` Content

To support richer interpretation, add a compact metric movement block before the interpretation bullets.

For each of these metrics, show:

- current value
- delta vs compared run
- delta vs base run
- qualitative label: `better`, `worse`, `flat`, or `undefined`

Required metrics:

- `weighted_cv_auc`
- `weighted_cv_rmse_mean`
- `weighted_cv_pearson_r_mean`
- `weighted_cv_spearman_r_mean`
- `cv_rmse_std`
- `num_params`
- `train_seconds`

This should be short and human-readable, not a large table.

### Required Rule-Based Interpretation Patterns

The interpretation should be generated from deterministic rules so the markdown is consistent across runs.

At minimum, implement explicit handling for these cases:

1. `AUC down`, `Pearson up`, `Spearman up`
   - interpretation:
     - the run improved global correlation or ordering structure, but worsened threshold-based separation
     - the mutation direction may still be informative
   - next-run implication:
     - keep the change direction in mind, but try a smaller or more isolated mutation instead of repeating the same full step

2. `AUC up`, `Pearson down` or `Spearman down`
   - interpretation:
     - the run improved the decision metric but may have done so in a threshold-specific or brittle way
   - next-run implication:
     - keep the run if the AUC rule says so, but probe nearby variants to recover the lost secondary behavior

3. `AUC flat`, `RMSE down`
   - interpretation:
     - the run improved absolute error without improving classification-style separation
   - next-run implication:
     - consider nearby variants only if absolute-fit improvements are strategically useful for later AUC recovery

4. `AUC down`, `RMSE down`
   - interpretation:
     - the run fit the continuous target better but hurt effective/non-effective separation
   - next-run implication:
     - do not keep it, but it may indicate the mutation improved calibration rather than ranking

5. `AUC up`, `RMSE up`
   - interpretation:
     - the run improved separation while hurting absolute fit
   - next-run implication:
     - keep by rule if warranted, but note that the gain may come with calibration cost

6. `Pearson up`, `Spearman flat/down`
   - interpretation:
     - the run improved linear fit more than ordering quality

7. `Spearman up`, `Pearson flat/down`
   - interpretation:
     - the run improved ordering more than linear calibration

8. `cv_rmse_std up` or large best-fold / worst-fold spread
   - interpretation:
     - the run became less robust across folds
   - next-run implication:
     - prefer smaller or simpler nearby mutations

9. undefined AUC or correlation count increases
   - interpretation:
     - caution that metric coverage became less informative
   - next-run implication:
     - avoid overstating the meaning of any small metric movement

10. near-miss discard
   - when AUC declines only slightly but multiple secondary metrics improve
   - interpretation:
     - explicitly call the run a near miss rather than a plain failure
   - next-run implication:
     - recommend a nearby follow-up mutation

### New Interpretation Vocabulary

The writer should use a small, controlled vocabulary for summary phrases:

- `clear improvement`
- `clear regression`
- `mixed signal`
- `near miss`
- `brittle gain`
- `broader-based gain`
- `calibration gain`
- `ranking gain`
- `threshold-separation gain`
- `robustness concern`

This keeps the synopsis consistent across runs and makes it easier for future agents to scan many synopses quickly.

### Exact Implementation Changes

Modify [session_manager.py](/Users/lucasplatter/sirchml-autoresearch/session_manager.py), specifically the `write_run_synopsis(...)` path, to do the following:

1. load both comparison summaries
   - current run summary
   - compared-against summary
   - session base-run summary

2. compute explicit metric delta bundles
   - add helper(s) such as:
     - `build_metric_delta_bundle(...)`
     - `classify_metric_direction(...)`
     - `classify_tradeoff_patterns(...)`
     - `build_interpretation_bullets(...)`

3. allow richer hypothesis outcomes
   - extend the generated hypothesis result vocabulary from the current three-way style to:
     - `supported`
     - `partially_supported`
     - `unsupported`
     - `inconclusive`

4. write richer markdown
   - expand the `## Performance Overview` and `## Interpretation` blocks in `synopsis.md`
   - do not change the frontmatter decision fields
   - do not change the keep/discard rule

5. keep the output deterministic
   - the same inputs should always produce the same interpretation bullets
   - avoid free-form agent prose generation here

### Suggested Heuristic Thresholds

Use explicit interpretation thresholds so the writer does not overreact to tiny changes.

Recommended approach:

- `weighted_cv_auc`
  - use the existing decision epsilon for decision semantics
  - use a slightly larger "meaningful interpretation" threshold for descriptive language

- `weighted_cv_rmse_mean`
  - treat very small movements as `flat`

- `weighted_cv_pearson_r_mean`
  - treat very small movements as `flat`

- `weighted_cv_spearman_r_mean`
  - treat very small movements as `flat`

- `cv_rmse_std`
  - call out only meaningful robustness changes

The exact thresholds should live in `session_manager.py` as named constants so they are easy to inspect and tune later.

### Example Of Desired Interpretation

For a discard where AUC fell slightly but correlation improved:

```markdown
## Interpretation

- Decision interpretation: discarded because `weighted_cv_auc` fell by `-0.0016`, which did not beat the incumbent.
- Metric tradeoff: this was a mixed-signal run; `weighted_cv_pearson_r_mean` and `weighted_cv_spearman_r_mean` improved while AUC declined.
- Metric tradeoff: the mutation may have improved global ranking or calibration structure without improving threshold-based separation.
- Robustness interpretation: fold-level instability did not improve, so the secondary gains should be treated cautiously.
- Hypothesis assessment: `partially_supported`
- Next-run implication: try a smaller or more localized mutation in the same direction rather than repeating the full change.
```

For a keep where AUC improved but RMSE worsened:

```markdown
## Interpretation

- Decision interpretation: kept because `weighted_cv_auc` improved by `+0.0042`, above epsilon.
- Metric tradeoff: this was a valid primary-metric win, but `weighted_cv_rmse_mean` worsened, so the gain may be separation-specific rather than a broad regression improvement.
- Robustness interpretation: gains were concentrated in a small number of folds, which makes this a potentially brittle incumbent.
- Hypothesis assessment: `supported`
- Next-run implication: keep the incumbent, then probe nearby variants that try to recover RMSE without giving back AUC.
```

### Tests To Add

Extend [tests/test_session_manager.py](/Users/lucasplatter/sirchml-autoresearch/tests/test_session_manager.py) with focused cases covering:

- `AUC down`, `Pearson/Spearman up` produces an explicit mixed-signal interpretation
- `AUC up`, `RMSE up` produces an explicit tradeoff interpretation
- near-miss discard produces `partially_supported`
- robustness deterioration produces a robustness warning
- increased undefined metric counts produce a caution note
- crash synopses remain short and do not attempt fabricated interpretation

The tests should assert on concrete synopsis phrases, not vague substring fragments.

### Acceptance Criteria For This Improvement

This improvement is complete only when all of the following are true:

- each `synopsis.md` contains an `Interpretation` section that does more than restate the decision
- mixed-signal metric outcomes are explicitly named as tradeoffs
- the synopsis compares the run against both the compared run and the session base run
- `partially_supported` can be emitted for mixed-signal hypotheses
- robustness and undefined-metric cautions appear when warranted
- the keep/discard rule remains tied only to `weighted_cv_auc`
- tests cover the new interpretation logic and phrasing
