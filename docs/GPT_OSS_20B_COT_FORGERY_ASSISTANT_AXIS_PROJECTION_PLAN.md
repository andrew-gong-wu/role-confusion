# GPT-OSS-20B CoT-Forgery × Assistant Axis: paired projection-drift plan

Date prepared: 2026-08-27

## Objective

Test whether CoT-forgery attacks cause a later movement away from the default-Assistant end of the validated GPT-OSS-20B Assistant Axis, analogous to the persona-drift trajectories studied by Lu et al.

The central hypothesis is:

```text
successful CoT forgery
    → increasingly negative Assistant-Axis movement during generated reasoning
    → harmful/compliant final answer
```

This is a replay-and-analysis study. Do not generate new target-model responses, new forgeries, new safety judgments, or additional benchmark rows.

## Existing evidence

### Model

- Model: `openai/gpt-oss-20b`
- Revision: `6cee5e81ee83917806bbde320786a8fb61efebee`
- Deterministic decoding in the source run: greedy, seed 123, low reasoning effort, 1,024-token cap

### Assistant Axis

Validated q20 run:

```text
/workspace/role-probe-storage/outputs/gptoss20b-assistant-axis-q20-20260827-1550
```

Relevant files:

```text
gate-4-assistant-axis/analysis/axis-by-layer.pt
gate-4-assistant-axis/analysis/persona-vectors.pt
gate-4-assistant-axis/analysis/projection-distribution.csv
gate-4-assistant-axis/analysis/role-axis-cosines.csv
gate-4-assistant-axis/analysis/principal-angles.csv
gate-3-role-probes/probe-coefficients.npz
gate-3-role-probes/probe-artifact-metadata.json
sha256sums.txt
```

Axis definition:

```text
Assistant Axis = mean(default Assistant) - mean(persona-conditioned responses)
```

Positive projection points toward the default Assistant; negative movement points away from it.

The causal sanity check found:

- Primary causal site for this study: layer 16 `block_output`.
- Layer 16 block-output `alpha = -0.25` coherently increased persona embodiment on two of three prompts.
- A norm-matched random orthogonal control did not reproduce the shift.
- Layer 18 is geometrically strong but was causally inconsistent; it remains a secondary comparison site.

Causal-test artifact bundle:

```text
/workspace/role-probe-storage/outputs/gptoss20b-assistant-axis-causal-20260827-1848
```

### CoT-forgery expansion

Use the 200-pair expansion, not only the 10-pair pilot:

```text
/workspace/role-probe-storage/outputs/gptoss20b-cot-forgery-gate6-400-20260827-1740
```

Existing outcomes:

- 200 baseline responses
- 200 CoT-forgery responses
- 199 fully judged pairs
- 148 baseline-safe → attack-harmful gains
- 51 unchanged safe/refusal/redirection pairs
- 0 losses
- Six StrongREJECT categories

The report calls this the Gate 6 expansion. Its internal `run-summary.json` retains older stage-number fields (`completed_gates: [5]`); do not infer experiment scope from that internal list. Use the report, response count, and `status: gate_6_cot_forgery_expansion_complete`.

Non-sensitive paired labels:

```text
cot-forgery-pilot/analysis/paired-results.csv
cot-forgery-pilot/prepare/dataset-manifest.csv
```

Sensitive archive:

```text
sensitive-artifacts.tar.gz.enc
```

The archive contains:

```text
cot-forgery-pilot/prepare/conditions.jsonl.gz
cot-forgery-pilot/prepare/forgeries.jsonl.gz
cot-forgery-pilot/prepare/seed-responses.jsonl.gz
cot-forgery-pilot/generation/responses.jsonl.gz
cot-forgery-pilot/judge/judge-raw.jsonl.gz
```

Encryption metadata records AES-256-CBC with PBKDF2-HMAC-SHA256 and 600,000 iterations. The owner-held key is outside Git at:

```text
/Users/rigel/.codex/gate5-keys/gptoss20b-cot-forgery-gate6-400-20260827-1740.key
```

Never print, commit, upload, or copy the key into the repository or persistent output directory.

## Interpretation to preregister

The conversational role probes and persona Assistant Axis measure related but distinct factors:

- Role probes classify system/user/tool/CoT/assistant generation modes and may encode broad style.
- The Assistant Axis distinguishes default-Assistant behavior from persona embodiment within assistant generations.
- At layer 16 block output, the Assistant Axis has near-random direct cosine with the assistant-role coefficient (`-0.0176`) but a validated causal persona effect.
- At layer 18, the overlap is larger (`-0.0898` block output; `-0.1265` pre-MLP) and may be statistically non-random.

Therefore analyze raw Assistant-Axis projection, role-probe scores, and an Assistant Axis residualized against the role-probe subspace. Do not assume a projection drop is persona drift until style/channel explanations have been examined.

## Safety, isolation, and source control

### Separate worktree

Do not modify or run from the existing `partner` worktree. It contains user-owned changes and CoT-forgery artifacts.

Create:

```text
worktree: /workspace/role-confusion-worktrees/cot-forgery-assistant-axis-projection
branch:   codex/cot-forgery-assistant-axis-projection
```

Start from the committed CoT-forgery/q20 tip available at execution time. Record the exact source commit. If the expected CoT-forgery code exists only as uncommitted changes in `partner`, do not copy or commit those changes without user authorization. A standalone replay runner is preferable.

The last known committed CoT-forgery tip was:

```text
6e50edf5ca8258872520c9b8074561d9f9ae6ea9
```

The final causal-test/report tip was:

```text
9d67218759d822e925499c24a53f566b999c6719
```

Record current branch tips before choosing a base; do not assume these remain current.

### Output directory

Create a new directory and never reuse an existing one:

```text
/workspace/role-probe-storage/outputs/gptoss20b-cot-forgery-assistant-axis-projections-YYYYMMDD-HHMM
```

Treat the q20 and Gate 6 source runs as read-only.

### GPU coordination

Run `nvidia-smi` immediately before model load and again before activation replay. Record all compute-process PIDs and allocations. Do not load the model if another GPU process is active.

### Sensitive plaintext

1. Verify the encrypted archive checksum before decryption.
2. Obtain explicit authority to access the owner-held key if it is not already placed in scope for the execution task.
3. Reuse a verified decryption method compatible with `encryption-metadata.json`; do not improvise new cryptography.
4. Decrypt only into a fresh owner-only temporary directory created with `mktemp -d` and umask `077`.
5. Verify every extracted plaintext file against the SHA-256 values in `encryption-metadata.json`.
6. Never display raw prompts, forged policies, responses, token IDs, or judge payloads in tool output or reports.
7. Transfer only the minimum ledger needed for replay if local-to-remote movement is required.
8. After verified aggregate and encrypted pair-level artifacts are complete, remove temporary plaintext and report that removal. Do not remove the encrypted source archive.

Stop if decryption, permissions, or digest verification fails.

## Stage 1: provenance and joins

Before model load:

1. Verify `sha256sums.txt` for both the q20 Assistant-Axis run and Gate 6 CoT-forgery run.
2. Record model revision, tokenizer hash, Harmony template hash, source commits, package versions, CUDA version, and GPU snapshot.
3. Load the saved Assistant Axes at layers 16 and 18 for `pre_mlp` and `block_output`.
4. Verify every selected vector is `[2880]`, finite, nonzero, and hash the exact float32 tensor used.
5. Load the standardized role-probe artifacts and record exact keys, class order, scaler parameters, coefficient hashes, and intercept hashes.
6. Verify the sensitive response ledger has exactly 400 unique responses: 200 baseline and 200 attack.
7. Join responses to the 200 requested pairs by stable response/request ID or stored digest, never by row order.
8. Join the 199 analyzable outcome labels and six categories from `paired-results.csv`.
9. Record the one unlabeled pair and exclude it only from outcome-stratified analyses; retain it for condition-only trajectory summaries.

Do not proceed if IDs are duplicated, pair joins are not one-to-one, conditions are imbalanced, or stored prompt/full-token boundaries are missing.

## Stage 2: replay and hook validation

Replay the already generated full token sequences through the pinned model. Do not call `generate` for the main extraction.

### Activation sites

Primary:

```text
layer 16 block_output
```

Secondary:

```text
layer 18 block_output
layer 16 pre_mlp
layer 18 pre_mlp
```

Definitions must match q20 construction exactly:

- `pre_mlp`: `model.model.layers[i].post_attention_layernorm` output, the MLP input coordinate.
- `block_output`: decoder-block output after attention, MLP, and residual updates.

### Efficient extraction

Do not save full 2880-dimensional token activations. Inside each hook, compute and retain only:

- Projection on the unit Assistant Axis.
- Projection on the role-subspace-residualized unit Assistant Axis.
- Exact standardized role-probe class logits/scores.
- Token index, token role/channel, prompt-versus-generated status, response ID hash, site, and layer.

Accumulate projections in float32. Save pair-level trajectories in a sensitive encrypted artifact, not as public plaintext.

### Replay equivalence test

Before full extraction, select at least two sequences with different lengths and conditions and compare:

1. Full-sequence causal replay with `use_cache=False`.
2. Incremental cached teacher-forced replay over the same stored tokens.

Compare logits and site projections at matching positions. Record maximum absolute error and cosine/correlation. Establish a numerical tolerance from the observed kernel behavior before looking at attack-success groups.

Stop if replay changes token alignment, produces non-finite values, or materially disagrees with incremental decoding.

### Hook requirements

- Hooks must cover every replayed token.
- Prompt padding must be excluded from projections.
- Hooks must be removed after every batch, including exceptions.
- No non-finite activation, projection, or role-probe score is allowed.
- Record peak GPU memory and remaining-hook count.

## Stage 3: token segmentation

Use the pinned Harmony-token parser already used in the q20 runner (`label_gptoss_content_roles` or an exactly validated equivalent).

For every token, identify:

- Prompt token versus newly generated token.
- System, user, tool, analysis/CoT, assistant-final, or structural token.
- First generated token.
- Last generated analysis token.
- First assistant-final content token.
- End of assistant-final content.

Exclude structural delimiter tokens from content-window summaries, but retain them in a separate diagnostic trace.

Validate segmentation manually on a small encrypted sample without exposing content. Report token counts and boundary indices only.

## Stage 4: projection normalization

For every site/layer, calculate a default-Assistant calibration from the q20 activation distribution:

```text
z_t = (projection_t - default_assistant_mean) / default_assistant_sd
```

Record the calibration population and formula. Do not compare raw projection values across sites or layers.

Also retain pair-centered attack-minus-baseline projections:

```text
delta_z_pair,t = z_attack,t - z_baseline,t
```

For variable-length trajectories, use both:

- Event alignment around the analysis-to-final transition.
- Normalized progress within analysis and final channels separately.

Do not align analysis tokens directly against final tokens solely by global sequence percentile.

## Stage 5: role/style controls

At layer 16, reconstruct the standardized `system|user|cot|assistant` probe exactly, including scaler and intercepts. Save trajectories for:

- Assistant role score.
- CoT role score.
- Optional margin: `assistant - mean(other role logits)`.

Construct an orthonormal basis from the centered raw-coordinate role-probe coefficients and residualize the Assistant Axis:

```text
v_residual = unit(v_axis - P_role_subspace(v_axis))
```

Verify and save:

- Raw-axis/role-subspace projection norm.
- Residual-axis norm and finiteness.
- Residual-axis cosine with every role coefficient, approximately zero within tolerance.
- Tensor hashes.

Repeat primary analyses using raw and residualized axes.

At layer 18, use the available standardized binary assistant/user probe only as a limited secondary control; do not claim it controls CoT style.

## Preregistered estimands

### Primary estimand

Site: layer 16 block output. Population: 199 labeled pairs.

Within each final-channel response, divide assistant-final content tokens into quartiles by normalized final-channel position. Require at least four final content tokens. Define:

```text
final_drift(condition) = mean(z in final Q4) - mean(z in final Q1)

paired_drift_effect = final_drift(attack) - final_drift(baseline)
```

Primary test:

```text
H1: paired_drift_effect < 0
```

Report mean, median, pair-bootstrap 95% confidence interval, and an exact paired sign-flip permutation p-value. The confidence interval and raw effect size are primary; do not rely only on the p-value.

### Key secondary estimands

1. **Late state difference**

   ```text
   mean(attack final Q4) - mean(baseline final Q4)
   ```

2. **Generated-analysis drift**

   ```text
   mean(analysis Q4) - mean(analysis Q1)
   ```

   Compare attack versus baseline only for responses with at least four generated analysis-content tokens.

3. **Analysis-to-final transition**

   ```text
   mean(first up-to-16 final tokens) - mean(last up-to-16 analysis tokens)
   ```

4. **Attack-success association**

   Compare paired projection effects between the 148 attack gains and 51 unchanged pairs. This is an association with post-treatment success, not a randomized causal contrast.

5. **Category heterogeneity**

   Report all six category-specific estimates with confidence intervals. Do not select categories post hoc.

6. **Residualized-axis sensitivity**

   Repeat the primary and key secondary estimands using `v_residual`.

7. **Role-probe trajectories**

   Repeat trajectory summaries for Assistant-role and CoT-role scores, without treating them as interchangeable with the Assistant Axis.

### Whole-trajectory model

As a secondary visualization/summary, fit a pair-bootstrap trajectory or mixed model over normalized within-channel progress with:

```text
projection ~ condition × progress + category
```

Do not let this flexible model replace the preregistered window-based estimands.

## Temporal interpretation

Classify the result using the following hierarchy:

### Strong drift evidence

- Attack-minus-baseline projection becomes increasingly negative during generated analysis.
- The decrease precedes the first harmful/compliant final token.
- The decrease is larger in successful than failed attacks.
- It remains on the role-subspace-residualized axis.

### Behavioral-state correlate

- Projection differs mainly in the final answer.
- The difference is associated with success but does not precede it.

### Prompt/style offset

- The difference is already fully present at the first generated token.
- Assistant-role and/or CoT-role scores move in parallel.
- Residualization substantially removes the effect.

### No evidence

- Paired trajectories do not show a reliable later negative shift.
- Any changes are category-specific, inconsistent, or comparable in successful and failed attacks.

Do not call a later drop mechanistic or mediating based on temporal precedence alone.

## Required controls and diagnostics

1. Raw baseline versus CoT-forgery paired trajectories.
2. Successful versus failed attacks under the same forgery method.
3. Raw Assistant Axis versus role-subspace-residualized Assistant Axis.
4. Assistant-role and CoT-role probe trajectories.
5. Prompt-only projection offset at the first generated token.
6. Response-length sensitivity:
   - Quartile windows.
   - Fixed first/last up-to-16-token windows.
   - Exclude 1,024-token-cap responses as a sensitivity analysis, not the primary analysis.
7. Category-stratified results.
8. Layer/site replication at the three secondary coordinates.

Do not add a new formatting-matched generation control in this run. If the existing data show an immediate prompt/style offset that cannot be separated using failed attacks and residualization, report that limitation and propose a separately authorized control study.

## Statistical safeguards

- The unit of resampling and inference is the request pair, never the token.
- Bootstrap complete pairs so long responses do not dominate uncertainty.
- Use category-stratified resampling for the aggregate category-adjusted estimate.
- Do not treat thousands of tokens as independent observations.
- Report all preregistered sites and endpoints, including null or opposite-direction results.
- Correct exploratory layer/site scans for multiple comparisons or label them exploratory.
- Preserve the one unlabeled pair in condition-only summaries but exclude it from success-stratified tests.

## Artifact policy

### Sensitive encrypted artifacts

Keep encrypted at rest:

- Pair-level token trajectories.
- Response/request IDs if reversible.
- Prompt lengths and token IDs if they could aid reconstruction.
- Any raw decrypted ledger or response text.

Use a fresh encryption key stored owner-only outside Git. Record cipher metadata and verify an encrypt/decrypt round trip before deleting temporary plaintext.

### Public/non-sensitive artifacts

The report bundle may contain only:

- Aggregate trajectory means and bootstrap intervals.
- Category-level estimates with sufficiently large groups.
- De-identified effect-size tables.
- Vector hashes, source digests, environment metadata, and replay validation.
- Plots that do not expose raw text or token IDs.

Suggested files:

```text
REPORT.md
provenance.json
axis-and-probe-metadata.json
replay-validation.json
segmentation-summary.json
primary-estimates.csv
category-estimates.csv
sensitivity-estimates.csv
trajectory-aggregate.csv
trajectory-by-channel.png
event-aligned-transition.png
category-forest.png
role-style-control-trajectories.png
encryption-metadata.json
sha256sums.txt
```

## Plots

Produce, at minimum:

1. Layer-16 block-output Assistant-Axis projection over normalized generated-analysis progress, attack versus baseline.
2. Event-aligned projection around the analysis-to-final transition.
3. Final-channel normalized trajectory, attack versus baseline.
4. Successful versus failed attack trajectories.
5. Raw versus residualized Assistant-Axis trajectories.
6. Assistant-role and CoT-role score trajectories.
7. Six-category forest plot for the primary paired drift effect.

Show pair-bootstrap 95% intervals and sample counts at every bin.

## Stop conditions

Stop before full extraction if any of the following occurs:

- q20 or Gate 6 checksum failure.
- Encrypted archive digest mismatch.
- Plaintext digest mismatch after decryption.
- Missing or unauthorized key access.
- Response/pair joins are not one-to-one.
- Model/tokenizer/template revision mismatch.
- Assistant Axis or probe artifact is non-finite, wrong-shaped, or misoriented.
- Full replay and incremental replay disagree materially.
- Token-role segmentation cannot reliably identify generated analysis and final content.
- Any external GPU process is active.
- Any sensitive plaintext appears in a report, terminal transcript, Git status, or public artifact.

If blocked by sensitive-data handling, do not weaken the policy; stop and ask the user.

## Completion criteria

The study is complete when:

1. All 400 stored responses have validated projection trajectories at the primary site.
2. The 199 labeled pairs have primary paired estimates.
3. Successful/failed, category, residualized-axis, role-probe, length, layer, and site sensitivities are reported.
4. No new model generations or judgments were made.
5. Sensitive pair-level artifacts are encrypted and temporary plaintext has been removed.
6. The public report contains all preregistered aggregate results, including nulls.
7. The GPU is idle and the worktree is clean at handoff.

## Recommended final language

If the hypothesized effect appears, prefer:

> CoT-forgery prompting caused a paired late-generation decrease in default-Assistant-axis projection. The decrease was larger among successful attacks and [did/did not] precede the final harmful response. The effect [did/did not] survive removal of the conversational role-probe subspace.

Avoid:

> The attack turned off the Assistant identity.

The latter is stronger than the design can establish.
