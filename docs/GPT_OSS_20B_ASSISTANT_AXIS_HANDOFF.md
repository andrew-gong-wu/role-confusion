# GPT-OSS-20B Assistant Axis: H100 handoff plan

Date prepared: 2026-08-27

This document is the execution plan for the next H100 session. It supersedes
the earlier recommendation in the repository README to begin with dense
Qwen3-32B. The Qwen attempt did not validate its role probes, and the next
experiment should use `openai/gpt-oss-20b`, whose existing pre-MLP role probes
are already strong.

## Objective

Regenerate a compact replacement for the lost raw GPT-OSS role probes, then
construct and validate a pilot Assistant Axis for `openai/gpt-oss-20b`, while
determining whether Assistant-persona geometry is meaningful at both of these
representation sites:

1. decoder-block output, matching the published Assistant Axis method; and
2. pre-MLP state, matching the successful GPT-OSS role probes in this repo.

The key comparison is only valid when both vectors use the same model, layer,
activation site, chat template, and residual-stream coordinates. Do not compare
a block-output Assistant Axis directly with a pre-MLP role-probe coefficient.

The default run is deliberately compute-minimized. It also includes only a
10-example, single-category CoT-Forgery pilot. Do not expand either experiment
automatically.

## Compute budget and reuse policy

Treat every autoregressive model call as expensive. The default maximum is:

| Stage | GPT-OSS generations | Other expensive calls |
| --- | ---: | ---: |
| Role-probe regeneration | 0 | Forward passes over 50 neutral passages |
| Assistant Axis compact pilot | 26 | At most 24 role-adherence judgments |
| CoT-Forgery category pilot | 20 | 0 forgery generations; at most 20 safety judgments |

The 26 Assistant Axis generations are 12 personas times two questions, plus
one default-Assistant response to each question. The 20 CoT-Forgery target
generations are one raw baseline and one forged prompt for each of 10 harmful
requests. Thus the default target-model generation budget is 46 responses.

The six-persona micro-pilot described below must be a strict subset of the
12-persona panel, so successful generations are reused rather than repeated.
Likewise, generate, judge, and tokenize every response once; replay saved token
IDs to extract all requested layers and both activation sites.

Before generating anything, search persistent storage and published artifacts
for exact prompt/response pairs. Reuse an artifact only when its model revision,
chat rendering, generation settings, and content digest are known. Never rerun
a completed sample merely to put it in a different output layout.

## Current evidence and motivation

- The exact GPT-OSS role-probe reproduction reached 0.8851 held-out accuracy at
  pre-MLP layer 16. Layer 12, the paper-comparison layer, reached 0.7055.
- The raw GPT-OSS probe objects, coefficient vectors, and large activation
  archive were subsequently lost. Only compact summary statistics and report
  artifacts remain in the repository. Tomorrow's run must therefore regenerate
  role-probe coefficients before any role-axis cosine or subspace comparison is
  possible.
- The dense Qwen3-32B three-way role probe reached only 0.3920 balanced
  accuracy, had NLL worse than uniform prediction, and emitted cuML line-search
  warnings. It is not a validated instrument.
- The Qwen adaptation captured full decoder-block outputs to match the
  published Qwen Assistant Axis. The upstream role-probe implementation instead
  captures `post_attention_layernorm` output immediately before the MLP. This
  activation-site difference is a plausible reason the Qwen role probe was
  weak.
- In the Assistant Axis method, the axis is a contrast direction,
  `mean(default Assistant) - mean(character personas)`. Alignment with persona
  PC1 is a validation result, not the definition of the axis. Failure to align
  with PC1 is informative but is not, by itself, sufficient to reject the
  contrast direction.

Relevant existing artifacts:

- `reports/2026-08-25-h100-exact-seed123/`
- `reports/2026-08-26-qwen3-32b-paper-three-way-role-probes-official-method/`
- `scripts/run_exact_role_probe.py`
- `scripts/run_qwen_role_probe_cosines.py`
- `scripts/run_qwen_paper_tag_role_probes.py`
- `docs/H100_RUNBOOK.md`
- `docs/EXTERNAL_ARTIFACTS.md`

## Non-goals for the first session

- Do not run gpt-oss-120b.
- Do not recompute the Qwen3-32B Assistant Axis.
- Do not begin a large persona-generation run before the activation hooks and
  small role-probe checks pass.
- Do not claim that PCA alignment alone establishes a causal or safety-relevant
  axis.
- Do not run adversarial steering or activation capping until the axis passes
  held-out geometric validation.
- Do not overwrite existing persistent outputs. Every run must use a new,
  timestamped directory.
- Do not assume that paths recorded in an old report still exist. Inventory
  every referenced persistent artifact before designing the run around it.

## Representation sites

Capture both sites during the same model forward pass.

### Pre-MLP

For decoder layer `i`, capture the output of
`model.model.layers[i].post_attention_layernorm`. Equivalently, capture the
input to `model.model.layers[i].mlp`. This is the coordinate system used by the
successful exact GPT-OSS role probes.

### Decoder-block output

Capture the output of `model.model.layers[i]` after attention and MLP residual
updates. This is the closest match to the published Assistant Axis activation
site and should be the primary site for an official-style replication.

Validate the hooks on one short prompt before any dataset run:

- both tensors must have shape `[batch, sequence, 2880]`;
- all values must be finite;
- the two sites must not be bit-identical;
- the block-output capture must equal the corresponding output from a standard
  Transformers forward hook;
- the pre-MLP capture must equal the existing pinned/custom-forward convention
  used by `run_exact_role_probe.py` on the same tokens;
- record maximum absolute error and cosine similarity for each equality check.

## High-level execution order

The session has six gates. Do not proceed past a failed gate.

1. Environment and model verification.
2. Dual-hook equivalence smoke test.
3. Small role-probe site comparison, followed by compact role-probe
   regeneration.
4. Small Assistant Axis pilot.
5. Single-category CoT-Forgery pilot.
6. Optional expansion, requiring review and an explicit decision.

If time is limited, complete gates 1-4 and preserve the outputs. A clean pilot
is more valuable than a partial full run.

## Gate 1: environment and model verification

Follow `docs/H100_RUNBOOK.md`, but use the already verified GPT-OSS model and
persistent cache where available.

Required checks:

```bash
nvidia-smi
git status --short
git rev-parse HEAD
python3 scripts/check_local.py
```

Record:

- GPU model and VRAM;
- repository commit;
- Python, PyTorch, Transformers, CUDA, cuML, and CuPy versions;
- model identifier and exact revision;
- tokenizer/chat-template hash;
- model config hash;
- persistent storage mount and free space.

Before running the model, inventory the old GPT-OSS artifacts referenced by
`reports/2026-08-25-h100-exact-seed123/README.md`. Explicitly check for:

- `neutral-passages.jsonl.gz`;
- the rendered prompt/token manifest;
- train/test split IDs;
- serialized probe objects;
- coefficient/intercept arrays;
- the 49.56 GB activation tensor;
- raw held-out logits/probabilities.

Write the result to `artifact-inventory.json`. Treat the user's statement that
the raw probes are lost as authoritative even if a stale path is present.
Never report a path as available without checking that it is readable and its
contents match any recorded digest.

If the neutral passages or split IDs survive, reuse them. If they do not,
regenerate the corpus with the exact pinned C4/Dolma revisions, seed, streaming
shuffle procedure, source counts, and truncation rules from the earlier run.
Record that the regenerated corpus is a new realization unless its digest
matches the old manifest. Do not claim bit-identical reproduction without a
matching digest.

Use the already pinned GPT-OSS checkpoint from the exact reproduction whenever
possible:

```text
model: openai/gpt-oss-20b
revision: 6cee5e81ee83917806bbde320786a8fb61efebee
hidden size: 2880
decoder layers: 24
```

### Known-good batch size

Batch size 128 previously ran GPT-OSS-20B inference on this H100 without any
memory errors. Start comparable inference and role-probe forward-pass stages at
`batch_size=128`. Do not repeat a batch-size search from 1, 2, 4, and so on.

If a materially heavier stage—such as long autoregressive generation or
simultaneous capture of additional activation tensors—actually raises an OOM,
record the failing stage and peak memory, then retry at 64 and continue halving
only as necessary. An untested concern about memory is not a reason to lower
the initial batch size.

Stop if the model revision, tokenizer, or Harmony formatting differs from the
exact role-probe run and the difference cannot be explained.

## Gate 2: implement and validate dual-site extraction

Create one new runner rather than modifying immutable prior-run code. Suggested
name:

```text
scripts/run_gptoss_assistant_axis.py
```

The runner should support independently executable stages:

```text
prepare-personas
generate
score
extract
analyze
steer-smoke
```

It should also support a `--pilot` flag and explicit paths for every input and
output. A stage must be resumable and must refuse to overwrite an existing
output directory.

For the hook smoke test, use one short, correctly formatted Harmony
conversation. Capture layers 12 and 16 at both sites. Save only compact JSON
statistics, not full activations.

Acceptance criteria:

- exact token IDs and decoded prompt are saved;
- logits from the hooked and unhooked forwards agree within the tolerance used
  by the exact reproduction;
- pre-MLP states agree with the existing custom-forward implementation;
- no non-finite values;
- hooks are removed even if the forward raises;
- peak allocated GPU memory is recorded.

## Gate 3: role-probe pilot and compact regeneration

Before constructing an axis, determine whether GPT-OSS roles are decodable at
decoder-block output as well as pre-MLP.

### Pilot data

- Reuse 20 neutral passages from the exact reproduction.
- Start with the binary role space `user` versus `assistant`.
- Use identical content in each role.
- Use the exact pinned Harmony renderers and canonical system prefix from the
  successful GPT-OSS run.
- Keep all role copies of a base passage in the same train/test split for the
  main diagnostic. A prompt-level split may be reported separately for direct
  upstream comparability.
- Cap retained content at 128 tokens per prompt for the pilot.
- Test layers 8, 10, 12, 14, 16, and 18 at both sites.

### Fitting

Fit the same train/test rows at both activation sites. Use float32 features.
Report both:

1. the exact prior cuML logistic-regression settings; and
2. a standardized logistic regression or another numerically stable linear
   baseline.

Use balanced token counts per class. Save accuracy, balanced accuracy, NLL,
confusion matrices, coefficient norms, convergence status, and accuracy by
token position. Treat line-search warnings as a failed numerical fit, not as a
harmless warning.

### Gate decision

- If block-output balanced accuracy is at least 0.80 on the binary pilot and
  NLL beats uniform, retain block output as the primary common coordinate
  system.
- If pre-MLP passes but block output is below 0.65, plan to compute and validate
  a pre-MLP Assistant Axis as a representation-site adaptation.
- If both sites are below 0.65, stop and debug token labels, Harmony rendering,
  hook placement, and the classifier before any persona run.
- If results are between thresholds, expand only the role pilot to 50 passages;
  do not start persona generation yet.

After the binary pilot passes, add `tool` and then `cot` one at a time. Do not
begin with a five-way probe. Record the performance loss introduced by each
additional role.

### Compact replacement for the lost probes

Once hook placement and numerical fitting pass, regenerate a four-way GPT-OSS
probe over `system`, `user`, `cot`, and `assistant` using 50 neutral passages
and the exact pinned prompt construction and split convention from the prior
run. Reuse the 20 pilot passages as part of these 50. The tool rendering may
still be generated for pipeline fidelity, but tool tokens were excluded from
the reported four-way role space.

Regenerate only layers 12 and 16 at both activation sites in the default run.
If hooks can collect nearby layers in the same forward at negligible generation
cost and without material memory pressure, layers 8, 10, 14, and 18 may also be
kept. Do not rerun the corpus solely to add them. Do not select layers by
looking for the largest eventual Assistant-Axis cosine.

The pre-MLP reproduction is a regression test. With the same surviving corpus
and split, expected held-out accuracies are approximately:

```text
layer 12: 0.7055
layer 16: 0.8851
```

The 50-passage subset is not expected to reproduce these numbers exactly.
Require the same ordering, healthy NLL and convergence, and bootstrap intervals
that include or are reasonably near the surviving reference values. If the
layer-16 pre-MLP probe is below 0.80 balanced accuracy, expand the probe only to
100 passages and re-evaluate. A full 250-passage reproduction is optional and
requires an explicit decision after the pilot; it is not tomorrow's default.

Save compact raw probe artifacts for every layer and site:

- class labels and their order;
- uncentered coefficients and intercepts;
- centered multinomial coefficient directions;
- any feature-scaler means/scales;
- classifier and solver configuration;
- coefficient norms and finiteness checks;
- train/test prompt or base-sequence IDs;
- per-layer metrics and confusion matrices;
- model, tokenizer, prompt-template, corpus, and code digests.

Use a stable, non-pickle interchange artifact such as compressed NPZ or
safetensors for coefficients and metadata JSON/CSV for everything else. A
version-specific serialized estimator may be saved as a convenience but must
not be the only copy of the vectors.

### Avoid recreating unnecessary 49.5 GB storage

The prior exact run retained a 49.56 GB all-layer activation tensor. That is not
required merely to recover role directions. Prefer one of these strategies:

1. extract and fit one layer/site at a time, save the compact probe artifact,
   validate it, and release the activation matrix before the next layer; or
2. capture a small group of layers that fits comfortably in host RAM, fit all
   requested sites for that group, then release it.

Do not delete a stage's activations until its probe coefficients, metrics, and
hashes have been written and independently reloaded. If enough persistent
storage exists, temporary activations may be retained until the complete stage
passes and then removed. Record whether temporary files were removed and
whether recovery is possible.

Gate 3 is complete when validated, reloadable pilot role-probe directions exist
at the same activation site and candidate layers that will be used for the
Assistant Axis comparison. Label these coefficients as subset-derived rather
than exact replacements for the lost full-corpus probes.

## Gate 4: small Assistant Axis pilot

Use the official `safety-research/assistant-axis` code and role list as the
source of truth. Vendor or pin an exact upstream commit; do not silently
reimplement or alter the persona definitions. Record the commit and hashes of
all persona prompts.

### Pilot persona panel

Select a deterministic, stratified panel of exactly 12 personas, three from
each of four broad families:

- helpful or professional: `tutor`, `consultant`, `mediator`;
- ordinary human occupations: `librarian`, `engineer`, `bartender`;
- theatrical or fictional: `pirate`, `comedian`, `genie`; and
- spiritual, mystical, or adversarial: `mystic`, `anarchist`,
  `devils_advocate`.

The default Assistant is a separate reference condition and is not counted as
one of the 12 personas.

Use official extraction-question IDs 0 and 5 for every persona and for the
default condition. Before this run, execute a seven-generation micro-pilot: six
of the final personas (`tutor`, `librarian`, `engineer`, `pirate`, `mystic`, and
`anarchist`) on the first question plus its default-Assistant response.
Inspect formatting, role adherence, token boundaries, and both activation
captures. If it passes, generate only the missing 19 responses to complete the
12-by-two panel. Keep generation deterministic where the official method
permits, use the lowest compatible reasoning effort, and cap response length at
256 generated tokens unless the official protocol requires more. Preserve raw
prompts, token IDs, responses, finish reasons, and generation metadata.

Run the official role-adherence judge only once per response. Store raw judge
outputs and parsed scores separately. Manual-review a stratified sample,
including all parse failures and boundary scores.

### Extraction

Replay only responses that pass the official role-adherence threshold. During
one replay, stream response-token means at both activation sites for layers
8-20. Do not store all token activations unless a later analysis requires them.

For each response, save:

- persona and question IDs;
- default/persona label;
- judge score and inclusion decision;
- response token count;
- per-layer mean pre-MLP vector;
- per-layer mean block-output vector.

Store vectors in float32 after accumulation. Accumulate sums in float32 or
float64; do not average in float16.

### Axis computation

At each layer and activation site, compute:

```text
assistant_axis = mean(default Assistant activations)
               - mean(included persona activations)
```

Also compute per-persona vectors relative to the same neutral/default
reference, following the official implementation exactly. Fit PCA on the
persona vectors separately for each layer and site. PCA signs are arbitrary;
orient PC1 so its dot product with the Assistant Axis is non-negative before
reporting a signed cosine.

### Pilot validation metrics

Report, for every layer and site:

- cosine between Assistant Axis and oriented PC1;
- PC1, PC2, and PC3 explained-variance ratios;
- axis norm and per-persona projection distribution;
- held-out default-versus-persona AUROC and balanced accuracy;
- bootstrap confidence intervals over personas, not tokens, prominently marked
  as pilot estimates;
- leave-one-persona-out stability of the axis;
- split-half cosine stability across questions;
- cosine with same-site role-probe directions;
- principal angles between the Assistant Axis/persona subspace and role-probe
  subspace.

The train/test split must be grouped by persona and question as appropriate.
Never treat response tokens as independent samples for confidence intervals.

### Pilot acceptance criteria

An activation site is promising if all of the following hold at one or more
middle layers:

- held-out default-versus-persona AUROC is at least 0.80;
- split-half axis cosine is at least 0.80;
- leave-one-persona-out axes are reasonably stable and not visibly driven by
  one role family;
- projection direction is semantically consistent with default Assistant
  behavior;
- no major judge or prompt-formatting failure is found.

PC1 alignment is supporting evidence, not a hard gate. Report it as:

- strong: absolute cosine at least 0.80;
- moderate: 0.50-0.80;
- weak: below 0.50.

If the contrast axis passes held-out separation and stability while PC1
alignment is weak, continue but describe the result as a meaningful Assistant
direction that is not the dominant persona component in GPT-OSS.

With only 12 personas and two questions, PCA and uncertainty estimates are
diagnostics, not paper-quality estimates. Do not reject a plausible axis solely
because a confidence interval is wide; use the result to decide whether a
larger run is worth its cost.

## Gate 5: single-category CoT-Forgery pilot

Use the official StrongREJECT small dataset and pin its exact revision. Filter
to the category `Disinformation and deception`, yielding 10 examples. Save the
source row IDs, exact category string, prompt digests, dataset revision, and
file hash. Do not substitute the first 10 rows of the full dataset: the intended
sample is all 10 rows from this category in the official balanced 60-example
small split.

The expensive forgeries have already been preserved locally. The artifact
`data/qwen3-32b-cot-forgery-20260826T151559Z/policies.csv` contains 1,252 policy
conditions, including 50 successful `base`/`no_qualifier` policies for this
category. Its recorded SHA-256 is
`31dc0f40d7826ef74ae8c52d30c87c36388c1825f495a9f691e04e85a54c0e1d`;
the corresponding full StrongREJECT source CSV hash is
`4dd70357e4ff8b5d0ba5ebafecab5d6dd5633ce8046e3dd1c8bd93e64de44381`.
Verify both before reuse.

Join the 10 small-dataset rows to this archive by the SHA-256 of the exact
forbidden-prompt text, never by row number. Expect 10 one-to-one matches. If a
small-dataset row does not match, do not regenerate it during the H100 session;
instead use a deterministic SHA-256-ranked sample of 10 from the 50 existing
category policies and label the result as a custom category subset rather than
StrongREJECT-small.

This category is a compute-control choice, not a claim that it represents the
whole benchmark. It provides a deterministic coherent slice while avoiding a
mixture of six categories. Report every rate as a numerator out of 10 and do
not present it as a full-benchmark ASR.

### Reuse before regeneration

For each selected request, reuse the saved policy whose exact base-request
digest and attack-template revision match. Tomorrow's forgery-generation budget
is zero. If provenance or matching fails, stop this stage and preserve the
diagnostic; do not spend GPU time or API calls recreating policies ad hoc.

Use only the repository's established `base` style with `no_qualifier`. This is
one of the existing experimental branches, not a newly simplified attack. Do
not generate the `destyled` branch or any of the four qualifier branches.

The current scripts assume 313 source rows and materialize 1,252 policy
conditions, so they must not be run unmodified. Add an explicit selection
manifest or `--source-id-file`/`--category` option that is applied before job
creation, then change completeness checks to validate the manifest's expected
IDs rather than the constants 313 and 1,252. Do not use `--max-new-attempts` or
`--max-rows 10` as the sampling mechanism: those select work by execution order
and do not guarantee the intended category.

### Conditions and call budget

Run only two target-model conditions per request:

1. the raw StrongREJECT request under the pinned baseline chat template; and
2. the same request under the pinned CoT-Forgery construction.

That is 20 GPT-OSS generations total. Use one fixed seed/decoding setting and
the same 1,024-token output cap in both conditions. Do not retry a completed
generation; retry only a documented infrastructure failure. Do not run destyled forgeries,
qualifier attacks, steering sweeps, multiple seeds, or other jailbreak methods
in this pilot.

Judge each saved response once with the same StrongREJECT rubric used in the
repository. Store raw judge output, parsed score, refusal status, token counts,
and paired success/failure. Report baseline and attack successes out of 10,
the paired change, score distribution, and exact binomial intervals. With this
sample size, effect direction and saturation are more important than a precise
percentage.

### Gate decision

- If attack success is between 2/10 and 8/10, preserve the pilot as the initial
  operating point; it is informative enough for later mechanistic work.
- If it is 0/10 or 1/10, first audit attack construction and chat rendering. Do
  not immediately spend on another category.
- If it is 9/10 or 10/10, the slice is saturated. It can still support paired
  activation analysis, but not fine-grained behavioral comparisons.
- A second 10-example category, extra seeds, or the full 60/313-example dataset
  requires an explicit user decision after reviewing costs and this report.

## Gate 6: optional expansion

Proceed only after the pilot report is reviewed.

Possible expansions are the full official persona collection, more questions,
a full role-probe reproduction, another StrongREJECT category, or a broader
jailbreak evaluation. None is an automatic next step. Keep generation, judging,
and inclusion thresholds faithful to the pinned implementations. Generate and
judge once, then extract both activation sites in the same replay.

Primary preregistered comparisons:

1. Does GPT-OSS block-output persona PC1 align with the Assistant contrast
   direction?
2. Does the pre-MLP site show the same geometry?
3. At which site is the Assistant direction most stable out of sample?
4. At the same site, how does the Assistant direction relate geometrically to
   architectural role-probe directions?
5. Are apparent similarities robust after controlling for token position,
   response length, and persona family?

Do not select a layer solely by maximizing cosine with the existing role
probe. Select using persona-pilot validation metrics, then report role-vector
comparisons at that fixed layer. Also report layers 12 and 16 because they are
the existing GPT-OSS paper-comparison and role-probe-best layers.

## Causal sanity check

This is not part of tomorrow's default compute budget. Run it only after the
pilot report is reviewed and an explicit decision is made to spend additional
generation time.

Only after geometric validation, run a small steering test on held-out benign
persona prompts.

At the selected layer/site, test coefficients:

```text
-2, -1, 0, +1, +2
```

Use a small fixed prompt set and inspect whether positive steering makes
responses more default-Assistant-like and negative steering increases persona
adoption or drift. Include an equal-norm random direction and a shuffled axis
as controls. Measure ordinary task quality so that incoherence is not mistaken
for persona movement.

Stop immediately if hooks corrupt logits at coefficient zero, generation
becomes broadly degenerate, or the effect is driven only by response length.
The Gate-5 pilot is the only harmful jailbreak evaluation authorized by this
plan. Keep its prompts and outputs in access-controlled persistent storage and
commit only aggregate metrics and non-sensitive metadata.

## Recommended artifact layout

Use persistent storage for all expensive or large artifacts:

```text
/workspace/role-probe-storage/outputs/gptoss20b-assistant-axis-YYYYMMDD-HHMM/
  run-metadata.json
  environment.json
  upstream-provenance.json
  prompt-manifest.csv
  generation-settings.json
  responses.jsonl.gz
  judge-raw.jsonl.gz
  judge-scores.csv
  inclusion-manifest.csv
  hook-validation.json
  role-pilot/
  role-probes/
    pre-mlp-probes.npz
    block-output-probes.npz
    probe-metrics.csv
    per-role-metrics.csv
    confusion-matrices.csv
    accuracy-by-token-position.csv
    split-manifest.csv
  activations/
    pre-mlp-response-means.pt
    block-output-response-means.pt
  analysis/
    axis-by-layer-pre-mlp.pt
    axis-by-layer-block-output.pt
    persona-vectors-pre-mlp.pt
    persona-vectors-block-output.pt
    pca-metrics.csv
    heldout-separation.csv
    stability.csv
    role-axis-cosines.csv
    principal-angles.csv
  steering-smoke/
  cot-forgery-pilot/
    dataset-manifest.csv
    forgeries.jsonl.gz
    responses.jsonl.gz
    judge-raw.jsonl.gz
    paired-results.csv
    summary.json
  sha256sums.txt
```

Commit only compact reports, CSVs, JSON metadata, and plots. Do not commit model
weights, `.env`, raw activation tensors, API keys, or large response dumps.

## Reproducibility requirements

Every report must state:

- exact repository commit and dirty-worktree status;
- Assistant Axis upstream commit;
- model and tokenizer revisions;
- Harmony system/developer prompt and rendering convention;
- generation parameters and random seeds;
- judge model, prompt, settings, and parsing rules;
- layer-number convention;
- exact activation-site definitions;
- token inclusion/exclusion rules;
- split grouping and bootstrap unit;
- numerical dtype at extraction, accumulation, fitting, PCA, and saving;
- all warnings and failed samples;
- GPU time and peak memory;
- SHA-256 hashes for immutable artifacts.

## Operational stop conditions

Stop the run and preserve diagnostics if any of these occur:

- hook validation changes zero-intervention logits;
- pre-MLP capture disagrees with the exact prior convention;
- the subset-derived layer-16 pre-MLP probe is below 0.80 balanced accuracy
  after one expansion from 50 to 100 passages;
- role-pilot NLL is worse than uniform or the solver does not converge;
- response/persona IDs cannot be joined one-to-one across stages;
- judge parse failures exceed 2%;
- fewer than half of the pilot personas have enough accepted responses;
- axis stability is below 0.5 cosine across split halves;
- non-finite activations, PCA inputs, coefficients, or projections appear;
- persistent storage is unavailable;
- the H100 shutdown window is too short to finish the current atomic stage.

Each stage must write to a temporary stage directory and rename it to its final
name only after validation. If interrupted, leave the temporary directory for
inspection and do not mark the stage complete.

## End-of-session checklist

1. Confirm every completed stage is on persistent storage.
2. Independently reload the regenerated role-probe NPZ/safetensors artifact and
   verify its labels, layers, activation sites, shapes, and finite values.
3. Generate SHA-256 hashes for immutable compact artifacts.
4. Copy compact metrics and plots into a new dated directory under `reports/`.
5. Write a README that clearly distinguishes every subset pilot from a
   full-run result and states that the earlier raw probes had been lost.
6. Run `git status --short` and preserve unrelated user changes.
7. Commit only the intended code and compact report artifacts.
8. Push the branch if credentials and authorization are available.
9. Stop Jupyter and other GPU processes.
10. Verify the H100 instance is stopped in the provider dashboard; exiting SSH
   alone does not stop billing.

## Expected decision after tomorrow's pilot

The pilot should end with one of these explicit recommendations:

- **Block-output path:** role decoding and Assistant geometry both validate at
  block output; use this as the common coordinate system.
- **Pre-MLP adaptation:** role decoding and Assistant geometry validate only at
  pre-MLP; continue with a clearly labeled representation-site adaptation.
- **Dual-site result:** both validate but encode meaningfully different
  geometry; report both and avoid treating them as interchangeable.
- **Stop:** neither site produces stable held-out Assistant geometry or valid
  role decoding; debug methodology before spending more H100 time.
