# Qwen3-32B hierarchical role probes (invalid methodology)

This directory preserves an exploratory Qwen3-32B hierarchical run and its
comparison with the downloaded assistant persona axis. It is **not a valid
replication of the paper's tag-controlled role-probe methodology** and should
not be cited as evidence that Qwen3-32B has a well-separated linear role space.
No text generation or behavioral experiment was performed.

## Setup

The runner created seven deterministic renderings for each of the 250 saved
neutral passages:

- minimal outer `system`, `user`, and `assistant` messages;
- ordinary user text and a native `<tool_response>` after a valid tool schema
  and tool call;
- native `<think>` content and final-answer content inside valid assistant
  turns.

For every passage, each condition contributed the same number of evenly spaced
target-content tokens, capped at 128. Scaffold and tag tokens were excluded.
The split was grouped by passage, so all seven renderings of a passage remained
entirely in train or test. This produced 216,573 activation rows, with 25 of 250
passages held out.

At layers 0, 4, ..., 60, the analysis fit four standardized logistic probes:

1. outer role: system vs user vs assistant;
2. user subtype: plain user vs tool response;
3. assistant subtype: final answer vs CoT;
4. a secondary five-way leaf probe for system/user/tool/CoT/assistant vectors.

Standardization parameters were estimated only on the training split. Fitted
coefficients were transformed back into the original 5,120-dimensional
decoder-layer-output coordinates, then class-centered before cosine comparison
with the persona axis.

## Why this run is invalid for the replication

The role conditions used different role-specific conversational scaffolds.
Those scaffolds introduced contextual features beyond the architectural tags,
so a classifier could distinguish conditions without recovering the
tag-induced role geometry described in the paper. The hierarchical design also
changed the token sampling and fitting procedure. Its saved classification
metrics therefore measure separability in this confounded construction, not a
valid tag-controlled role probe.

The corrected paper-style run holds neutral target text and positional controls
constant across roles. Its much lower accuracy is the relevant result. The
metrics in this directory are retained only as provenance for the discarded
exploratory run.

## Persona-axis comparison

The saved persona-axis comparisons were computed from the confounded probe
directions and are not interpretable as evidence for or against alignment
between the persona axis and a valid role space.

## Files

- `centralized-hierarchical-vectors.npz`: selected-layer persona and leaf-role
  vectors plus all three selected hierarchical heads.
- `all-hierarchical-probe-vectors.npz`: raw-coordinate coefficients and
  intercepts for all four heads at all 16 layers.
- `probe-accuracy.csv`, `per-class-accuracy.csv`,
  `layer-selection-scores.csv`, and `layer-selection.json`: held-out metrics and
  selection details.
- `cosine-similarity.csv` and `cosine-similarity-heatmap.png`: the selected
  six-vector comparison.
- `prompt-conditions.csv`, `passage-split.csv`, `prompt-summary.json`,
  `smoke-validation.json`, and `run-metadata.json`: reproducibility metadata.

The standalone runner is `scripts/run_qwen_hierarchical_role_probes.py`.
