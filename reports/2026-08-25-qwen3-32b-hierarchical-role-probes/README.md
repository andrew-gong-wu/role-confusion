# Qwen3-32B hierarchical role probes

This directory contains the context-valid hierarchical rerun of the Qwen3-32B
role probes and the comparison with the downloaded assistant persona axis. No
text generation or behavioral experiment was performed.

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

## Layer selection and accuracy

The selection criterion was the mean balanced accuracy of the first three
hierarchical heads, with the lower layer breaking ties. Layer **44** was
selected:

| Probe | Balanced accuracy |
|---|---:|
| Outer role | 0.727242 |
| User vs tool | 0.941406 |
| Assistant vs CoT | 0.949049 |
| Mean of hierarchical heads | **0.872566** |
| Secondary five-way leaf probe | **0.918546** |

The five-way held-out per-class accuracies were system 0.955163, user 0.950408,
tool 0.888587, CoT 0.904552, and assistant 0.894022. These are materially more
coherent than the preliminary flat-wrapper run, though they are not a direct
apples-to-apples accuracy comparison because the context construction, passage
split, and token sampling all changed.

cuML emitted L-BFGS line-search early-stop warnings for many fits. The returned
probes nevertheless produced stable, high held-out accuracies. The warnings are
preserved as a numerical caveat rather than silently treated as convergence.

## Persona-axis comparison

At layer 44, the persona assistant axis remains nearly orthogonal to every
centered leaf-role direction. Its cosines are -0.0003 with system, 0.0058 with
user, 0.0027 with tool, -0.0010 with CoT, and -0.0080 with assistant. Thus the
largest absolute persona/role cosine is 0.0080 in this setup.

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
