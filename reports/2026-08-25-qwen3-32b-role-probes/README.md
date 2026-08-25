# Qwen3-32B five-role probes and assistant-axis comparison

This directory contains a preliminary Qwen3-32B replication of the existing
role-probe paradigm and a cosine comparison with the downloaded assistant
persona axis from Lu et al. (2026). No text generation or new behavioral
experiment was run.

## Method

- Reused the 250 neutral passages and seed 123 split from the prior GPT-OSS run.
- Rendered each passage as Qwen-native `system`, `user`, tool-response, thinking
  (`cot`), and `assistant` context.
- Fit token-level five-way multinomial logistic probes at layers
  0, 4, 8, ..., 60 with the prior cuML settings (`C=0.005`, L2 penalty, 10%
  held-out prompts).
- Captured decoder-layer outputs rather than GPT-OSS pre-MLP states. This is the
  representation site used to construct the downloaded assistant persona axis,
  so both kinds of vector are in the same 5,120-dimensional space.
- Selected the layer with maximum held-out token accuracy, breaking ties toward
  the lower layer.
- Subtracted the mean coefficient vector across the five softmax classes before
  computing role-vector cosines. This removes the multinomial classifier's
  common-mode component. The persona axis itself is already a contrast vector.

## Result

Layer 48 was selected with held-out accuracy **0.357030** across 69,969 tokens.
Per-role accuracies at that layer were system 0.611613, user 0.326955, tool
0.183194, CoT 0.596404, and assistant 0.204396.

The assistant persona axis is nearly orthogonal to every centered role-probe
direction at the selected layer: its absolute cosine is at most 0.019. This is
consistent with persona-assistantness and message-role identity being distinct
directions in this preliminary setup. It should not be read as a strong null
result: five-way probe accuracy is modest, especially for Qwen tool and
assistant tokens, and cuML reported L-BFGS line-search early-stop warnings.

## Files

- `centralized-vectors.npz`: persona axis, raw selected-layer role coefficients,
  centered role directions, labels, and selected layer.
- `all-layer-role-probe-vectors.npz`: raw coefficients and intercepts for all 16
  tested layers.
- `cosine-similarity.csv`: the 6-by-6 cosine matrix.
- `cosine-similarity-heatmap.png`: annotated heatmap of that matrix.
- `overall-accuracy.csv`, `per-role-accuracy.csv`, and `layer-selection.json`:
  probe selection metrics.
- `prompt-summary.json`, `prompt-split.csv`, `smoke-validation.json`, and
  `run-metadata.json`: reproducibility metadata.

The standalone runner is `scripts/run_qwen_role_probe_cosines.py`.
