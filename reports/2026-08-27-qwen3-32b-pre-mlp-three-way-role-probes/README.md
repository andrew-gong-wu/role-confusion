# Qwen3-32B pre-MLP three-way role probes

This directory preserves the completed `user` / `assistant` / `tool` probe run
from `cambria-union`. The run adapts the paper-style Qwen construction to dense
`Qwen/Qwen3-32B` and captures the output of each decoder layer's
`post_attention_layernorm`, immediately before the MLP.

No Qwen model weights are included.

## Selected result

Layer 48 was selected by maximum held-out token accuracy.

| Metric | Value |
|---|---:|
| Held-out accuracy | 0.453074 |
| Balanced held-out accuracy | 0.457564 |
| Held-out NLL | 1.129267 |
| Training tokens | 390,327 |
| Held-out tokens | 43,686 |
| Logistic-regression C | 0.1 |

Per-role recall at layer 48:

| Role | Recall | Held-out tokens |
|---|---:|---:|
| User | 0.430217 | 18,751 |
| Assistant | 0.678862 | 12,409 |
| Tool | 0.263612 | 12,526 |

The held-out majority-class baseline is approximately 0.4292, balanced chance
is 1/3, and uniform three-way NLL is `ln(3) = 1.0986`. The pre-MLP result is
therefore above balanced chance but remains weak: tool recall is below chance
and NLL is slightly worse than uniform prediction.

## Method

- Model: dense Qwen3-32B, thinking disabled by the controlled renderings.
- Data: 250 neutral C4/Dolma passages rendered under `user`, `assistant`, and
  `tool`, yielding 750 sequences and 434,013 target-content tokens.
- Activation site: `model.model.layers[i].post_attention_layernorm` output.
- Layers: 0, 4, ..., 60.
- Classifier: cuML multinomial L2 logistic regression, no feature scaling,
  fixed `C=0.1`.
- Split: seeded 90/10 split over rendered prompts, matching the released
  notebook implementation.
- Tags and neutral positional filler were excluded from probe training.
- Peak GPU allocation: 61.54 GiB.

cuML emitted nine L-BFGS line-search warnings. The run nevertheless completed
and wrote vectors and metrics for every requested layer.

## Comparison caveat

The earlier post-MLP three-way run used `skip_first_n_content_tokens=32`, while
this run used `0`. It is therefore not a strict one-variable activation-site
ablation. Its selected accuracy (0.4531) is higher than the earlier post-MLP
accuracy (0.3796), but a matched pre-MLP rerun with `--skip-first-n 32` is
needed to attribute that difference solely to activation site.

Direct cosine comparison with Lu et al.'s Assistant Axis was intentionally
omitted because that published axis is constructed from post-MLP block-output
activations, not this pre-MLP site.

## Preservation and provenance

- Persistent remote output:
  `/workspace/role-probe-storage/outputs/qwen3-32b-paper-three-way-pre-mlp-seed123-20260827`
- Persistent remote archive:
  `/workspace/role-probe-storage/archives/qwen3-32b-paper-three-way-pre-mlp-seed123-20260827.tar.gz`
- Remote base repository commit: `d0e91ffa24c7f232f18c06771a1d1e77f0632a0e`
- Remote archive SHA-256:
  `ea8d92cd36216fe3ae1a71d2ab05f023fb2b6b5da5b030444e258f9d1eda5399`

`sha256sums.txt` covers every extracted remote artifact. The downloaded archive
was verified against the hash above, and all extracted files were verified
against the internal checksum manifest.

The directory includes complete CSV/JSON metrics, probe vectors, prompt
manifest and split, smoke validation, full run log, dependency freeze, NVIDIA
diagnostics, exact source snapshots, and the original compressed archive.
