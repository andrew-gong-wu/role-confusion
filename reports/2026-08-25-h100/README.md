# H100 probe reproduction — 2026-08-25

## Outcome

The probe-only `gpt-oss-20b` demonstration completed end to end on one H100.
Role identity becomes strongly linearly decodable in later layers, peaking at
0.874 accuracy for the upstream-style split and 0.896 for the grouped-by-source
split. Four-way chance accuracy is 0.25.

| Layer | Upstream-style `prompt_ix` split | Grouped `base_seq_ix` split |
|---:|---:|---:|
| 0 | 0.134 | 0.322 |
| 4 | 0.277 | 0.539 |
| 8 | 0.787 | 0.860 |
| 12 | 0.677 | 0.798 |
| 16 | **0.874** | **0.896** |
| 20 | 0.815 | 0.857 |

The grouped split keeps all role-rendered copies of a source passage on the
same side of the train/test boundary. It performed better at every layer in
this run, so identical-content overlap in the upstream-style split does not
explain the late-layer decoding result here. The upstream-style split is not
class-balanced by construction, which may contribute to its lower and noisier
accuracy.

## Experiment settings

- Upstream code snapshot: `ec333c40fd43fe991e1ebf66765051b6d7e35784`
- Local project commit used for final runs: `a14bc6c575edf81053c9f17feb9747fb6f187c5f`
- Model: `openai/gpt-oss-20b`
- Model revision: `6cee5e81ee83917806bbde320786a8fb61efebee`
- Source data: 150 passages, half C4 and half Dolma 3
- C4 revision: `f3b95a11ff318ce8b651afc7eb8e7bd2af469c10`
- Dolma 3 revision: `3a8349c`
- Maximum source length: 512 tokens
- Rendered roles: system, user, CoT/analysis, assistant, and tool
- Probe classes: system, user, CoT/analysis, and assistant
- Layers: 0, 4, 8, 12, 16, and 20
- Batch size: 32
- Seed: 123
- Probe: cuML L2 logistic regression, `C=5e-3`, maximum 2,000 iterations
- Captured activation matrix at each retained layer: 266,005 × 2,880
- First-batch diagnostic perplexity: 216.89

## Environment

- GPU: NVIDIA H100 80GB HBM3 (79.18 GiB visible)
- Driver: 580.126.09
- Python: 3.12.14
- PyTorch: 2.9.1+cu128
- Transformers: 5.15.1
- Datasets: 5.0.1
- CuPy: 14.2.0
- cuML: 25.10.00
- scikit-learn: 1.7.2

See `environment/h100-diagnostics.json` and
`environment/requirements-freeze.txt` for the captured machine and package
details.

## Compatibility changes from the pinned upstream setup

The upstream files were preserved unchanged under `vendor/upstream`. The
generated runnable notebook and setup required these compatibility changes:

- Transformers 5 instead of the upstream setup script's contradictory 4.57.5.
- `kernels==0.16.0`, required by Transformers 5.15 for MXFP4 and the requested
  hub attention kernel.
- RAPIDS cuDF/cuML 25.10.0 because the upstream 25.9 build was unavailable as a
  released package.
- scikit-learn 1.7.2 because cuML 25.10 is incompatible with scikit-learn 1.9.
- `zstandard` for streaming the pinned Dolma 3 files.
- C4's pinned `en/` directory selected with `data_dir="en"` under Datasets 5.
- The virtual environment placed on pod-local disk; model cache, diagnostics,
  and results remained on the RunPod network volume.

## Artifacts

Small, reviewable artifacts copied into this report:

- `baseline-prompt-split/probe-accuracy.csv`
- `grouped-source-split/probe-accuracy.csv`
- `smoke/probe-accuracy.csv`
- `environment/h100-diagnostics.json`
- `environment/requirements-freeze.txt`

The executed notebooks and serialized probe objects remain on persistent remote
storage under `/workspace/role-probe-storage/outputs/`. Probe pickle files must
only be loaded from this trusted experiment.

## Limitations and next checks

This reproduces the authors' lightweight demonstration settings, not the full
paper's hyperparameter search or all role-confusion plots. The result is a
single seed. Next steps should include multiple seeds, confusion matrices and
per-class metrics, followed by applying the trained probes to the controlled
correct-tag, no-tag, and conflicting-tag conversations.
