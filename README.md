# Reproducing Role-Confusion Linear Probes

This repository prepares a probe-only reproduction of [*Prompt Injection as
Role Confusion*](https://arxiv.org/abs/2603.12277), following the authors'
[`gpt-oss-20b` demonstration](https://github.com/role-confusion/prompt-injection-as-role-confusion/blob/master/demo/role-probe-demo.ipynb).

The experiment places identical neutral text under different genuine chat
roles, extracts token activations, and trains a linear classifier to recover the
role. It then applies the probe where writing style and architectural role
disagree.

## What can run where

On an Apple Silicon or other non-NVIDIA laptop:

- edit code and notebooks;
- use Git/GitHub with a collaborator;
- validate role rendering and dataset bookkeeping;
- inspect saved CSV results and create plots.

On the remote NVIDIA H100:

- install CUDA, CuPy, and RAPIDS cuML;
- load `openai/gpt-oss-20b`;
- extract hidden states;
- train and apply the GPU probes.

No NVIDIA GPU is required for the local checks. The H100 provides compute, not
version control: collaborators should continue to exchange code through GitHub.

## Repository layout

```text
demo/                         generated runnable notebook and helper
docs/H100_RUNBOOK.md          first-session SSH and execution checklist
scripts/check_local.py        free Mac-safe checks
scripts/setup_h100.sh         remote CUDA environment setup
scripts/check_h100.py         paid-machine fail-fast diagnostics
scripts/prepare_demo.py       reproducibly adapts the upstream notebook
src/role_probe/               CPU-only preparation helpers
tests/                        CPU-only tests
vendor/upstream/              untouched pinned upstream snapshot
```

The upstream snapshot is pinned to commit
`ec333c40fd43fe991e1ebf66765051b6d7e35784`. Its provenance and license are in
[`vendor/upstream/PROVENANCE.md`](vendor/upstream/PROVENANCE.md).

## Run the local preparation checks

The checks support the macOS system Python 3.9+ and install nothing:

```bash
python3 scripts/check_local.py
```

This command:

1. regenerates `demo/role-probe-demo.ipynb` from the pinned upstream copy;
2. validates the notebook JSON;
3. tests all five Harmony role renderings;
4. verifies identifiers needed for leakage-safe grouped splitting.

It intentionally does not download a model or pretend to validate CUDA.

## Configuration

`.env` is present locally and ignored by Git. `.env.example` documents shared
variable names without containing secrets:

```dotenv
HF_TOKEN=
ROLE_PROBE_STORAGE_ROOT=/workspace/role-probe-storage
HF_HOME=/workspace/role-probe-storage/huggingface
ROLE_PROBE_OUTPUT_DIR=/workspace/role-probe-storage/outputs
ROLE_PROBE_MODEL_REVISION=6cee5e81ee83917806bbde320786a8fb61efebee
ROLE_PROBE_C4_REVISION=f3b95a11ff318ce8b651afc7eb8e7bd2af469c10
ROLE_PROBE_N_SAMPLES=150
ROLE_PROBE_BATCH_SIZE=32
ROLE_PROBE_SPLIT_GROUP=prompt_ix
```

The generated notebook loads `.env`; it contains no hard-coded cache directory
or API key.

## Baseline settings

- Model: `openai/gpt-oss-20b`
- Model revision: `6cee5e81ee83917806bbde320786a8fb61efebee`
- Source sequences: 150, split between C4 and Dolma 3
- C4 revision: `f3b95a11ff318ce8b651afc7eb8e7bd2af469c10`
- Dolma 3 revision: `3a8349c` (as pinned upstream)
- Maximum content length: 512 tokens
- Probe layers: `0, 4, 8, 12, 16, 20`
- Seed: 123
- Probe roles: `system`, `user`, `cot`, `assistant`
- Classifier: L2 logistic regression, `C=5e-3`, 2,000 iterations
- H100 batch size: 32

Start with 10 sequences and batch size 1 as a smoke test. Restore the baseline
only after the full pipeline works.

## Important split distinction

The upstream demo splits by `prompt_ix`, where every role-rendered copy has a
different ID. This permits identical source text across train and test. Use
`ROLE_PROBE_SPLIT_GROUP=prompt_ix` for direct demo comparability, then rerun
with `base_seq_ix` so all role copies remain together. Report the grouped result
as a separate robustness check.

High probe accuracy establishes linear decodability; it does not by itself show
that the model causally uses the probe direction to follow instructions.

## Remote execution

Follow [`docs/H100_RUNBOOK.md`](docs/H100_RUNBOOK.md) when compute is available.
It covers SSH, persistent storage, installation, Jupyter tunneling, smoke tests,
the baseline, result preservation, and stopping billing.

To execute only the probe-training section without opening Jupyter or running
the later API-backed attack cells:

```bash
./.venv/bin/python scripts/run_probe.py
```

The pinned upstream snapshot has a dependency inconsistency: the notebook
demands Transformers 5 while its setup script pins 4.57.5. The local H100 setup
follows the notebook (`transformers>=5,<6`) and records the installed version.
Note this with the final results.

## Collaboration workflow

Use normal Git branches on both laptops and the H100:

```bash
git pull --ff-only
git switch -c your-name/short-task
# edit and test
git add README.md scripts src tests
git commit -m "Describe the change"
git push -u origin your-name/short-task
```

Do not commit `.env`, model caches, activations, notebooks containing secrets,
or generated probe pickles. Keep large artifacts on the persistent volume and
commit small summary tables only when intentionally reviewed.

## References

- [LessWrong explanation](https://www.lesswrong.com/posts/d8xDGzCEYE639qqEv/a-mechanistic-explanation-of-prompt-injection-and-why-you)
- [ICML paper](https://arxiv.org/abs/2603.12277)
- [Authors' reproduction repository](https://github.com/role-confusion/prompt-injection-as-role-confusion)
