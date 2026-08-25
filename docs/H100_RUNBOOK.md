# First H100 session runbook

Commands containing `THE_...` are placeholders; replace them with the values
Cambria provides.

## Before the GPU starts

1. Make sure all code is committed and pushed to GitHub.
2. Ask where the persistent volume will be mounted. This guide assumes
   `/workspace/role-probe-storage`.
3. Run the free local checks: `python3 scripts/check_local.py`.
4. Ask for the H100 roughly 30 minutes before the planned session.

## Connect over SSH

Paste the provided command into Terminal. It will resemble:

```bash
ssh root@THE_IP_ADDRESS -p THE_SSH_PORT
```

The first connection may ask whether to trust the host. Check that the address
matches the provided machine, then enter `yes`. Everything typed after
connecting runs on the remote Linux machine. Confirm the GPU:

```bash
nvidia-smi
```

The output should name an H100 and show roughly 80 GB or more of GPU memory.

## Clone and configure

```bash
git clone THE_GITHUB_REPOSITORY_URL
cd role-confusion
cp .env.example .env
nano .env
```

Fill `HF_TOKEN` if needed and confirm the volume paths. In nano, save with
Control-O, Enter, then exit with Control-X. Never commit `.env`.

## Install and validate CUDA

```bash
bash scripts/setup_h100.sh
```

This keeps the Python environment on the pod's fast local disk while putting
the model cache, diagnostics, and results on persistent storage. It generates
the adapted notebook and runs CUDA diagnostics. Do not start the experiment unless the final output says
`H100 diagnostics passed.` Then activate the environment:

```bash
source ~/role-confusion/.venv/bin/activate
```

The environment is intentionally disposable and can be rebuilt. Network
volumes may return stale-file-handle errors when used for thousands of small
package files; the expensive model download and experiment artifacts remain
persistent.

## Run a cheap smoke test

Edit `.env` to use:

```dotenv
ROLE_PROBE_N_SAMPLES=10
ROLE_PROBE_BATCH_SIZE=1
ROLE_PROBE_SPLIT_GROUP=prompt_ix
```

This still loads the full model, but reduces activation work. It is a pipeline
test, not the reported experiment. Start Jupyter remotely:

```bash
jupyter lab --ip=127.0.0.1 --no-browser --port=8888
```

Leave that terminal open. In a second laptop Terminal window, create a tunnel:

```bash
ssh -N -L 8888:127.0.0.1:8888 root@THE_IP_ADDRESS -p THE_SSH_PORT
```

Open `http://127.0.0.1:8888`, paste the printed token if requested, open
`demo/role-probe-demo.ipynb`, and choose `Role probe (H100)`. Run only through
**Train probes**; the later CoT Forgery section is outside the current scope.

## Run the baseline

After the smoke test succeeds, restart the kernel and restore:

```dotenv
ROLE_PROBE_N_SAMPLES=150
ROLE_PROBE_BATCH_SIZE=32
ROLE_PROBE_SPLIT_GROUP=prompt_ix
```

If batch size 32 runs out of memory, try 16, then 8, and record the change. The
notebook writes `role-probes.pkl` and `probe-accuracy.csv` to
`ROLE_PROBE_OUTPUT_DIR`.

## Run the leakage check

The upstream demo gives every role-rendered copy a unique `prompt_ix`, allowing
identical source content across train and test. After the baseline, rerun with:

```dotenv
ROLE_PROBE_SPLIT_GROUP=base_seq_ix
ROLE_PROBE_OUTPUT_DIR=/workspace/role-probe-storage/outputs/grouped-split
```

This keeps all role copies of a source passage together. Report it separately
from the directly comparable upstream-demo result.

## Finish safely

1. Confirm the CSV, probe pickle, and environment JSON are on the persistent
   volume.
2. Back up the small CSV and metadata files.
3. Commit and push code changes, but not secrets, weights, or probe pickles.
4. Stop Jupyter with Control-C and leave SSH with `exit`.
5. Stop/terminate the H100 in the dashboard or ask Cambria to do so.

Closing Terminal or running `exit` does **not** necessarily stop billing.
