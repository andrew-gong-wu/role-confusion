#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ -f "$PROJECT_DIR/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$PROJECT_DIR/.env"
  set +a
fi

STORAGE_ROOT="${ROLE_PROBE_STORAGE_ROOT:-/workspace/role-probe-storage}"
VENV_DIR="${ROLE_PROBE_VENV_DIR:-${STORAGE_ROOT}/venv}"
HF_HOME="${HF_HOME:-${STORAGE_ROOT}/huggingface}"
UV_CACHE_DIR="${UV_CACHE_DIR:-${STORAGE_ROOT}/uv-cache}"
ROLE_PROBE_OUTPUT_DIR="${ROLE_PROBE_OUTPUT_DIR:-${STORAGE_ROOT}/outputs}"

export HF_HOME UV_CACHE_DIR ROLE_PROBE_STORAGE_ROOT ROLE_PROBE_OUTPUT_DIR
export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-120}"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "ERROR: nvidia-smi is unavailable; run this on the NVIDIA GPU machine." >&2
  exit 1
fi

mkdir -p "$STORAGE_ROOT" "$HF_HOME" "$UV_CACHE_DIR"
cd "$PROJECT_DIR"

nvidia-smi

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="${HOME}/.local/bin:${PATH}"
fi

uv python install 3.12
if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  uv venv "$VENV_DIR" --python 3.12 --seed
fi

uv pip install --python "$VENV_DIR/bin/python" \
  --index-url https://download.pytorch.org/whl/cu128 \
  torch==2.9.1

uv pip install --python "$VENV_DIR/bin/python" \
  "transformers>=5,<6" \
  accelerate==1.12.0 \
  datasets \
  hf_transfer==0.1.9 \
  kernels==0.11.5 \
  compressed-tensors==0.13.0 \
  tiktoken==0.12.0 \
  blobfile==3.1.0 \
  pandas numpy scikit-learn plotly \
  python-dotenv packaging requests pyyaml tqdm termcolor \
  jupyterlab jupyter_server ipykernel ipywidgets nbformat notebook

uv pip install --python "$VENV_DIR/bin/python" \
  libucx-cu12==1.18.1 ucx-py-cu12==0.45.0
uv pip install --python "$VENV_DIR/bin/python" \
  --extra-index-url https://pypi.nvidia.com \
  "cudf-cu12==25.9.*" "cuml-cu12==25.9.*"

"$VENV_DIR/bin/python" -m ipykernel install --user \
  --name role-probe \
  --display-name "Role probe (H100)"

"$VENV_DIR/bin/python" scripts/prepare_demo.py
"$VENV_DIR/bin/python" scripts/check_h100.py

echo
echo "Setup complete. Activate with:"
echo "  source $VENV_DIR/bin/activate"
echo "Then launch from the repository root with:"
echo "  jupyter lab --ip=127.0.0.1 --no-browser --port=8888"
