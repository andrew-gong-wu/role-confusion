#!/usr/bin/env python3
"""Fail-fast diagnostics for the paid CUDA session."""

import importlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv


REQUIRED_MODULES = (
    "torch",
    "transformers",
    "datasets",
    "pandas",
    "numpy",
    "sklearn",
    "cupy",
    "cuml",
)


def main() -> int:
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    versions = {}
    failed = []
    for name in REQUIRED_MODULES:
        try:
            module = importlib.import_module(name)
            versions[name] = getattr(module, "__version__", "unknown")
        except Exception as exc:  # diagnostics should report all import failures
            failed.append(f"{name}: {exc}")

    if failed:
        print("Import failures:")
        print("\n".join(f"- {item}" for item in failed))
        return 1

    import cupy
    import torch
    import transformers

    if not torch.cuda.is_available():
        print("ERROR: PyTorch cannot see a CUDA device.")
        return 1
    if int(transformers.__version__.split(".", 1)[0]) != 5:
        print(f"ERROR: demo requires Transformers 5.x; found {transformers.__version__}")
        return 1

    props = torch.cuda.get_device_properties(0)
    total_gib = props.total_memory / 1024**3
    probe = cupy.arange(4)
    if int(probe.sum().get()) != 6:
        print("ERROR: CuPy smoke test returned an unexpected result.")
        return 1

    report = {
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "gpu": props.name,
        "gpu_memory_gib": round(total_gib, 2),
        "torch_cuda": torch.version.cuda,
        "versions": versions,
        "hf_home": os.environ.get("HF_HOME"),
        "model_revision": os.environ.get("ROLE_PROBE_MODEL_REVISION"),
        "c4_revision": os.environ.get("ROLE_PROBE_C4_REVISION"),
    }
    driver = subprocess.run(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
        check=False,
        capture_output=True,
        text=True,
    )
    report["nvidia_driver"] = driver.stdout.strip() or "unknown"
    git_head = subprocess.run(
        ["git", "rev-parse", "HEAD"], check=False, capture_output=True, text=True
    )
    report["project_git_commit"] = git_head.stdout.strip() or "unknown"
    print(json.dumps(report, indent=2, sort_keys=True))
    if total_gib < 70:
        print("WARNING: less than 70 GiB VRAM; lower ROLE_PROBE_BATCH_SIZE.")

    storage_root = Path(os.environ.get("ROLE_PROBE_STORAGE_ROOT", "outputs"))
    output = storage_root / "environment"
    output.mkdir(parents=True, exist_ok=True)
    (output / "h100-diagnostics.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    freeze = subprocess.run(
        [sys.executable, "-m", "pip", "freeze"],
        check=True,
        capture_output=True,
        text=True,
    )
    (output / "requirements-freeze.txt").write_text(freeze.stdout)
    print("H100 diagnostics passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
