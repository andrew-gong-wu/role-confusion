#!/usr/bin/env python3
"""Run checks that do not require CUDA, large models, or third-party packages."""

import json
import platform
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    print(f"Platform: {platform.platform()}")
    print(f"Machine: {platform.machine()}")
    print(f"Python: {platform.python_version()}")
    if sys.version_info < (3, 9):
        print("ERROR: local checks require Python 3.9+")
        return 1

    subprocess.run([sys.executable, str(ROOT / "scripts/prepare_demo.py")], check=True)
    with (ROOT / "demo/role-probe-demo.ipynb").open() as handle:
        notebook = json.load(handle)
    if not notebook.get("cells"):
        print("ERROR: generated notebook contains no cells")
        return 1

    result = subprocess.run(
        [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"],
        cwd=ROOT,
        check=False,
    )
    if result.returncode:
        return result.returncode

    print("Local preparation checks passed.")
    print("CUDA/model checks were intentionally skipped on this machine.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

