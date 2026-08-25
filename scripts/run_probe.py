#!/usr/bin/env python3
"""Execute only the probe-training portion of the generated demo notebook."""

import os
from pathlib import Path

import nbformat
from dotenv import load_dotenv
from nbclient import NotebookClient


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "demo/role-probe-demo.ipynb"


def probe_only_notebook():
    notebook = nbformat.read(NOTEBOOK_PATH, as_version=4)
    stop_index = None
    for index, cell in enumerate(notebook.cells):
        if cell.cell_type == "code" and "Saved probe artifacts" in cell.source:
            stop_index = index
            break
    if stop_index is None:
        raise RuntimeError("Could not find the end of the probe-training section")
    notebook.cells = notebook.cells[: stop_index + 1]
    return notebook


def main() -> None:
    load_dotenv(ROOT / ".env")
    output_dir = Path(os.environ.get("ROLE_PROBE_OUTPUT_DIR", ROOT / "outputs"))
    output_dir.mkdir(parents=True, exist_ok=True)
    executed_path = output_dir / "executed-role-probe.ipynb"

    notebook = probe_only_notebook()

    def announce(cell, cell_index, **_kwargs):
        first_line = next(
            (line.strip() for line in cell.source.splitlines() if line.strip()),
            "empty cell",
        )
        print(
            f"CELL {cell_index + 1}/{len(notebook.cells)}: {first_line[:100]}",
            flush=True,
        )

    client = NotebookClient(
        notebook,
        timeout=7200,
        kernel_name=os.environ.get("ROLE_PROBE_KERNEL", "role-probe"),
        resources={"metadata": {"path": str(ROOT)}},
        on_cell_start=announce,
    )

    print(f"Input notebook: {NOTEBOOK_PATH}", flush=True)
    print(f"Artifacts: {output_dir}", flush=True)
    print(
        "Settings: "
        f"samples={os.environ.get('ROLE_PROBE_N_SAMPLES', '150')}, "
        f"max_seqlen={os.environ.get('ROLE_PROBE_MAX_SEQLEN', '512')}, "
        f"batch={os.environ.get('ROLE_PROBE_BATCH_SIZE', '32')}, "
        f"split={os.environ.get('ROLE_PROBE_SPLIT_GROUP', 'prompt_ix')}",
        flush=True,
    )
    try:
        client.execute()
    finally:
        nbformat.write(notebook, executed_path)
        print(f"Saved executed notebook: {executed_path}", flush=True)
    print("Probe-only notebook completed successfully.", flush=True)


if __name__ == "__main__":
    main()
