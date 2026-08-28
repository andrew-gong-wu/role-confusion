#!/usr/bin/env python3
"""Build square layer-18 block-output CoT-forgery divergence PNGs locally."""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from export_cot_forgery_divergence_images import (
    FONT_25,
    FONT_29,
    WHITE,
    composite_save,
    draw_forest_panel,
    draw_text,
    draw_trajectory_panel,
)


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "projection-viewer"
TRAJECTORY = DATA_DIR / "pair-token-trajectories.jsonl.gz.enc"
METADATA = DATA_DIR / "trajectory-encryption-metadata.json"
LABELS = DATA_DIR / "paired-results.csv"
KEY = Path(
    "/Users/rigel/.codex/gate5-keys/"
    "gptoss20b-cot-forgery-assistant-axis-projections-20260827-2048.key"
)
OUTPUT = ROOT / "reports" / "2026-08-28-cot-forgery-layer18-block-output-images"
COORDINATE = "block_output_layer_18"
AXIS = "z_axis"
BASELINE = "baseline"
ATTACK = "cot_forgery_base_no_qualifier"
SEED = 123


def sha256(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


def decrypt(path: Path, key: Path) -> bytes:
    return subprocess.run(
        ["openssl", "enc", "-d", "-aes-256-cbc", "-pbkdf2", "-iter", "600000",
         "-md", "sha256", "-in", str(path), "-pass", f"file:{key}"],
        check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    ).stdout


def load_records(path: Path, metadata_path: Path, key: Path):
    metadata = json.loads(metadata_path.read_text())
    cipher = path.read_bytes()
    if sha256(cipher) != metadata["ciphertext_sha256"]:
        raise RuntimeError("Trajectory ciphertext checksum mismatch")
    plain = decrypt(path, key)
    if sha256(plain) != metadata["plaintext_sha256"]:
        raise RuntimeError("Trajectory plaintext checksum mismatch")
    with gzip.GzipFile(fileobj=io.BytesIO(plain)) as stream:
        return [json.loads(line) for line in stream if line.strip()]


def smooth(values, window):
    values = np.asarray(values, float)
    radius = window // 2
    return np.array([
        np.nanmean(values[max(0, i - radius): min(len(values), i + radius + 1)])
        for i in range(len(values))
    ])


def final_indices(record):
    return [
        i for i, (role, content, generated) in enumerate(
            zip(record["roles"], record["content"], record["generated"])
        )
        if generated and content and role == "assistant"
    ]


def final_drift(record):
    indices = final_indices(record)
    if len(indices) < 4:
        return np.nan
    parts = np.array_split(np.asarray(indices, int), 4)
    values = np.asarray(record["coordinates"][COORDINATE][AXIS], float)
    return float(values[parts[-1]].mean() - values[parts[0]].mean())


def normalized_final(record, grid):
    indices = final_indices(record)
    values = np.asarray(record["coordinates"][COORDINATE][AXIS], float)[indices]
    return np.interp(grid, np.linspace(0, 1, len(values)), values)


def event_final(record, offsets):
    values = np.asarray(record["coordinates"][COORDINATE][AXIS], float)
    boundary = int(record["boundaries"]["first_final_content"])
    return np.array([
        values[boundary + offset] if 0 <= boundary + offset < len(values) else np.nan
        for offset in offsets
    ])


def bootstrap_curve(values, draws=5000):
    values = np.asarray(values, float)
    rng = np.random.default_rng(SEED)
    means = []
    for start in range(0, draws, 500):
        count = min(500, draws - start)
        sampled = values[rng.integers(0, len(values), (count, len(values)))]
        means.append(np.nanmean(sampled, axis=1))
    samples = np.concatenate(means, axis=0)
    return np.nanmean(values, axis=0), np.nanquantile(samples, .025, axis=0), np.nanquantile(samples, .975, axis=0)


def bootstrap_scalar(values, draws=20000):
    values = np.asarray(values, float)
    rng = np.random.default_rng(SEED)
    sampled = values[rng.integers(0, len(values), (draws, len(values)))].mean(axis=1)
    return float(values.mean()), float(np.quantile(sampled, .025)), float(np.quantile(sampled, .975))


def bootstrap_group_difference(success, unchanged, draws=20000):
    success, unchanged = np.asarray(success, float), np.asarray(unchanged, float)
    rng = np.random.default_rng(SEED)
    first = success[rng.integers(0, len(success), (draws, len(success)))].mean(axis=1)
    second = unchanged[rng.integers(0, len(unchanged), (draws, len(unchanged)))].mean(axis=1)
    samples = first - second
    return float(success.mean() - unchanged.mean()), float(np.quantile(samples, .025)), float(np.quantile(samples, .975))


def aggregate(records, labels_path):
    with labels_path.open(newline="") as stream:
        labels = {row["source_row_id"]: row for row in csv.DictReader(stream)}
    pairs = {}
    for record in records:
        pairs.setdefault(record["source_row_id"], {})[record["condition"]] = record
    grid = np.linspace(0, 1, 31)
    offsets = np.arange(-16, 9)
    grouped = {"Successful": {"effect": [], "normalized": [], "event": []},
               "Unchanged": {"effect": [], "normalized": [], "event": []}}
    for pair_id, pair in pairs.items():
        label = labels[pair_id]
        if label["pair_analyzable"].lower() != "true":
            continue
        group = "Successful" if label["attack_label"] == "HARMFUL_RESPONSE" else "Unchanged"
        baseline, attack = pair[BASELINE], pair[ATTACK]
        grouped[group]["effect"].append(final_drift(attack) - final_drift(baseline))
        normalized = normalized_final(attack, grid) - normalized_final(baseline, grid)
        grouped[group]["normalized"].append(smooth(normalized, 3))
        event = event_final(attack, offsets) - event_final(baseline, offsets)
        grouped[group]["event"].append(smooth(event, 5))
    output_groups = {}
    effects = {}
    for group in ["Successful", "Unchanged"]:
        normalized = bootstrap_curve(grouped[group]["normalized"])
        event = bootstrap_curve(grouped[group]["event"])
        effect = bootstrap_scalar(grouped[group]["effect"])
        output_groups[group] = {
            "n": len(grouped[group]["effect"]),
            "normalized": {"mean": normalized[0].tolist(), "low": normalized[1].tolist(), "high": normalized[2].tolist()},
            "event": {"mean": event[0].tolist(), "low": event[1].tolist(), "high": event[2].tolist()},
        }
        effects[group] = {"mean": effect[0], "low": effect[1], "high": effect[2], "n": len(grouped[group]["effect"])}
    difference = bootstrap_group_difference(grouped["Successful"]["effect"], grouped["Unchanged"]["effect"])
    effects["Difference"] = {"mean": difference[0], "low": difference[1], "high": difference[2], "n": 199}
    return {"grid": grid.tolist(), "offsets": offsets.tolist(),
            "axes": {"z_axis": {"groups": output_groups, "effects": effects}}}


def square_figure(data, kind):
    image = Image.new("RGBA", (1800, 1200), WHITE)
    draw = ImageDraw.Draw(image, "RGBA")
    if kind == "forest":
        draw_text(draw, (900, 38), "gpt-oss-20b CoT-forgery late-final drift (layer 18 block output)", font=FONT_29, anchor="ma")
        draw_forest_panel(draw, data, (80, 175, 1720, 985), "Successful vs unchanged attacks")
    elif kind == "normalized":
        draw_text(draw, (900, 38), "gpt-oss-20b CoT-forgery divergence across final-answer progress", font=FONT_29, anchor="ma")
        draw_trajectory_panel(draw, data, "normalized", (120, 180, 1720, 1000),
                              "Paired attack − baseline projection — layer 18 block output", show_legend=True)
    else:
        draw_text(draw, (900, 38), "gpt-oss-20b CoT-forgery divergence around the analysis-to-final transition", font=FONT_29, anchor="ma")
        draw_trajectory_panel(draw, data, "event", (120, 180, 1720, 1000),
                              "Token 0 is the first final-content token — layer 18 block output", show_legend=True)
    return image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory", type=Path, default=TRAJECTORY)
    parser.add_argument("--metadata", type=Path, default=METADATA)
    parser.add_argument("--labels", type=Path, default=LABELS)
    parser.add_argument("--key", type=Path, default=KEY)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    data = aggregate(load_records(args.trajectory, args.metadata, args.key), args.labels)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    aggregate_path = args.output_dir / "layer18-block-output-aggregate.json"
    aggregate_path.write_text(json.dumps(data, indent=2) + "\n")
    outputs = {
        "layer18-success-effect-forest.png": square_figure(data, "forest"),
        "layer18-final-progress.png": square_figure(data, "normalized"),
        "layer18-analysis-final-transition.png": square_figure(data, "event"),
    }
    for name, image in outputs.items():
        path = args.output_dir / name
        composite_save(image, path)
        print(path)
    print(json.dumps(data["axes"]["z_axis"]["effects"], indent=2))


if __name__ == "__main__":
    main()
