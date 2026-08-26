#!/usr/bin/env python3
"""Restore the shutdown-safe Qwen3-32B bundle from persistent storage.

The August 25 persistent download contains complete shards 1-12 and prefixes
of shards 13-16.  A separate completion bundle contains the exact missing
tails plus all of shard 17.  This script reconstructs a new, loadable model
directory without modifying either persistent source directory.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
from pathlib import Path


DEFAULT_BASE = Path("/workspace/role-confusion-qwen3-32b-assistant-axis-20260825/model")
DEFAULT_COMPLETION = Path("/workspace/qwen3-32b-completion-20260825")

# Exact byte sizes and SHA-256 digests of the live, load-tested model shards.
SHARDS: dict[str, tuple[int, str]] = {
    "model-00001-of-00017.safetensors": (3957109648, "52562b2ff97b61764260273e71bf5b4cf8a66f569399398f26dec0300fcf1316"),
    "model-00002-of-00017.safetensors": (3900791760, "e26764b2c6878e3fb7198895fa833ec62838d84a19665e6abfbae43c6daf02b3"),
    "model-00003-of-00017.safetensors": (3900791760, "6c5ba7bed9c52bc121e75cbe8a7be46936d0006cc80f42a6d5886ed40b4c2a62"),
    "model-00004-of-00017.safetensors": (3900791800, "f736f6ac4d8c30866107fb1185a05b3c3cfce9717720082f466fa44e691bcec8"),
    "model-00005-of-00017.safetensors": (3900791800, "a52ed375c083209c54d42ac510afeb1fbb5af4f193be2dc7d103f665a0f212d3"),
    "model-00006-of-00017.safetensors": (3900791800, "37fae28990b0e4a70228549d040c0393e87bee3820d59e58e47844974d8dff5b"),
    "model-00007-of-00017.safetensors": (3900791800, "37776006aeaba29eca8bc73b2b963fe3477e1c2e3f6a27cb9527be75b905e1bf"),
    "model-00008-of-00017.safetensors": (3900791800, "73e74e9129674fe330948005075d70ab4fa0b92b68fb220c5a693f9cea553730"),
    "model-00009-of-00017.safetensors": (3900791800, "a044b3602a01bd8ea62ff51badf9cc038ab1d73d97399480e6a55b4c86fa7fa6"),
    "model-00010-of-00017.safetensors": (3900791800, "9966612ba7ecfc2cd2e592fb95224b86d743271ff88e172cc272a4b26382aa75"),
    "model-00011-of-00017.safetensors": (3900791800, "e2a058a0ac7d4b992b731c29221ecfb4b76b8a48d9004d0e8a62ba44f699845c"),
    "model-00012-of-00017.safetensors": (3900791800, "58a1aa89093fea07325f787072a468e3482a470ff4b7fe5ead5f749683907c40"),
    "model-00013-of-00017.safetensors": (3900791800, "35f3381bab31a23370c37d922290aeecdf603418336058fb86fe42d8f51ac40c"),
    "model-00014-of-00017.safetensors": (3900791800, "8713b062ddc178acf5917610b7f4b64eede833b2ea4aa37bd562dcf2f3a3339d"),
    "model-00015-of-00017.safetensors": (3900791800, "bec439d23931821a236d8f62fa79deecf5551bd25602278aa2ae0ce432b378cf"),
    "model-00016-of-00017.safetensors": (3900791800, "e569139fadd61fe7c8f9eb1c976d9a627cae48c57ddf228cfbd0593c59c64ff7"),
    "model-00017-of-00017.safetensors": (3055341992, "1f47c318fcd7797c0f85b4233cb754438b10e795b8bc874889090c416a94bd38"),
}

PARTIAL_SIZES = {
    "model-00013-of-00017.safetensors": 3690987520,
    "model-00014-of-00017.safetensors": 3690987520,
    "model-00015-of-00017.safetensors": 3556769792,
    "model-00016-of-00017.safetensors": 3489660928,
    "model-00017-of-00017.safetensors": 0,
}

TAIL_HASHES = {
    "model-00013-of-00017.safetensors": "f75e2accf98f3a85f3c899d78a5abacd6697ba5d546c2c1fdf8238a40dfa976e",
    "model-00014-of-00017.safetensors": "c528e8a99d71e9f3c00d17f5cc5eaf608a10aff0808fc8b0039a263cc09186ca",
    "model-00015-of-00017.safetensors": "52cd68e5933b4a360ecffaaabab7d20521d31119775ad78f62020b28797974c4",
    "model-00016-of-00017.safetensors": "0ac7721feb9e96fabe9b6a869e5b674b592d0e8cfce25229986e16d2694d9dee",
    "model-00017-of-00017.safetensors": "1f47c318fcd7797c0f85b4233cb754438b10e795b8bc874889090c416a94bd38",
}

AXIS_SIZE = 656986
AXIS_SHA256 = "a207fe7a36563280b7b29010880aa0082bd8e3113c141cb4a2eed6b46c140211"
CHUNK_SIZE = 16 * 1024 * 1024


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate(path: Path, expected_size: int, expected_hash: str) -> None:
    if path.stat().st_size != expected_size:
        raise RuntimeError(
            f"Unexpected size for {path}: {path.stat().st_size} != {expected_size}"
        )
    actual_hash = sha256(path)
    if actual_hash != expected_hash:
        raise RuntimeError(f"SHA-256 mismatch for {path}: {actual_hash}")


def append_file(source: Path, destination_handle) -> None:
    with source.open("rb") as source_handle:
        shutil.copyfileobj(source_handle, destination_handle, length=CHUNK_SIZE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--completion", type=Path, default=DEFAULT_COMPLETION)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to alter existing path: {args.output_dir}")
    model_output = args.output_dir / "model"
    axis_output = args.output_dir / "axis"
    model_output.mkdir(parents=True)
    axis_output.mkdir()

    for source in args.base_model.iterdir():
        if source.is_file() and source.name not in SHARDS:
            shutil.copy2(source, model_output / source.name)

    for shard_name, (expected_size, expected_hash) in SHARDS.items():
        source = args.base_model / shard_name
        destination = model_output / shard_name
        if shard_name not in PARTIAL_SIZES:
            validate(source, expected_size, expected_hash)
            shutil.copy2(source, destination)
        else:
            partial_size = PARTIAL_SIZES[shard_name]
            if source.stat().st_size != partial_size:
                raise RuntimeError(
                    f"Unexpected persistent prefix size for {source}: "
                    f"{source.stat().st_size} != {partial_size}"
                )
            tail = args.completion / "model-tails" / f"{shard_name}.tail"
            validate(tail, expected_size - partial_size, TAIL_HASHES[shard_name])
            with destination.open("xb") as destination_handle:
                append_file(source, destination_handle)
                append_file(tail, destination_handle)
        validate(destination, expected_size, expected_hash)
        print(f"verified {shard_name}", flush=True)

    axis_source = args.completion / "axis" / "assistant_axis.pt"
    validate(axis_source, AXIS_SIZE, AXIS_SHA256)
    shutil.copy2(axis_source, axis_output / "assistant_axis.pt")
    validate(axis_output / "assistant_axis.pt", AXIS_SIZE, AXIS_SHA256)
    print(f"restored model and axis under {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
