#!/usr/bin/env python3
"""Train Qwen3 role probes and compare them with Lu et al.'s assistant axis.

This deliberately reuses the neutral passages and classifier hyperparameters
from ``run_exact_role_probe.py``.  The activation hook is the decoder-layer
output (residual stream), matching the hook used to construct the assistant
persona axis.  No text generation is performed.
"""

from __future__ import annotations

import argparse
import gc
import gzip
import json
import os
import time
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import cupy
import cuml
import matplotlib
import numpy as np
import pandas as pd
import torch

matplotlib.use("Agg")

from matplotlib import pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer

SEED = 123
ROLES = ["system", "user", "tool", "cot", "assistant"]
DEFAULT_LAYERS = list(range(0, 64, 4))
MAX_CONTENT_TOKENS = 1024


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("/root/role-confusion-qwen3-32b-assistant-axis-20260825-ready/model"),
    )
    parser.add_argument(
        "--axis",
        type=Path,
        default=Path(
            "/root/role-confusion-qwen3-32b-assistant-axis-20260825-ready/axis/assistant_axis.pt"
        ),
    )
    parser.add_argument(
        "--passages",
        type=Path,
        default=Path(
            "/workspace/role-probe-storage/outputs/exact-full-pipeline-seed123-v3/neutral-passages.jsonl.gz"
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--layers", type=int, nargs="+", default=DEFAULT_LAYERS)
    parser.add_argument("--max-passages", type=int)
    parser.add_argument(
        "--smoke-only",
        action="store_true",
        help="Validate wrappers and one model forward without allocating the full activation tensor.",
    )
    return parser.parse_args()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_passages(path: Path, limit: int | None) -> list[dict[str, object]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle]
    return rows if limit is None else rows[:limit]


def role_wrapper(role: str) -> tuple[str, str]:
    """Return the literal Qwen3 wrapper around neutral content.

    Qwen represents tool responses inside a user message and reasoning inside
    an assistant message.  Those model-native tags are the distinguishing role
    context, while only neutral-content tokens are used to fit the probes.
    """
    wrappers = {
        "system": ("<|im_start|>system\n", "<|im_end|>\n"),
        "user": ("<|im_start|>user\n", "<|im_end|>\n"),
        "tool": (
            "<|im_start|>user\n<tool_response>\n",
            "\n</tool_response><|im_end|>\n",
        ),
        "cot": (
            "<|im_start|>assistant\n<think>\n",
            "\n</think>\n<|im_end|>\n",
        ),
        "assistant": ("<|im_start|>assistant\n", "<|im_end|>\n"),
    }
    return wrappers[role]


def build_prompts(tokenizer, passages: list[dict[str, object]]) -> list[dict[str, object]]:
    prompts: list[dict[str, object]] = []
    prompt_ix = 0
    for passage in passages:
        content_ids = tokenizer(
            str(passage["text"]),
            add_special_tokens=False,
            truncation=True,
            max_length=MAX_CONTENT_TOKENS,
        ).input_ids
        content = tokenizer.decode(content_ids, skip_special_tokens=False)
        for role_ix, role in enumerate(ROLES):
            prefix, suffix = role_wrapper(role)
            rendered = prefix + content + suffix
            encoded = tokenizer(
                rendered,
                add_special_tokens=False,
                return_offsets_mapping=True,
                truncation=False,
            )
            content_start = len(prefix)
            content_end = content_start + len(content)
            content_mask = [
                bool(end > content_start and start < content_end)
                for start, end in encoded.offset_mapping
            ]
            n_content = int(sum(content_mask))
            if not n_content:
                raise RuntimeError(f"No content tokens for prompt {prompt_ix} ({role})")
            prompts.append(
                {
                    "prompt_ix": prompt_ix,
                    "passage_ix": int(passage["passage_ix"]),
                    "source": str(passage["source"]),
                    "role": role,
                    "role_ix": role_ix,
                    "input_ids": encoded.input_ids,
                    "content_mask": content_mask,
                    "n_content": n_content,
                }
            )
            prompt_ix += 1
    return prompts


def validate_wrappers(tokenizer) -> dict[str, str]:
    rendered: dict[str, str] = {}
    for role in ("system", "user", "assistant", "tool"):
        rendered[role] = tokenizer.apply_chat_template(
            [{"role": role, "content": "TOKENTEST"}],
            tokenize=False,
            add_generation_prompt=False,
        )
    for role in ("system", "user", "assistant"):
        prefix, suffix = role_wrapper(role)
        if rendered[role] != prefix + "TOKENTEST" + suffix:
            raise AssertionError(f"Manual {role} wrapper differs from Qwen chat template")
    tool_prefix, tool_suffix = role_wrapper("tool")
    if rendered["tool"] != tool_prefix + "TOKENTEST" + tool_suffix:
        raise AssertionError("Manual tool wrapper differs from Qwen chat template")
    cot_prefix, cot_suffix = role_wrapper("cot")
    rendered["cot"] = cot_prefix + "TOKENTEST" + cot_suffix
    return rendered


def collate(batch: list[dict[str, object]], pad_token_id: int) -> tuple[torch.Tensor, ...]:
    max_length = max(len(row["input_ids"]) for row in batch)
    input_ids = torch.full((len(batch), max_length), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((len(batch), max_length), dtype=torch.long)
    content_mask = torch.zeros((len(batch), max_length), dtype=torch.bool)
    for row_ix, row in enumerate(batch):
        ids = torch.tensor(row["input_ids"], dtype=torch.long)
        mask = torch.tensor(row["content_mask"], dtype=torch.bool)
        input_ids[row_ix, : len(ids)] = ids
        attention_mask[row_ix, : len(ids)] = 1
        content_mask[row_ix, : len(ids)] = mask
    return input_ids, attention_mask, content_mask


def load_model(model_path: Path):
    return AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        device_map={"": 0},
        low_cpu_mem_usage=True,
        local_files_only=True,
        attn_implementation="sdpa",
    ).eval()


@torch.inference_mode()
def smoke_forward(model, tokenizer, prompts: list[dict[str, object]], layers: list[int]) -> dict[str, object]:
    input_ids, attention_mask, content_mask = collate(prompts[:1], tokenizer.pad_token_id)
    captured: dict[int, tuple[int, ...]] = {}
    handles = []

    def hook_for(layer_ix: int):
        def hook(_module, _inputs, output):
            state = output[0] if isinstance(output, (tuple, list)) else output
            selected = state[content_mask.to(state.device)]
            captured[layer_ix] = tuple(selected.shape)
        return hook

    for layer_ix in layers:
        handles.append(model.model.layers[layer_ix].register_forward_hook(hook_for(layer_ix)))
    try:
        model.model(
            input_ids=input_ids.to(model.device),
            attention_mask=attention_mask.to(model.device),
            use_cache=False,
            return_dict=True,
        )
    finally:
        for handle in handles:
            handle.remove()
    torch.cuda.synchronize()
    return {
        "captured_shapes": {str(key): list(value) for key, value in captured.items()},
        "gpu_peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
    }


@torch.inference_mode()
def extract_activations(
    model,
    tokenizer,
    prompts: list[dict[str, object]],
    layers: list[int],
    batch_size: int,
) -> tuple[torch.Tensor, dict[str, np.ndarray]]:
    total_tokens = sum(int(row["n_content"]) for row in prompts)
    hidden_size = int(model.config.hidden_size)
    print(
        f"Allocating {total_tokens:,} x {len(layers)} x {hidden_size} float16 CPU tensor "
        f"({total_tokens * len(layers) * hidden_size * 2 / 2**30:.2f} GiB)",
        flush=True,
    )
    activations = torch.empty((total_tokens, len(layers), hidden_size), dtype=torch.float16)
    labels = np.empty(total_tokens, dtype=np.int16)
    prompt_ids = np.empty(total_tokens, dtype=np.int32)
    token_positions = np.empty(total_tokens, dtype=np.int16)
    passage_ids = np.empty(total_tokens, dtype=np.int16)

    current_mask: torch.Tensor | None = None
    write_start = 0
    write_end = 0
    handles = []

    def hook_for(save_ix: int):
        def hook(_module, _inputs, output):
            if current_mask is None:
                raise RuntimeError("Missing current content mask")
            state = output[0] if isinstance(output, (tuple, list)) else output
            selected = state[current_mask]
            activations[write_start:write_end, save_ix, :].copy_(
                selected.to(device="cpu", dtype=torch.float16)
            )
        return hook

    for save_ix, layer_ix in enumerate(layers):
        handles.append(model.model.layers[layer_ix].register_forward_hook(hook_for(save_ix)))

    started = time.time()
    try:
        for batch_start in range(0, len(prompts), batch_size):
            batch = prompts[batch_start : batch_start + batch_size]
            input_ids, attention_mask, cpu_content_mask = collate(batch, tokenizer.pad_token_id)
            counts = [int(row["n_content"]) for row in batch]
            write_end = write_start + sum(counts)
            current_mask = cpu_content_mask.to(model.device)

            cursor = write_start
            for row, count in zip(batch, counts, strict=True):
                labels[cursor : cursor + count] = int(row["role_ix"])
                prompt_ids[cursor : cursor + count] = int(row["prompt_ix"])
                token_positions[cursor : cursor + count] = np.arange(count, dtype=np.int16)
                passage_ids[cursor : cursor + count] = int(row["passage_ix"])
                cursor += count

            output = model.model(
                input_ids=input_ids.to(model.device),
                attention_mask=attention_mask.to(model.device),
                use_cache=False,
                return_dict=True,
            )
            del output, input_ids, attention_mask, cpu_content_mask, current_mask
            current_mask = None
            write_start = write_end
            batch_number = batch_start // batch_size + 1
            if batch_number % 25 == 0 or write_start == total_tokens:
                elapsed = time.time() - started
                print(
                    f"extraction {min(batch_start + batch_size, len(prompts))}/{len(prompts)} prompts; "
                    f"{write_start:,}/{total_tokens:,} tokens; {elapsed / 60:.1f} min",
                    flush=True,
                )
    finally:
        for handle in handles:
            handle.remove()

    if write_start != total_tokens:
        raise AssertionError(f"Wrote {write_start} of {total_tokens} token rows")
    metadata = {
        "labels": labels,
        "prompt_ids": prompt_ids,
        "token_positions": token_positions,
        "passage_ids": passage_ids,
    }
    return activations, metadata


def to_numpy(value) -> np.ndarray:
    return value.get() if hasattr(value, "get") else np.asarray(value)


def train_probes(
    activations: torch.Tensor,
    metadata: dict[str, np.ndarray],
    layers: list[int],
    output_dir: Path,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame]:
    unique_prompt_ids = np.unique(metadata["prompt_ids"])
    train_prompt_ids, test_prompt_ids = cuml.train_test_split(
        unique_prompt_ids, test_size=0.1, random_state=SEED
    )
    train_prompt_ids = to_numpy(train_prompt_ids).astype(np.int32)
    test_prompt_ids = to_numpy(test_prompt_ids).astype(np.int32)
    train_mask = np.isin(metadata["prompt_ids"], train_prompt_ids)
    test_mask = np.isin(metadata["prompt_ids"], test_prompt_ids)
    train_ix = np.flatnonzero(train_mask)
    test_ix = np.flatnonzero(test_mask)
    pd.DataFrame(
        {
            "prompt_ix": np.concatenate([train_prompt_ids, test_prompt_ids]),
            "split": ["train"] * len(train_prompt_ids) + ["test"] * len(test_prompt_ids),
        }
    ).to_csv(output_dir / "prompt-split.csv", index=False)

    y_train = cupy.asarray(metadata["labels"][train_ix])
    y_test = cupy.asarray(metadata["labels"][test_ix])
    all_coefs = np.empty((len(layers), len(ROLES), activations.shape[-1]), dtype=np.float32)
    all_intercepts = np.empty((len(layers), len(ROLES)), dtype=np.float32)
    overall_rows: list[dict[str, object]] = []
    role_rows: list[dict[str, object]] = []

    for save_ix, layer_ix in enumerate(layers):
        started = time.time()
        print(f"Training probe at layer {layer_ix}", flush=True)
        x_train_cpu = activations[train_ix, save_ix, :].float().numpy()
        x_test_cpu = activations[test_ix, save_ix, :].float().numpy()
        x_train = cupy.asarray(x_train_cpu)
        x_test = cupy.asarray(x_test_cpu)
        classifier = cuml.linear_model.LogisticRegression(
            penalty="l2",
            max_iter=5_000,
            linesearch_max_iter=100,
            fit_intercept=True,
            C=5.0e-3,
        )
        classifier.fit(x_train, y_train)
        predictions = to_numpy(classifier.predict(x_test)).astype(np.int16)
        truth = to_numpy(y_test).astype(np.int16)
        correct = predictions == truth
        accuracy = float(correct.mean())
        all_coefs[save_ix] = to_numpy(classifier.coef_).astype(np.float32)
        all_intercepts[save_ix] = to_numpy(classifier.intercept_).astype(np.float32)
        overall_rows.append(
            {"layer_ix": layer_ix, "accuracy": accuracy, "n_test_tokens": int(len(truth))}
        )
        for role_ix, role in enumerate(ROLES):
            role_mask = truth == role_ix
            role_rows.append(
                {
                    "layer_ix": layer_ix,
                    "role": role,
                    "accuracy": float(correct[role_mask].mean()),
                    "n_test_tokens": int(role_mask.sum()),
                }
            )
        print(
            f"layer {layer_ix}: accuracy={accuracy:.6f}; training={(time.time() - started) / 60:.1f} min",
            flush=True,
        )
        del classifier, predictions, truth, correct, x_train, x_test, x_train_cpu, x_test_cpu
        cupy.get_default_memory_pool().free_all_blocks()
        gc.collect()

    return all_coefs, all_intercepts, pd.DataFrame(overall_rows), pd.DataFrame(role_rows)


def cosine_matrix(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise RuntimeError("Cannot compute cosine similarity for a zero vector")
    normalized = vectors / norms
    return normalized @ normalized.T


def save_heatmap(matrix: np.ndarray, labels: list[str], path: Path, layer: int) -> None:
    figure, axis = plt.subplots(figsize=(8.8, 7.6), constrained_layout=True)
    image = axis.imshow(matrix, cmap="RdBu_r", vmin=-1, vmax=1)
    axis.set_xticks(range(len(labels)), labels=labels, rotation=35, ha="right")
    axis.set_yticks(range(len(labels)), labels=labels)
    axis.set_title(f"Qwen3-32B vector cosine similarities (layer {layer})")
    for row_ix in range(len(labels)):
        for column_ix in range(len(labels)):
            value = matrix[row_ix, column_ix]
            axis.text(
                column_ix,
                row_ix,
                f"{value:.3f}",
                ha="center",
                va="center",
                color="white" if abs(value) > 0.58 else "black",
                fontsize=9,
            )
    figure.colorbar(image, ax=axis, label="Cosine similarity", shrink=0.82)
    figure.savefig(path, dpi=200)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to alter existing path: {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    layers = sorted(set(args.layers))
    if any(layer < 0 or layer >= 64 for layer in layers):
        raise ValueError(f"Invalid Qwen3-32B layer list: {layers}")

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    wrappers = validate_wrappers(tokenizer)
    passages = load_passages(args.passages, args.max_passages)
    prompts = build_prompts(tokenizer, passages)
    prompt_summary = {
        "n_passages": len(passages),
        "n_prompts": len(prompts),
        "n_content_tokens": sum(int(row["n_content"]) for row in prompts),
        "roles": ROLES,
        "layers": layers,
        "wrappers": wrappers,
    }
    write_json(args.output_dir / "prompt-summary.json", prompt_summary)
    print(json.dumps(prompt_summary, indent=2), flush=True)

    torch.manual_seed(SEED)
    torch.cuda.reset_peak_memory_stats()
    model = load_model(args.model)
    if len(model.model.layers) != 64 or int(model.config.hidden_size) != 5120:
        raise AssertionError("Expected Qwen3-32B to have 64 layers and hidden size 5120")
    smoke = smoke_forward(model, tokenizer, prompts, layers)
    write_json(args.output_dir / "smoke-validation.json", smoke)
    print(json.dumps(smoke, indent=2), flush=True)
    if args.smoke_only:
        return

    activations, token_metadata = extract_activations(
        model, tokenizer, prompts, layers, args.batch_size
    )
    extraction_peak_gib = torch.cuda.max_memory_allocated() / 2**30
    del model
    gc.collect()
    torch.cuda.empty_cache()

    coefs, intercepts, overall, per_role = train_probes(
        activations, token_metadata, layers, args.output_dir
    )
    overall.to_csv(args.output_dir / "overall-accuracy.csv", index=False)
    per_role.to_csv(args.output_dir / "per-role-accuracy.csv", index=False)
    best = overall.sort_values(["accuracy", "layer_ix"], ascending=[False, True]).iloc[0]
    selected_layer = int(best.layer_ix)
    selected_save_ix = layers.index(selected_layer)
    selection = {
        "criterion": "maximum held-out neutral-text token accuracy; lower layer breaks ties",
        "selected_layer": selected_layer,
        "selected_accuracy": float(best.accuracy),
    }
    write_json(args.output_dir / "layer-selection.json", selection)

    axis = torch.load(args.axis, map_location="cpu", weights_only=True)
    if tuple(axis.shape) != (64, 5120):
        raise AssertionError(f"Unexpected assistant-axis shape: {tuple(axis.shape)}")
    persona_axis = axis[selected_layer].float().numpy()
    raw_role_weights = coefs[selected_save_ix]
    centered_role_directions = raw_role_weights - raw_role_weights.mean(axis=0, keepdims=True)
    labels = ["persona_assistant_axis"] + [f"role_{role}" for role in ROLES]
    vectors = np.concatenate([persona_axis[None, :], centered_role_directions], axis=0)
    similarities = cosine_matrix(vectors)

    np.savez_compressed(
        args.output_dir / "all-layer-role-probe-vectors.npz",
        layers=np.asarray(layers, dtype=np.int16),
        roles=np.asarray(ROLES),
        coefficients=coefs,
        intercepts=intercepts,
    )
    np.savez_compressed(
        args.output_dir / "centralized-vectors.npz",
        selected_layer=np.asarray(selected_layer, dtype=np.int16),
        labels=np.asarray(labels),
        vectors=vectors,
        persona_assistant_axis=persona_axis,
        role_probe_weights_raw=raw_role_weights,
        role_probe_directions_centered=centered_role_directions,
    )
    pd.DataFrame(similarities, index=labels, columns=labels).to_csv(
        args.output_dir / "cosine-similarity.csv"
    )
    save_heatmap(similarities, labels, args.output_dir / "cosine-similarity-heatmap.png", selected_layer)
    write_json(
        args.output_dir / "run-metadata.json",
        {
            "model": str(args.model),
            "assistant_axis": str(args.axis),
            "neutral_passages": str(args.passages),
            "seed": SEED,
            "activation_site": "decoder layer output (residual stream after the full block)",
            "classifier": {
                "type": "cuML multinomial logistic regression",
                "penalty": "l2",
                "C": 5.0e-3,
                "max_iter": 5000,
                "test_size_by_prompt": 0.1,
            },
            "role_vector_centering": "subtract mean coefficient vector across the five softmax classes",
            "extraction_gpu_peak_gib": extraction_peak_gib,
            **selection,
        },
    )
    del activations
    print(f"Completed. Outputs: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
