#!/usr/bin/env python3
"""Train paper-style tag-induced role probes for Qwen3-32B.

This follows Appendix G of Ye, Cui, and Hadfield-Menell (2026): each neutral
base sequence is rendered under each architectural role while target content is
held fixed.  Qwen's nested reasoning format is controlled with variable-length
neutral filler inside the thought preceding final-answer content, and matching
filler is prepended before the role tags for every other role.  Filler and tag
states are discarded; only target-content states train the probe.

By default the script fits the paper's five roles.  ``--roles`` can select a
paper-supported subset (for example, user/assistant/tool) and fits a fresh
multinomial probe in that role space.  Probes are trained at decoder-block
outputs so their coefficients occupy the same 5,120-dimensional coordinates
as the downloaded Qwen3-32B assistant-persona axis.  No text generation is
performed.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import time
from collections import Counter
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import cupy
import cuml
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer

import run_qwen_role_probe_cosines as base


SEED = 123
ALL_ROLES = ["system", "user", "tool", "cot", "assistant"]
LAYERS = list(range(0, 64, 4))
MAX_TARGET_TOKENS = 1024
EXPECTED_BASE_SEQUENCES = 250
EXPECTED_SOURCES = {"c4": 62, "dolma3": 188}
FILLER_MAX_TOKENS = MAX_TARGET_TOKENS // 2
PAPER_C_GRID = [1.0e-4, 1.0e-3, 1.0e-2, 1.0e-1, 1.0, 10.0, 100.0, 1000.0]


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
    parser.add_argument("--layers", type=int, nargs="+", default=LAYERS)
    parser.add_argument(
        "--roles",
        nargs="+",
        choices=ALL_ROLES,
        default=ALL_ROLES,
        help="Role subset and class order for a freshly fitted multinomial probe.",
    )
    parser.add_argument(
        "--skip-first-n",
        type=int,
        default=0,
        help="Discard this many initial content tokens per rendered sequence.",
    )
    parser.add_argument(
        "--fixed-c",
        type=float,
        help="Use a fixed logistic-regression C instead of running the grid search.",
    )
    parser.add_argument("--smoke-only", action="store_true")
    return parser.parse_args()


def truncate_text(tokenizer, text: str, max_tokens: int) -> str:
    token_ids = tokenizer(
        text,
        add_special_tokens=False,
        truncation=True,
        max_length=max_tokens,
    ).input_ids
    return tokenizer.decode(token_ids, skip_special_tokens=False)


def paper_fillers(tokenizer, passages: list[dict[str, object]]) -> list[str]:
    """Reproduce the paper code's variable-length, deranged neutral fillers."""
    rng = np.random.RandomState(SEED)
    lengths = (
        rng.beta(0.5, 4.0, size=len(passages)) * (FILLER_MAX_TOKENS + 1)
    ).astype(int)
    permutation = rng.permutation(len(passages))
    while np.any(permutation == np.arange(len(passages))):
        permutation = rng.permutation(len(passages))
    fillers = []
    for base_ix, partner_ix in enumerate(permutation):
        filler = truncate_text(
            tokenizer,
            str(passages[int(partner_ix)]["text"]),
            max(1, int(lengths[base_ix])),
        )
        fillers.append(filler.strip() if lengths[base_ix] else "")
    return fillers


def render_role(role: str, target: str, filler: str) -> tuple[str, int, int]:
    """Render Qwen roles using the nested-tag and positional controls in Appendix G."""
    positional_prefix = filler
    if role == "system":
        prefix = positional_prefix + "<|im_start|>system\n"
        suffix = "<|im_end|>\n"
    elif role == "user":
        prefix = positional_prefix + "<|im_start|>user\n"
        suffix = "<|im_end|>\n"
    elif role == "tool":
        prefix = positional_prefix + "<|im_start|>user\n<tool_response>\n"
        suffix = "\n</tool_response><|im_end|>\n"
    elif role == "cot":
        prefix = positional_prefix + "<|im_start|>assistant\n<think>\n"
        suffix = "\n</think>\n\n<|im_end|>\n"
    elif role == "assistant":
        prefix = (
            "<|im_start|>assistant\n<think>\n"
            + filler
            + "\n</think>\n\n"
        )
        suffix = "<|im_end|>\n"
    else:
        raise KeyError(role)
    rendered = prefix + target + suffix
    return rendered, len(prefix), len(prefix) + len(target)


def build_prompts(
    tokenizer,
    passages: list[dict[str, object]],
    roles: list[str],
    skip_first_n: int,
) -> tuple[list[dict[str, object]], pd.DataFrame]:
    if len(passages) != EXPECTED_BASE_SEQUENCES:
        raise AssertionError(
            f"Expected {EXPECTED_BASE_SEQUENCES} base sequences, found {len(passages)}"
        )
    sources = Counter(str(row["source"]) for row in passages)
    if dict(sources) != EXPECTED_SOURCES:
        raise AssertionError(f"Expected source mixture {EXPECTED_SOURCES}, found {dict(sources)}")

    targets = [
        truncate_text(tokenizer, str(row["text"]), MAX_TARGET_TOKENS)
        for row in passages
    ]
    fillers = paper_fillers(tokenizer, passages)
    prompts: list[dict[str, object]] = []
    manifest_rows: list[dict[str, object]] = []
    prompt_ix = 0
    for base_ix, (passage, target, filler) in enumerate(
        zip(passages, targets, fillers, strict=True)
    ):
        target_digest = hashlib.sha256(target.encode("utf-8")).hexdigest()
        filler_token_count = len(
            tokenizer(filler, add_special_tokens=False).input_ids
        ) if filler else 0
        role_targets: dict[str, str] = {}
        manifest_start = len(manifest_rows)
        role_to_index = {role: index for index, role in enumerate(roles)}
        for role in roles:
            rendered, target_start, target_end = render_role(role, target, filler)
            encoded = tokenizer(
                rendered,
                add_special_tokens=False,
                return_offsets_mapping=True,
                truncation=False,
            )
            target_token_indices = [
                index
                for index, (start, end) in enumerate(encoded.offset_mapping)
                if end > target_start and start < target_end
            ]
            if not target_token_indices:
                raise RuntimeError(f"No target tokens for base {base_ix}, role {role}")
            reconstructed = tokenizer.decode(
                [encoded.input_ids[index] for index in target_token_indices],
                skip_special_tokens=False,
            )
            role_targets[role] = reconstructed
            retained_token_indices = target_token_indices[skip_first_n:]
            if not retained_token_indices:
                # The authors apply the position cutoff after token labeling, so
                # short rendered prompts simply contribute no rows to the probe.
                prompt_ix += 1
                continue
            content_mask = [False] * len(encoded.input_ids)
            for token_ix in retained_token_indices:
                content_mask[token_ix] = True
            prompts.append(
                {
                    "prompt_ix": prompt_ix,
                    "passage_ix": int(passage["passage_ix"]),
                    "source": str(passage["source"]),
                    "role": role,
                    "role_ix": role_to_index[role],
                    "input_ids": encoded.input_ids,
                    "content_mask": content_mask,
                    "n_content": len(retained_token_indices),
                }
            )
            manifest_rows.append(
                {
                    "prompt_ix": prompt_ix,
                    "base_sequence_ix": base_ix,
                    "passage_ix": int(passage["passage_ix"]),
                    "source": str(passage["source"]),
                    "role": role,
                    "target_sha256": target_digest,
                    "target_tokens_total": len(target_token_indices),
                    "target_tokens": len(retained_token_indices),
                    "skipped_initial_content_tokens": skip_first_n,
                    "target_start_token_ix": min(retained_token_indices),
                    "rendered_tokens": len(encoded.input_ids),
                    "filler_tokens": filler_token_count,
                }
            )
            prompt_ix += 1
        if len({row for row in role_targets.values()}) != 1:
            # Context-sensitive boundary tokenization may alter the decoded edge token.
            # The original target string and digest remain exactly paired across roles;
            # record this explicitly rather than silently claiming token identity.
            for row in manifest_rows[manifest_start:]:
                row["decoded_target_tokens_match"] = False
        else:
            for row in manifest_rows[manifest_start:]:
                row["decoded_target_tokens_match"] = True
    return prompts, pd.DataFrame(manifest_rows)


def prompt_split(prompt_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Match the published notebook's seeded 90/10 split over rendered prompts."""
    train_ids, heldout_ids = cuml.train_test_split(
        np.unique(prompt_ids), test_size=0.1, random_state=SEED
    )
    return (
        base.to_numpy(train_ids).astype(np.int32),
        base.to_numpy(heldout_ids).astype(np.int32),
    )


def fit_one(
    x_train: cupy.ndarray,
    y_train: cupy.ndarray,
    x_heldout: cupy.ndarray,
    y_heldout: cupy.ndarray,
    c_value: float,
) -> tuple[object, float, float, np.ndarray]:
    classifier = cuml.linear_model.LogisticRegression(
        penalty="l2",
        max_iter=5_000,
        linesearch_max_iter=100,
        fit_intercept=True,
        C=c_value,
    )
    classifier.fit(x_train, y_train)
    predictions = base.to_numpy(classifier.predict(x_heldout)).astype(np.int16)
    probabilities = classifier.predict_proba(x_heldout)
    truth = base.to_numpy(y_heldout).astype(np.int16)
    accuracy = float((predictions == truth).mean())
    nll = float(cuml.metrics.log_loss(y_heldout, probabilities))
    return classifier, accuracy, nll, predictions


def train_probes(
    activations: torch.Tensor,
    metadata: dict[str, np.ndarray],
    layers: list[int],
    roles: list[str],
    output_dir: Path,
    fixed_c: float | None,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame, float]:
    train_prompt_ids, heldout_prompt_ids = prompt_split(metadata["prompt_ids"])
    train_ix = np.flatnonzero(np.isin(metadata["prompt_ids"], train_prompt_ids))
    heldout_ix = np.flatnonzero(np.isin(metadata["prompt_ids"], heldout_prompt_ids))
    split_frame = pd.DataFrame(
        {
            "prompt_ix": np.concatenate([train_prompt_ids, heldout_prompt_ids]),
            "split": ["train"] * len(train_prompt_ids)
            + ["heldout"] * len(heldout_prompt_ids),
        }
    )
    split_frame.to_csv(output_dir / "prompt-split.csv", index=False)
    # cuML's classifier accepts int16 labels, but its log_loss implementation
    # requires int32/int64.  Use one dtype consistently for both operations.
    y_train = cupy.asarray(metadata["labels"][train_ix], dtype=cupy.int32)
    y_heldout = cupy.asarray(metadata["labels"][heldout_ix], dtype=cupy.int32)

    if fixed_c is not None:
        if fixed_c <= 0:
            raise ValueError("--fixed-c must be positive")
        selected_c = float(fixed_c)
        pd.DataFrame([{"C": selected_c, "selection": "fixed"}]).to_csv(
            output_dir / "regularization-grid.csv", index=False
        )
        print(f"Using fixed C={selected_c:g}", flush=True)
    else:
        tuning_layer = layers[(len(layers) - 1) // 2]
        tuning_save_ix = layers.index(tuning_layer)
        x_train = cupy.asarray(activations[train_ix, tuning_save_ix, :].float().numpy())
        x_heldout = cupy.asarray(activations[heldout_ix, tuning_save_ix, :].float().numpy())
        tuning_rows = []
        for c_value in PAPER_C_GRID:
            started = time.time()
            classifier, accuracy, nll, _ = fit_one(
                x_train, y_train, x_heldout, y_heldout, c_value
            )
            tuning_rows.append({"layer_ix": tuning_layer, "C": c_value, "accuracy": accuracy, "nll": nll})
            print(
                f"C grid layer {tuning_layer}: C={c_value:g}, accuracy={accuracy:.6f}, "
                f"nll={nll:.6f}, minutes={(time.time() - started) / 60:.1f}",
                flush=True,
            )
            del classifier
            cupy.get_default_memory_pool().free_all_blocks()
            gc.collect()
        del x_train, x_heldout
        cupy.get_default_memory_pool().free_all_blocks()
        tuning = pd.DataFrame(tuning_rows)
        tuning.to_csv(output_dir / "regularization-grid.csv", index=False)
        selected_c = float(tuning.sort_values(["nll", "C"], ascending=[True, True]).iloc[0].C)
        print(f"Selected C={selected_c:g} by minimum held-out NLL at layer {tuning_layer}", flush=True)

    coefficients = np.empty((len(layers), len(roles), activations.shape[-1]), dtype=np.float32)
    intercepts = np.empty((len(layers), len(roles)), dtype=np.float32)
    metric_rows = []
    per_class_rows = []
    heldout_truth = base.to_numpy(y_heldout).astype(np.int16)
    for save_ix, layer_ix in enumerate(layers):
        started = time.time()
        x_train = cupy.asarray(activations[train_ix, save_ix, :].float().numpy())
        x_heldout = cupy.asarray(activations[heldout_ix, save_ix, :].float().numpy())
        classifier, accuracy, nll, predictions = fit_one(
            x_train, y_train, x_heldout, y_heldout, selected_c
        )
        raw_coef = base.to_numpy(classifier.coef_).astype(np.float32)
        raw_intercept = base.to_numpy(classifier.intercept_).astype(np.float32)
        coefficients[save_ix] = raw_coef - raw_coef.mean(axis=0, keepdims=True)
        intercepts[save_ix] = raw_intercept - raw_intercept.mean()
        recalls = []
        for role_ix, role in enumerate(roles):
            mask = heldout_truth == role_ix
            recall = float((predictions[mask] == role_ix).mean())
            recalls.append(recall)
            per_class_rows.append(
                {
                    "layer_ix": layer_ix,
                    "role": role,
                    "accuracy": recall,
                    "n_heldout_tokens": int(mask.sum()),
                }
            )
        metric_rows.append(
            {
                "layer_ix": layer_ix,
                "accuracy": accuracy,
                "balanced_accuracy": float(np.mean(recalls)),
                "nll": nll,
                "n_train_tokens": int(len(train_ix)),
                "n_heldout_tokens": int(len(heldout_ix)),
            }
        )
        print(
            f"layer {layer_ix}: accuracy={accuracy:.6f}, balanced={np.mean(recalls):.6f}, "
            f"nll={nll:.6f}, minutes={(time.time() - started) / 60:.1f}",
            flush=True,
        )
        del classifier, predictions, x_train, x_heldout
        cupy.get_default_memory_pool().free_all_blocks()
        gc.collect()
    return (
        coefficients,
        intercepts,
        pd.DataFrame(metric_rows),
        pd.DataFrame(per_class_rows),
        selected_c,
    )


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to alter existing path: {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    layers = sorted(set(args.layers))
    roles = list(dict.fromkeys(args.roles))
    if len(roles) < 2:
        raise ValueError("A multinomial role probe requires at least two distinct roles")
    if args.skip_first_n < 0:
        raise ValueError("--skip-first-n cannot be negative")
    if any(layer < 0 or layer >= 64 for layer in layers):
        raise ValueError(f"Invalid Qwen3-32B layer list: {layers}")

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    passages = base.load_passages(args.passages, None)
    prompts, prompt_manifest = build_prompts(
        tokenizer, passages, roles, args.skip_first_n
    )
    prompt_manifest.to_csv(args.output_dir / "prompt-manifest.csv", index=False)
    prompt_summary = {
        "paper": "Ye, Cui, and Hadfield-Menell (2026), arXiv:2603.12277v6, Appendix G",
        "seed": SEED,
        "base_sequences": len(passages),
        "rendered_sequences": len(prompts),
        "source_counts": dict(Counter(str(row["source"]) for row in passages)),
        "roles": roles,
        "max_target_tokens": MAX_TARGET_TOKENS,
        "filler_max_tokens": FILLER_MAX_TOKENS,
        "target_content_tokens": sum(int(row["n_content"]) for row in prompts),
        "layers": layers,
        "nested_reasoning_control": "variable neutral filler in assistant thought; matching filler before tags for other roles",
        "training_tokens": "target content only; tags and filler excluded",
        "skip_first_n_content_tokens": args.skip_first_n,
    }
    base.write_json(args.output_dir / "prompt-summary.json", prompt_summary)
    print(json.dumps(prompt_summary, indent=2), flush=True)

    torch.manual_seed(SEED)
    torch.cuda.reset_peak_memory_stats()
    model = base.load_model(args.model)
    if len(model.model.layers) != 64 or int(model.config.hidden_size) != 5120:
        raise AssertionError("Expected Qwen3-32B to have 64 layers and hidden size 5120")
    smoke = base.smoke_forward(model, tokenizer, prompts, layers)
    base.write_json(args.output_dir / "smoke-validation.json", smoke)
    print(json.dumps(smoke, indent=2), flush=True)
    if args.smoke_only:
        return

    activations, metadata = base.extract_activations(
        model, tokenizer, prompts, layers, args.batch_size
    )
    extraction_peak_gib = torch.cuda.max_memory_allocated() / 2**30
    del model
    gc.collect()
    torch.cuda.empty_cache()

    coefficients, intercepts, metrics, per_class, selected_c = train_probes(
        activations, metadata, layers, roles, args.output_dir, args.fixed_c
    )
    metrics.to_csv(args.output_dir / "probe-accuracy.csv", index=False)
    per_class.to_csv(args.output_dir / "per-class-accuracy.csv", index=False)
    best = metrics.sort_values(["accuracy", "layer_ix"], ascending=[False, True]).iloc[0]
    selected_layer = int(best.layer_ix)
    selected_save_ix = layers.index(selected_layer)

    assistant_axis = torch.load(args.axis, map_location="cpu", weights_only=True)
    if tuple(assistant_axis.shape) != (64, 5120):
        raise AssertionError(f"Unexpected assistant-axis shape: {tuple(assistant_axis.shape)}")
    persona_vector = assistant_axis[selected_layer].float().numpy()
    role_vectors = coefficients[selected_save_ix]
    labels = ["persona_assistant_axis"] + [f"role_{role}" for role in roles]
    vectors = np.concatenate([persona_vector[None, :], role_vectors], axis=0)
    similarities = base.cosine_matrix(vectors)

    np.savez_compressed(
        args.output_dir / "all-layer-paper-role-probe-vectors.npz",
        layers=np.asarray(layers, dtype=np.int16),
        roles=np.asarray(roles),
        coefficients_centered=coefficients,
        intercepts_centered=intercepts,
    )
    np.savez_compressed(
        args.output_dir / "centralized-paper-role-vectors.npz",
        selected_layer=np.asarray(selected_layer, dtype=np.int16),
        labels=np.asarray(labels),
        vectors=vectors,
        persona_assistant_axis=persona_vector,
        role_probe_directions=role_vectors,
    )
    pd.DataFrame(similarities, index=labels, columns=labels).to_csv(
        args.output_dir / "cosine-similarity.csv"
    )
    base.save_heatmap(
        similarities,
        labels,
        args.output_dir / "cosine-similarity-heatmap.png",
        selected_layer,
    )
    selection = {
        "criterion": "maximum held-out token accuracy; lower layer breaks ties",
        "selected_layer": selected_layer,
        "selected_accuracy": float(best.accuracy),
        "selected_balanced_accuracy": float(best.balanced_accuracy),
        "selected_nll": float(best.nll),
        "selected_C": selected_c,
    }
    base.write_json(args.output_dir / "layer-selection.json", selection)
    base.write_json(
        args.output_dir / "run-metadata.json",
        {
            **prompt_summary,
            **selection,
            "model": str(args.model),
            "assistant_axis": str(args.axis),
            "neutral_passages": str(args.passages),
            "activation_site": "decoder layer output (residual stream after the full block)",
            "classifier": f"cuML {len(roles)}-way multinomial logistic regression, L2, no feature scaling",
            "regularization_grid": PAPER_C_GRID if args.fixed_c is None else None,
            "regularization_selection": (
                "minimum held-out NLL at the middle probed layer"
                if args.fixed_c is None
                else "fixed by command line"
            ),
            "split": "seeded 90/10 rendered-prompt split matching the published notebook",
            "role_vector_centering": f"subtract mean coefficient across {len(roles)} multinomial classes",
            "extraction_gpu_peak_gib": extraction_peak_gib,
        },
    )
    del activations
    print(f"Completed. Outputs: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
