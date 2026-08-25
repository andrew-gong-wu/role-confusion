#!/usr/bin/env python3
"""Fit context-valid hierarchical Qwen3 role probes.

The primary selection score averages three balanced held-out accuracies:

* outer message role: system vs user vs assistant;
* user subtype: plain user text vs nested tool response;
* assistant subtype: nested thinking vs final answer.

A secondary five-way leaf probe supplies comparable system/user/tool/cot/
assistant coefficient vectors.  Activations are standardized for numerical
optimization, then coefficients are mapped back to the original residual-stream
coordinates before cosine comparison with Lu et al.'s assistant persona axis.
No generation is performed.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import time
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
LAYERS = list(range(0, 64, 4))
LEAF_ROLES = ["system", "user", "tool", "cot", "assistant"]
CONDITIONS = [
    "outer_system",
    "outer_user",
    "outer_assistant",
    "nested_user",
    "nested_tool",
    "nested_cot",
    "nested_assistant",
]
CONDITION_TO_INDEX = {name: index for index, name in enumerate(CONDITIONS)}
TOKENS_PER_CONDITION = 128
MAX_CONTENT_TOKENS = 1024

TOOL_SYSTEM = """<|im_start|>system
# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"type":"function","function":{"name":"lookup","description":"Retrieve a reference passage.","parameters":{"type":"object","properties":{"query":{"type":"string"}},"required":["query"]}}}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags.<|im_end|>
<|im_start|>user
Retrieve the reference passage for this request.<|im_end|>
"""

PLAIN_USER_PREFIX = TOOL_SYSTEM + """<|im_start|>assistant
I can address the next message directly without calling a tool.<|im_end|>
<|im_start|>user
"""

TOOL_PREFIX = TOOL_SYSTEM + """<|im_start|>assistant
<think>
I should retrieve the reference passage.
</think>

<tool_call>
{"name":"lookup","arguments":{"query":"reference passage"}}
</tool_call><|im_end|>
<|im_start|>user
<tool_response>
"""

ASSISTANT_CONTEXT = """<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
Read the supplied material and acknowledge it.<|im_end|>
"""


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
    parser.add_argument("--tokens-per-condition", type=int, default=TOKENS_PER_CONDITION)
    parser.add_argument("--max-passages", type=int)
    parser.add_argument("--smoke-only", action="store_true")
    return parser.parse_args()


def render_condition(condition: str, content: str) -> tuple[str, int, int]:
    if condition == "outer_system":
        prefix, suffix = "<|im_start|>system\n", "<|im_end|>\n"
    elif condition == "outer_user":
        prefix, suffix = "<|im_start|>user\n", "<|im_end|>\n"
    elif condition == "outer_assistant":
        prefix, suffix = "<|im_start|>assistant\n", "<|im_end|>\n"
    elif condition == "nested_user":
        prefix, suffix = PLAIN_USER_PREFIX, "<|im_end|>\n"
    elif condition == "nested_tool":
        prefix, suffix = TOOL_PREFIX, "\n</tool_response><|im_end|>\n"
    elif condition == "nested_cot":
        prefix = ASSISTANT_CONTEXT + "<|im_start|>assistant\n<think>\n"
        suffix = "\n</think>\n\nAcknowledged.<|im_end|>\n"
    elif condition == "nested_assistant":
        prefix = ASSISTANT_CONTEXT + "<|im_start|>assistant\n<think>\n\n</think>\n\n"
        suffix = "<|im_end|>\n"
    else:
        raise KeyError(condition)
    rendered = prefix + content + suffix
    return rendered, len(prefix), len(prefix) + len(content)


def content_candidate(tokenizer, condition: str, content: str) -> dict[str, object]:
    rendered, content_start, content_end = render_condition(condition, content)
    encoded = tokenizer(
        rendered,
        add_special_tokens=False,
        return_offsets_mapping=True,
        truncation=False,
    )
    content_token_indices = [
        index
        for index, (start, end) in enumerate(encoded.offset_mapping)
        if end > content_start and start < content_end
    ]
    if not content_token_indices:
        raise RuntimeError(f"No target tokens for {condition}")
    return {
        "condition": condition,
        "condition_ix": CONDITION_TO_INDEX[condition],
        "input_ids": encoded.input_ids,
        "content_token_indices": content_token_indices,
        "rendered_tokens": len(encoded.input_ids),
    }


def evenly_spaced_indices(size: int, count: int) -> np.ndarray:
    if count >= size:
        return np.arange(size, dtype=np.int64)
    if count == 1:
        return np.asarray([0], dtype=np.int64)
    selected = np.arange(count, dtype=np.int64) * (size - 1) // (count - 1)
    if len(np.unique(selected)) != count:
        raise AssertionError("Token-position sampler produced duplicates")
    return selected


def build_prompts(
    tokenizer,
    passages: list[dict[str, object]],
    tokens_per_condition: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    prompts: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    prompt_ix = 0
    for passage in passages:
        content_ids = tokenizer(
            str(passage["text"]),
            add_special_tokens=False,
            truncation=True,
            max_length=MAX_CONTENT_TOKENS,
        ).input_ids
        content = tokenizer.decode(content_ids, skip_special_tokens=False)
        candidates = [content_candidate(tokenizer, condition, content) for condition in CONDITIONS]
        paired_count = min(
            tokens_per_condition,
            min(len(row["content_token_indices"]) for row in candidates),
        )
        for candidate in candidates:
            all_target_indices = candidate.pop("content_token_indices")
            relative = evenly_spaced_indices(len(all_target_indices), paired_count)
            selected_token_indices = [all_target_indices[index] for index in relative]
            content_mask = [False] * len(candidate["input_ids"])
            for token_ix in selected_token_indices:
                content_mask[token_ix] = True
            prompts.append(
                {
                    "prompt_ix": prompt_ix,
                    "passage_ix": int(passage["passage_ix"]),
                    "source": str(passage["source"]),
                    "role_ix": int(candidate["condition_ix"]),
                    "input_ids": candidate["input_ids"],
                    "content_mask": content_mask,
                    "n_content": paired_count,
                }
            )
            summary_rows.append(
                {
                    "prompt_ix": prompt_ix,
                    "passage_ix": int(passage["passage_ix"]),
                    "condition": candidate["condition"],
                    "sampled_tokens": paired_count,
                    "rendered_tokens": int(candidate["rendered_tokens"]),
                }
            )
            prompt_ix += 1
    return prompts, summary_rows


def group_split(passage_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    unique = np.unique(passage_ids)
    rng = np.random.default_rng(SEED)
    shuffled = rng.permutation(unique)
    n_test = max(1, int(np.ceil(0.1 * len(shuffled))))
    return np.sort(shuffled[n_test:]), np.sort(shuffled[:n_test])


def balanced_accuracy(truth: np.ndarray, predictions: np.ndarray, n_classes: int) -> float:
    recalls = []
    for class_ix in range(n_classes):
        mask = truth == class_ix
        recalls.append(float((predictions[mask] == class_ix).mean()))
    return float(np.mean(recalls))


def class_centered_raw_weights(
    classifier,
    feature_mean: cupy.ndarray,
    feature_scale: cupy.ndarray,
    n_classes: int,
) -> tuple[np.ndarray, np.ndarray]:
    coef_standardized = base.to_numpy(classifier.coef_).astype(np.float32)
    intercept_standardized = base.to_numpy(classifier.intercept_).astype(np.float32)
    scale = base.to_numpy(feature_scale).astype(np.float32)
    mean = base.to_numpy(feature_mean).astype(np.float32)
    if n_classes == 2:
        positive_weight = coef_standardized.reshape(-1) / scale
        positive_intercept = float(intercept_standardized.reshape(-1)[0]) - float(
            np.dot(positive_weight, mean)
        )
        weights = np.stack([-0.5 * positive_weight, 0.5 * positive_weight])
        intercepts = np.asarray([-0.5 * positive_intercept, 0.5 * positive_intercept])
    else:
        weights = coef_standardized / scale[None, :]
        intercepts = intercept_standardized - weights @ mean
        weights -= weights.mean(axis=0, keepdims=True)
        intercepts -= intercepts.mean()
    return weights.astype(np.float32), intercepts.astype(np.float32)


def fit_probe(
    layer_features: np.ndarray,
    condition_ids: np.ndarray,
    passage_ids: np.ndarray,
    train_passages: np.ndarray,
    test_passages: np.ndarray,
    condition_to_class: dict[int, int],
    class_names: list[str],
) -> tuple[dict[str, object], np.ndarray, np.ndarray, list[dict[str, object]]]:
    allowed = np.asarray(sorted(condition_to_class), dtype=np.int16)
    included = np.isin(condition_ids, allowed)
    train_mask = included & np.isin(passage_ids, train_passages)
    test_mask = included & np.isin(passage_ids, test_passages)
    train_ix = np.flatnonzero(train_mask)
    test_ix = np.flatnonzero(test_mask)
    y_train_np = np.asarray(
        [condition_to_class[int(value)] for value in condition_ids[train_ix]], dtype=np.int32
    )
    y_test_np = np.asarray(
        [condition_to_class[int(value)] for value in condition_ids[test_ix]], dtype=np.int32
    )

    x_train = cupy.asarray(layer_features[train_ix], dtype=cupy.float32)
    x_test = cupy.asarray(layer_features[test_ix], dtype=cupy.float32)
    feature_mean = x_train.mean(axis=0)
    feature_scale = x_train.std(axis=0)
    feature_scale = cupy.maximum(feature_scale, cupy.float32(1.0e-6))
    x_train -= feature_mean
    x_train /= feature_scale
    x_test -= feature_mean
    x_test /= feature_scale
    y_train = cupy.asarray(y_train_np)

    classifier = cuml.linear_model.LogisticRegression(
        penalty="l2",
        max_iter=5_000,
        linesearch_max_iter=100,
        fit_intercept=True,
        C=5.0e-3,
        tol=1.0e-6,
    )
    classifier.fit(x_train, y_train)
    predictions = base.to_numpy(classifier.predict(x_test)).astype(np.int32)
    n_classes = len(class_names)
    weights, intercepts = class_centered_raw_weights(
        classifier, feature_mean, feature_scale, n_classes
    )
    accuracy = float((predictions == y_test_np).mean())
    macro_accuracy = balanced_accuracy(y_test_np, predictions, n_classes)
    per_class = []
    for class_ix, class_name in enumerate(class_names):
        mask = y_test_np == class_ix
        per_class.append(
            {
                "class": class_name,
                "accuracy": float((predictions[mask] == class_ix).mean()),
                "n_test_tokens": int(mask.sum()),
            }
        )
    metrics = {
        "accuracy": accuracy,
        "balanced_accuracy": macro_accuracy,
        "n_train_tokens": int(len(train_ix)),
        "n_test_tokens": int(len(test_ix)),
    }
    del classifier, x_train, x_test, y_train, predictions
    cupy.get_default_memory_pool().free_all_blocks()
    gc.collect()
    return metrics, weights, intercepts, per_class


HEADS = {
    "outer_role": {
        "conditions": {
            CONDITION_TO_INDEX["outer_system"]: 0,
            CONDITION_TO_INDEX["outer_user"]: 1,
            CONDITION_TO_INDEX["outer_assistant"]: 2,
        },
        "classes": ["system", "user", "assistant"],
    },
    "user_subtype": {
        "conditions": {
            CONDITION_TO_INDEX["nested_user"]: 0,
            CONDITION_TO_INDEX["nested_tool"]: 1,
        },
        "classes": ["user", "tool"],
    },
    "assistant_subtype": {
        "conditions": {
            CONDITION_TO_INDEX["nested_assistant"]: 0,
            CONDITION_TO_INDEX["nested_cot"]: 1,
        },
        "classes": ["assistant", "cot"],
    },
    "leaf_five_way": {
        "conditions": {
            CONDITION_TO_INDEX["outer_system"]: 0,
            CONDITION_TO_INDEX["nested_user"]: 1,
            CONDITION_TO_INDEX["nested_tool"]: 2,
            CONDITION_TO_INDEX["nested_cot"]: 3,
            CONDITION_TO_INDEX["nested_assistant"]: 4,
        },
        "classes": LEAF_ROLES,
    },
}


def train_all_layers(
    activations: torch.Tensor,
    metadata: dict[str, np.ndarray],
    layers: list[int],
    output_dir: Path,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], pd.DataFrame, pd.DataFrame]:
    train_passages, test_passages = group_split(metadata["passage_ids"])
    pd.DataFrame(
        {
            "passage_ix": np.concatenate([train_passages, test_passages]),
            "split": ["train"] * len(train_passages) + ["test"] * len(test_passages),
        }
    ).to_csv(output_dir / "passage-split.csv", index=False)

    weights = {
        head: np.empty(
            (len(layers), len(spec["classes"]), activations.shape[-1]), dtype=np.float32
        )
        for head, spec in HEADS.items()
    }
    intercepts = {
        head: np.empty((len(layers), len(spec["classes"])), dtype=np.float32)
        for head, spec in HEADS.items()
    }
    metric_rows: list[dict[str, object]] = []
    per_class_rows: list[dict[str, object]] = []
    for save_ix, layer_ix in enumerate(layers):
        started = time.time()
        print(f"Training hierarchical probes at layer {layer_ix}", flush=True)
        layer_features = activations[:, save_ix, :].float().numpy()
        for head, spec in HEADS.items():
            metrics, head_weights, head_intercepts, per_class = fit_probe(
                layer_features,
                metadata["labels"],
                metadata["passage_ids"],
                train_passages,
                test_passages,
                spec["conditions"],
                spec["classes"],
            )
            weights[head][save_ix] = head_weights
            intercepts[head][save_ix] = head_intercepts
            metric_rows.append({"layer_ix": layer_ix, "head": head, **metrics})
            per_class_rows.extend(
                {"layer_ix": layer_ix, "head": head, **row} for row in per_class
            )
            print(
                f"  {head}: balanced_accuracy={metrics['balanced_accuracy']:.6f}",
                flush=True,
            )
        del layer_features
        gc.collect()
        print(f"layer {layer_ix} finished in {(time.time() - started) / 60:.1f} min", flush=True)
    return weights, intercepts, pd.DataFrame(metric_rows), pd.DataFrame(per_class_rows)


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to alter existing path: {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    layers = sorted(set(args.layers))
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    passages = base.load_passages(args.passages, args.max_passages)
    prompts, prompt_rows = build_prompts(tokenizer, passages, args.tokens_per_condition)
    pd.DataFrame(prompt_rows).to_csv(args.output_dir / "prompt-conditions.csv", index=False)
    summary = {
        "n_passages": len(passages),
        "n_conditions": len(CONDITIONS),
        "n_prompts": len(prompts),
        "n_sampled_tokens": sum(int(row["n_content"]) for row in prompts),
        "conditions": CONDITIONS,
        "tokens_per_condition_ceiling": args.tokens_per_condition,
        "layers": layers,
    }
    base.write_json(args.output_dir / "prompt-summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)

    torch.manual_seed(SEED)
    torch.cuda.reset_peak_memory_stats()
    model = base.load_model(args.model)
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

    weights, intercepts, metrics, per_class = train_all_layers(
        activations, metadata, layers, args.output_dir
    )
    metrics.to_csv(args.output_dir / "probe-accuracy.csv", index=False)
    per_class.to_csv(args.output_dir / "per-class-accuracy.csv", index=False)
    primary_heads = ["outer_role", "user_subtype", "assistant_subtype"]
    selection_frame = (
        metrics[metrics["head"].isin(primary_heads)]
        .groupby("layer_ix", as_index=False)
        .balanced_accuracy.mean()
        .rename(columns={"balanced_accuracy": "hierarchical_mean_balanced_accuracy"})
    )
    flat_scores = metrics[metrics["head"] == "leaf_five_way"][
        ["layer_ix", "balanced_accuracy"]
    ].rename(columns={"balanced_accuracy": "leaf_five_way_balanced_accuracy"})
    selection_frame = selection_frame.merge(flat_scores, on="layer_ix")
    selection_frame.to_csv(args.output_dir / "layer-selection-scores.csv", index=False)
    best = selection_frame.sort_values(
        ["hierarchical_mean_balanced_accuracy", "layer_ix"], ascending=[False, True]
    ).iloc[0]
    selected_layer = int(best.layer_ix)
    selected_save_ix = layers.index(selected_layer)
    selection = {
        "criterion": "maximum mean balanced accuracy across the three hierarchical heads; lower layer breaks ties",
        "selected_layer": selected_layer,
        "selected_hierarchical_mean_balanced_accuracy": float(
            best.hierarchical_mean_balanced_accuracy
        ),
        "selected_leaf_five_way_balanced_accuracy": float(
            best.leaf_five_way_balanced_accuracy
        ),
    }
    base.write_json(args.output_dir / "layer-selection.json", selection)

    vector_payload: dict[str, np.ndarray] = {
        "layers": np.asarray(layers, dtype=np.int16),
    }
    for head, spec in HEADS.items():
        vector_payload[f"{head}_classes"] = np.asarray(spec["classes"])
        vector_payload[f"{head}_weights"] = weights[head]
        vector_payload[f"{head}_intercepts"] = intercepts[head]
    np.savez_compressed(args.output_dir / "all-hierarchical-probe-vectors.npz", **vector_payload)

    assistant_axis = torch.load(args.axis, map_location="cpu", weights_only=True)
    if tuple(assistant_axis.shape) != (64, 5120):
        raise AssertionError(f"Unexpected assistant axis shape: {tuple(assistant_axis.shape)}")
    persona_vector = assistant_axis[selected_layer].float().numpy()
    leaf_vectors = weights["leaf_five_way"][selected_save_ix]
    labels = ["persona_assistant_axis"] + [f"role_{role}" for role in LEAF_ROLES]
    centralized = np.concatenate([persona_vector[None, :], leaf_vectors], axis=0)
    similarities = base.cosine_matrix(centralized)
    np.savez_compressed(
        args.output_dir / "centralized-hierarchical-vectors.npz",
        selected_layer=np.asarray(selected_layer, dtype=np.int16),
        labels=np.asarray(labels),
        vectors=centralized,
        persona_assistant_axis=persona_vector,
        leaf_role_directions=leaf_vectors,
        outer_role_directions=weights["outer_role"][selected_save_ix],
        user_subtype_directions=weights["user_subtype"][selected_save_ix],
        assistant_subtype_directions=weights["assistant_subtype"][selected_save_ix],
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
    base.write_json(
        args.output_dir / "run-metadata.json",
        {
            "model": str(args.model),
            "assistant_axis": str(args.axis),
            "neutral_passages": str(args.passages),
            "seed": SEED,
            "activation_site": "decoder layer output (residual stream after the full block)",
            "split": "10% held out by passage; every condition for a passage remains in one split",
            "sampling": "equal count of evenly spaced target-content tokens for all seven conditions per passage",
            "optimization": "per-head z-scoring on training activations; coefficients mapped back to raw activation coordinates",
            "classifier": {
                "type": "cuML logistic regression",
                "penalty": "l2",
                "C": 5.0e-3,
                "max_iter": 5000,
                "tol": 1.0e-6,
            },
            "leaf_vector_centering": "subtract mean raw-coordinate coefficient vector across the five multinomial classes",
            "extraction_gpu_peak_gib": extraction_peak_gib,
            **selection,
        },
    )
    del activations
    print(f"Completed. Outputs: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
