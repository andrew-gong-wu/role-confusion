#!/usr/bin/env python3
"""Project the published gardening conversation under three tag conditions."""

from __future__ import annotations

import json
import os
import pickle
import sys
from pathlib import Path

import cupy
import matplotlib.pyplot as plt
import pandas as pd
import torch
import yaml
from dotenv import load_dotenv
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from demo.simple_test_helpers import (
    ReconstructableTextDataset,
    run_and_export_states,
    stack_collate,
)


load_dotenv(ROOT / ".env")

MODEL_ID = "openai/gpt-oss-20b"
MODEL_REVISION = os.environ.get(
    "ROLE_PROBE_MODEL_REVISION", "6cee5e81ee83917806bbde320786a8fb61efebee"
)
STORAGE_ROOT = Path(os.environ.get("ROLE_PROBE_STORAGE_ROOT", "/workspace/role-probe-storage"))
PROBE_PATH = Path(
    os.environ.get(
        "ROLE_PROBE_PROBE_PATH",
        STORAGE_ROOT / "outputs/baseline-prompt-split/role-probes.pkl",
    )
)
OUTPUT_DIR = Path(
    os.environ.get(
        "ROLE_PROBE_TAG_OUTPUT_DIR", STORAGE_ROOT / "outputs/tag-conditions"
    )
)
PLOT_LAYER = int(os.environ.get("ROLE_PROBE_TAG_LAYER", "12"))
LAYERS = [
    int(value)
    for value in os.environ.get("ROLE_PROBE_TAG_LAYERS", str(PLOT_LAYER)).split(",")
]
ROLE_COLORS = {
    "user": "#3288bd",
    "cot": "#f39c12",
    "assistant": "#36a657",
}


def role_header(role: str) -> str:
    if role in {"system", "user"}:
        return f"<|start|>{role}<|message|>"
    if role == "cot":
        return "<|start|>assistant<|channel|>analysis<|message|>"
    if role == "assistant":
        return "<|start|>assistant<|channel|>final<|message|>"
    raise ValueError(role)


def build_condition(messages: list[dict[str, str]], condition: str) -> tuple[str, list[dict]]:
    parts: list[str] = []
    spans: list[dict] = []
    cursor = 0

    def append(text: str, role: str | None = None) -> None:
        nonlocal cursor
        start = cursor
        parts.append(text)
        cursor += len(text)
        if role is not None:
            spans.append({"start": start, "end": cursor, "original_role": role})

    if condition == "correct_tags":
        for message in messages:
            append(role_header(message["role"]))
            append(message["content"], message["role"])
            append("<|end|>")
    elif condition == "no_tags":
        for index, message in enumerate(messages):
            if index:
                append("\n")
            append(message["content"], message["role"])
    elif condition == "all_user_tags":
        append("<|start|>user<|message|>")
        for index, message in enumerate(messages):
            if index:
                append("\n")
            append(message["content"], message["role"])
        append("<|end|>")
    else:
        raise ValueError(condition)
    return "".join(parts), spans


@torch.no_grad()
def run_gptoss_custom(model, input_ids, attention_mask, return_hidden_states=False):
    pre_mlp_states = []
    handles = []
    if return_hidden_states:
        def hook(_module, _inputs, output):
            pre_mlp_states.append(output.reshape(-1, output.shape[-1]).detach().cpu())

        for layer in model.model.layers:
            handles.append(layer.post_attention_layernorm.register_forward_hook(hook))
    try:
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )
    finally:
        for handle in handles:
            handle.remove()
    return {
        "logits": outputs.logits,
        "all_pre_mlp_hidden_states": pre_mlp_states,
        "all_hidden_states": [],
    }


def role_for_offset(offset: tuple[int, int], spans: list[dict]) -> str | None:
    start, end = offset
    if start == end:
        return None
    midpoint = (start + end) / 2
    for span in spans:
        if span["start"] <= midpoint < span["end"]:
            return span["original_role"]
    return None


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    messages = yaml.safe_load((ROOT / "data/gardening-conversation.yaml").read_text())["messages"]
    conditions = ["correct_tags", "no_tags", "all_user_tags"]
    built = [build_condition(messages, condition) for condition in conditions]
    prompts = [item[0] for item in built]
    spans_by_condition = [item[1] for item in built]

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        cache_dir=os.environ.get("HF_HOME"),
        add_eos_token=False,
        add_bos_token=False,
        padding_side="left",
    )
    offsets = tokenizer(
        prompts,
        add_special_tokens=False,
        padding=True,
        return_offsets_mapping=True,
    )["offset_mapping"]
    max_length = max(len(row) for row in offsets)
    dataset = ReconstructableTextDataset(
        prompts, tokenizer, max_length=max_length, prompt_ix=list(range(len(prompts)))
    )
    loader = DataLoader(dataset, batch_size=3, shuffle=False, collate_fn=stack_collate)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        cache_dir=os.environ.get("HF_HOME"),
        attn_implementation="kernels-community/vllm-flash-attn3",
    ).to("cuda:0").eval()
    model.set_experts_implementation("eager")

    outputs = run_and_export_states(
        model,
        tokenizer,
        run_model_return_states=run_gptoss_custom,
        dl=loader,
        layers_to_keep_acts=LAYERS,
    )
    sample_df = outputs["sample_df"].assign(sample_ix=lambda frame: range(len(frame)))
    sample_df["condition"] = sample_df["prompt_ix"].map(dict(enumerate(conditions)))
    sample_df["original_role"] = [
        role_for_offset(
            tuple(offsets[int(row.prompt_ix)][int(row.token_ix)]),
            spans_by_condition[int(row.prompt_ix)],
        )
        for row in sample_df.itertuples()
    ]

    with PROBE_PATH.open("rb") as handle:
        probes = pickle.load(handle)
    valid = sample_df[sample_df["original_role"].notna()].copy()
    layer_results = []
    for save_index, layer in enumerate(LAYERS):
        probe = next(item for item in probes if int(item["layer_ix"]) == layer)
        hidden = outputs["all_hs"][:, save_index, :]
        features = cupy.asarray(hidden[valid["sample_ix"].tolist(), :].to(torch.float32))
        probabilities = cupy.asnumpy(probe["probe"].predict_proba(features))
        probability_df = pd.DataFrame(probabilities, columns=probe["role_space"])
        layer_result = pd.concat(
            [valid.reset_index(drop=True), probability_df.reset_index(drop=True)], axis=1
        )
        layer_results.append(layer_result.assign(layer_ix=layer))
    result = pd.concat(layer_results, ignore_index=True)
    result["content_token_ix"] = result.groupby(["layer_ix", "condition"]).cumcount()
    result.to_csv(OUTPUT_DIR / "gardening-role-projections.csv", index=False)

    summary = (
        result[result["original_role"].isin(ROLE_COLORS)]
        .groupby(["layer_ix", "condition", "original_role"], as_index=False)[probe["role_space"]]
        .mean()
    )
    summary.to_csv(OUTPUT_DIR / "gardening-role-summary.csv", index=False)

    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharey=True)
    titles = {
        "correct_tags": "Correct architectural tags",
        "no_tags": "All architectural tags removed",
        "all_user_tags": "Entire conversation wrapped in user tags",
    }
    for axis, condition in zip(axes, conditions, strict=True):
        frame = result[
            (result["layer_ix"] == PLOT_LAYER)
            & (result["condition"] == condition)
            & (result["original_role"].isin(ROLE_COLORS))
        ]
        for role, color in ROLE_COLORS.items():
            role_frame = frame[frame["original_role"] == role]
            axis.scatter(
                role_frame["content_token_ix"],
                role_frame["cot"],
                s=8,
                alpha=0.72,
                color=color,
                label=role,
                linewidths=0,
            )
        cot_mean = frame.loc[frame["original_role"] == "cot", "cot"].mean()
        axis.set_title(f"{titles[condition]} — mean CoTness on CoT-style text: {cot_mean:.1%}")
        axis.set_ylabel("CoTness")
        axis.set_ylim(-0.03, 1.03)
        axis.grid(axis="y", alpha=0.2)
    axes[0].legend(title="Original role", ncol=3, loc="lower right")
    axes[-1].set_xlabel("Token position (system prompt omitted)")
    fig.suptitle(f"gpt-oss-20b role perception under conflicting tag conditions (layer {PLOT_LAYER})")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "gardening-cotness-tag-conditions.png", dpi=180)
    plt.close(fig)

    metadata = {
        "model": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "probe_path": str(PROBE_PATH),
        "plot_layer": PLOT_LAYER,
        "evaluated_layers": LAYERS,
        "conditions": conditions,
    }
    (OUTPUT_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(summary.to_string(index=False))
    print(f"Saved results to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
