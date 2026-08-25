#!/usr/bin/env python3
"""Run the pinned full gpt-oss-20b role-probe reproduction (seed 123).

This follows experiments/role-analysis/02-train-role-probes.ipynb at upstream
commit ec333c40fd43fe991e1ebf66765051b6d7e35784, restricted to the requested
four-way system/user/CoT/assistant probe and augmented with complete held-out
diagnostics plus the three tomato tag conditions at every probed layer.
"""

from __future__ import annotations

import gc
import gzip
import hashlib
import json
import os
import pickle
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import cupy
import cuml
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.pipeline
import torch
import yaml
from datasets import load_dataset
from dotenv import load_dotenv
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.masking_utils import (
    create_causal_mask as current_create_causal_mask,
    create_sliding_window_causal_mask as current_create_sliding_window_causal_mask,
)


ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")
sys.path.insert(0, str(ROOT))

from demo.simple_test_helpers import (  # noqa: E402
    ReconstructableTextDataset,
    label_gptoss_content_roles,
    run_and_export_states,
    stack_collate,
)


UPSTREAM_COMMIT = "ec333c40fd43fe991e1ebf66765051b6d7e35784"
MODEL_ID = "openai/gpt-oss-20b"
MODEL_REVISION = os.environ.get(
    "ROLE_PROBE_MODEL_REVISION", "6cee5e81ee83917806bbde320786a8fb61efebee"
)
C4_REVISION = os.environ.get(
    "ROLE_PROBE_C4_REVISION", "f3b95a11ff318ce8b651afc7eb8e7bd2af469c10"
)
DOLMA_REVISION = "3a8349c"
SEED = 123
N_PASSAGES = 250
C4_COUNT = int(N_PASSAGES * 0.25)
DOLMA_COUNT = N_PASSAGES - C4_COUNT
MAX_CONTENT_TOKENS = 1024
BATCH_SIZE = int(os.environ.get("ROLE_PROBE_EXACT_BATCH_SIZE", "16"))
LAYERS = list(range(0, 24, 2))
ROLE_SPACE = ["system", "user", "cot", "assistant"]
ALL_RENDERED_ROLES = ["system", "user", "tool", "cot", "assistant"]
POSITION_EDGES = [0, 16, 32, 64, 128, 256, 512, 1024, np.inf]
POSITION_LABELS = ["0-15", "16-31", "32-63", "64-127", "128-255", "256-511", "512-1023", "1024+"]

STORAGE_ROOT = Path(os.environ.get("ROLE_PROBE_STORAGE_ROOT", "/workspace/role-probe-storage"))
OUTPUT_DIR = Path(
    os.environ.get(
        "ROLE_PROBE_EXACT_OUTPUT_DIR",
        STORAGE_ROOT / "outputs/exact-full-pipeline-seed123",
    )
)
UPSTREAM_ROOT = Path(
    os.environ.get(
        "ROLE_PROBE_UPSTREAM_ROOT",
        STORAGE_ROOT / f"upstream/{UPSTREAM_COMMIT}",
    )
)


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def mark_phase(phase: str, **extra: object) -> None:
    payload = {
        "phase": phase,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        **extra,
    }
    write_json(OUTPUT_DIR / "run-status.json", payload)
    print(f"PHASE: {phase}", flush=True)


def require_pinned_upstream() -> None:
    required = [
        "experiments/role-analysis/config/probe.yaml",
        "experiments/role-analysis/config/tomato.yaml",
        "utils/chat_templates/gptoss.j2",
        "utils/role_templates.py",
        "utils/substring_assignments.py",
        "utils/pretrained_models/gptoss.py",
    ]
    missing = [name for name in required if not (UPSTREAM_ROOT / name).is_file()]
    if missing:
        raise RuntimeError(f"Pinned upstream tree is incomplete at {UPSTREAM_ROOT}: {missing}")
    actual = subprocess.check_output(
        ["git", "-C", str(UPSTREAM_ROOT), "rev-parse", "HEAD"], text=True
    ).strip()
    if actual != UPSTREAM_COMMIT:
        raise RuntimeError(f"Expected upstream {UPSTREAM_COMMIT}, found {actual}")
    sys.path.insert(0, str(UPSTREAM_ROOT))


def record_upstream_provenance() -> None:
    paths = [
        "experiments/role-analysis/02-train-role-probes.ipynb",
        "experiments/role-analysis/config/probe.yaml",
        "experiments/role-analysis/config/tomato.yaml",
        "utils/chat_templates/gptoss.j2",
        "utils/role_templates.py",
        "utils/pretrained_models/gptoss.py",
    ]
    records = []
    for relative in paths:
        data = (UPSTREAM_ROOT / relative).read_bytes()
        records.append(
            {"path": relative, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
        )
    probe_config = yaml.safe_load(
        (UPSTREAM_ROOT / "experiments/role-analysis/config/probe.yaml").read_text()
    )["gptoss-20b"]
    write_json(
        OUTPUT_DIR / "upstream-provenance.json",
        {
            "commit": UPSTREAM_COMMIT,
            "files": records,
            "upstream_probe_config": probe_config,
        },
    )


def load_raw_data() -> list[dict[str, str]]:
    def take(dataset, count: int, source: str) -> list[dict[str, str]]:
        iterator = iter(dataset)
        rows = []
        for _ in range(count):
            sample = next(iterator, None)
            if sample is None:
                raise RuntimeError(f"{source} ended after {len(rows)} of {count} rows")
            rows.append({"text": sample["text"], "source": source})
        return rows

    c4 = load_dataset(
        "allenai/c4",
        data_dir="en",
        split="validation",
        revision=C4_REVISION,
        streaming=True,
    ).shuffle(seed=SEED, buffer_size=50_000)
    dolma = load_dataset(
        "allenai/dolma3_mix-150B-1025",
        split="train",
        revision=DOLMA_REVISION,
        streaming=True,
    ).shuffle(seed=SEED, buffer_size=50_000)
    return take(c4, C4_COUNT, "c4") + take(dolma, DOLMA_COUNT, "dolma3")


def build_training_prompts(tokenizer, raw_data, render_single_message):
    texts = tokenizer.batch_decode(
        tokenizer(
            [row["text"] for row in raw_data],
            add_special_tokens=False,
            padding=False,
            truncation=True,
            max_length=MAX_CONTENT_TOKENS,
        ).input_ids
    )
    rows = []
    for question_ix, (text, source) in enumerate(zip(texts, [x["source"] for x in raw_data], strict=True)):
        for role in ALL_RENDERED_ROLES:
            rows.append(
                {
                    "question_ix": question_ix,
                    "source": source,
                    "role": role,
                    "question": text,
                    "prompt": render_single_message("gptoss-20b", role, text),
                }
            )
    frame = pd.DataFrame(rows).assign(prompt_ix=lambda df: range(len(df)))
    if len(frame) != N_PASSAGES * len(ALL_RENDERED_ROLES):
        raise AssertionError("Unexpected rendered prompt count")
    return frame


@torch.no_grad()
def hook_forward(model, input_ids, attention_mask, return_hidden_states=False):
    states = []
    handles = []
    if return_hidden_states:
        def hook(_module, _inputs, output):
            states.append(output.reshape(-1, output.shape[-1]).detach().cpu())

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
        "all_pre_mlp_hidden_states": states,
        "all_hidden_states": [],
    }


def validate_pre_mlp_capture(model, tokenizer, upstream_forward) -> None:
    inputs = tokenizer(
        ["Hi! I am a dog and I like to bark", "Vegetables are good for"],
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=64,
    ).to(model.device)
    with torch.no_grad():
        hooked = hook_forward(model, **inputs, return_hidden_states=True)
        upstream = upstream_forward(model, **inputs, return_hidden_states=True)
    if not torch.equal(hooked["logits"], upstream["logits"]):
        raise AssertionError("Hooked and pinned-upstream logits differ")
    for layer in LAYERS:
        if not torch.equal(
            hooked["all_pre_mlp_hidden_states"][layer],
            upstream["all_pre_mlp_hidden_states"][layer],
        ):
            raise AssertionError(f"Pre-MLP capture differs at layer {layer}")
    write_json(
        OUTPUT_DIR / "pre-mlp-validation.json",
        {"logits_exact": True, "layers_exact": LAYERS, "capture": "post_attention_layernorm output"},
    )


def patch_pinned_masking_api(upstream_module) -> None:
    """Adapt the pinned May-2026 masking call to Transformers 5.15.

    The pinned forward passes ``input_embeds`` and ``cache_position``. Current
    Transformers spells the former ``inputs_embeds`` and accepts position IDs
    instead of cache positions. No masking semantics are changed.
    """
    def adapter(function):
        def call(*, config, input_embeds, attention_mask, cache_position, past_key_values):
            return function(
                config=config,
                inputs_embeds=input_embeds,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                position_ids=cache_position.unsqueeze(0),
            )
        return call

    upstream_module.create_causal_mask = adapter(current_create_causal_mask)
    upstream_module.create_sliding_window_causal_mask = adapter(
        current_create_sliding_window_causal_mask
    )


def patch_pinned_model_api(model) -> None:
    """Restore the per-layer attribute used by the pinned custom forward."""
    layer_types = model.config.layer_types
    if len(layer_types) != len(model.model.layers):
        raise AssertionError("Unexpected GPT-OSS layer-type count")
    for layer, attention_type in zip(model.model.layers, layer_types, strict=True):
        layer.attention_type = attention_type


def cupy_to_numpy(value):
    if hasattr(value, "get"):
        return value.get()
    return np.asarray(value)


def train_probes(probe_df: pd.DataFrame, activations: torch.Tensor):
    roles_map = {role: index for index, role in enumerate(ROLE_SPACE)}
    prompt_ids = probe_df["prompt_ix"].unique()
    train_ids, test_ids = cuml.train_test_split(
        prompt_ids, test_size=0.1, random_state=SEED
    )
    train_ids = cupy_to_numpy(train_ids).astype(np.int64)
    test_ids = cupy_to_numpy(test_ids).astype(np.int64)
    train_df = probe_df[probe_df["prompt_ix"].isin(train_ids)].copy()
    test_df = probe_df[probe_df["prompt_ix"].isin(test_ids)].copy()
    if set(train_ids) & set(test_ids):
        raise AssertionError("Train/test prompt overlap")

    split_frame = pd.DataFrame(
        {"prompt_ix": np.concatenate([train_ids, test_ids]),
         "split": ["train"] * len(train_ids) + ["test"] * len(test_ids)}
    )
    split_frame.to_csv(OUTPUT_DIR / "prompt-split.csv", index=False)

    y_train = cupy.asarray([roles_map[x] for x in train_df["role"]])
    y_test = cupy.asarray([roles_map[x] for x in test_df["role"]])
    train_sample_ix = train_df["sample_ix"].to_numpy()
    test_sample_ix = test_df["sample_ix"].to_numpy()

    probes = []
    overall_rows = []
    role_rows = []
    confusion_rows = []
    position_rows = []
    exact_position_rows = []
    prediction_dir = OUTPUT_DIR / "heldout-predictions"
    prediction_dir.mkdir(exist_ok=True)

    for save_ix, layer in enumerate(LAYERS):
        print(f"Training layer {layer}", flush=True)
        x_train = cupy.asarray(
            activations[train_sample_ix, save_ix, :].to(torch.float32)
        )
        x_test = cupy.asarray(
            activations[test_sample_ix, save_ix, :].to(torch.float32)
        )
        pipeline = sklearn.pipeline.Pipeline(
            [("clf", cuml.linear_model.LogisticRegression(
                penalty="l2", max_iter=5_000, linesearch_max_iter=100,
                fit_intercept=True, C=5.0e-3,
            ))]
        )
        pipeline.fit(x_train, y_train)
        pred = cupy_to_numpy(pipeline.predict(x_test)).astype(np.int16)
        probs = cupy_to_numpy(pipeline.predict_proba(x_test)).astype(np.float32)
        clf = pipeline.named_steps["clf"]
        logits = cupy_to_numpy(clf.decision_function(x_test)).astype(np.float32)
        truth = cupy_to_numpy(y_test).astype(np.int16)
        correct = pred == truth
        accuracy = float(correct.mean())
        overall_rows.append({"layer_ix": layer, "accuracy": accuracy, "n_test_tokens": len(truth)})

        for role_ix, role in enumerate(ROLE_SPACE):
            mask = truth == role_ix
            role_rows.append(
                {"layer_ix": layer, "role": role, "accuracy": float(correct[mask].mean()), "n": int(mask.sum())}
            )
            for pred_ix, pred_role in enumerate(ROLE_SPACE):
                count = int(np.sum(mask & (pred == pred_ix)))
                confusion_rows.append(
                    {"layer_ix": layer, "true_role": role, "predicted_role": pred_role, "count": count,
                     "row_fraction": count / int(mask.sum())}
                )

        test_positions = test_df["token_in_seg_ix"].to_numpy(dtype=np.int64)
        buckets = pd.cut(test_positions, bins=POSITION_EDGES, labels=POSITION_LABELS, right=False)
        diagnostic = pd.DataFrame({"bucket": buckets, "position": test_positions, "correct": correct})
        for bucket, group in diagnostic.groupby("bucket", observed=False):
            position_rows.append(
                {"layer_ix": layer, "position_bucket": str(bucket), "accuracy": float(group["correct"].mean()), "n": len(group)}
            )
        for position, group in diagnostic.groupby("position"):
            exact_position_rows.append(
                {"layer_ix": layer, "token_in_seg_ix": int(position), "accuracy": float(group["correct"].mean()), "n": len(group)}
            )

        np.savez_compressed(
            prediction_dir / f"layer-{layer:02d}.npz",
            sample_ix=test_sample_ix,
            prompt_ix=test_df["prompt_ix"].to_numpy(dtype=np.int64),
            token_in_seg_ix=test_positions,
            true_role_id=truth,
            predicted_role_id=pred,
            logits=logits,
            probabilities=probs,
            role_space=np.asarray(ROLE_SPACE),
        )
        probes.append(
            {"probe": pipeline, "acc": accuracy, "layer_ix": layer,
             "role_space": ROLE_SPACE, "roles_map": roles_map,
             "n_inputs": len(probe_df)}
        )
        del x_train, x_test
        cupy.get_default_memory_pool().free_all_blocks()

    overall = pd.DataFrame(overall_rows)
    overall.to_csv(OUTPUT_DIR / "overall-accuracy.csv", index=False)
    pd.DataFrame(role_rows).to_csv(OUTPUT_DIR / "per-role-accuracy.csv", index=False)
    pd.DataFrame(confusion_rows).to_csv(OUTPUT_DIR / "confusion-matrices.csv", index=False)
    pd.DataFrame(position_rows).to_csv(OUTPUT_DIR / "accuracy-by-position-bucket.csv", index=False)
    pd.DataFrame(exact_position_rows).to_csv(OUTPUT_DIR / "accuracy-by-token-position.csv", index=False)
    with (OUTPUT_DIR / "role-probes.pkl").open("wb") as handle:
        pickle.dump(probes, handle)
    best = overall.sort_values(["accuracy", "layer_ix"], ascending=[False, True]).iloc[0]
    selection = {
        "criterion": "maximum held-out neutral-text token accuracy only",
        "selected_layer": int(best["layer_ix"]),
        "selected_accuracy": float(best["accuracy"]),
        "paper_comparison_layer": 12,
        "paper_comparison_accuracy": float(overall.loc[overall.layer_ix == 12, "accuracy"].iloc[0]),
    }
    write_json(OUTPUT_DIR / "layer-selection.json", selection)
    return probes, selection


def build_tomato_prompts(tokenizer, load_chat_template, render_single_message):
    tomato_path = UPSTREAM_ROOT / "experiments/role-analysis/config/tomato.yaml"
    messages = yaml.safe_load(tomato_path.read_text())["gptoss-20b"]
    contents = [item["content"] for item in messages]
    message_types = ["system", "user", "cot", "assistant", "user", "cot", "assistant"]
    bos = tokenizer.bos_token or ""
    tokenizer.chat_template = load_chat_template(
        str(UPSTREAM_ROOT / "utils/chat_templates"), "gptoss-20b"
    )
    prompts = {
        "no_tags": bos + "\n".join(contents),
        "everything_in_user_tags": bos + tokenizer.apply_chat_template(
            [{"role": "user", "content": "\n".join(contents)}],
            tokenize=False,
            add_generation_prompt=False,
        ),
        "correct_tags": bos + tokenizer.apply_chat_template(
            [
                {"role": "system", "content": contents[0]},
                {"role": "user", "content": contents[1]},
                {"role": "assistant", "content": f"<think>{contents[2]}</think>{contents[3]}"},
                {"role": "user", "content": contents[4]},
                {"role": "assistant", "content": f"<think>{contents[5]}</think>{contents[6]}"},
            ],
            tokenize=False,
            add_generation_prompt=False,
        ),
    }
    # Keep an independently rendered user wrapper in metadata to catch template drift.
    expected_user = bos + render_single_message("gptoss-20b", "user", "\n".join(contents))
    if prompts["everything_in_user_tags"] != expected_user:
        raise AssertionError("Modified chat template and exact role renderer disagree for user wrapper")
    return prompts, contents, message_types


def evaluate_tomato(model, tokenizer, probes, selection, load_chat_template, render_single_message, flag_message_types):
    prompts, contents, message_types = build_tomato_prompts(
        tokenizer, load_chat_template, render_single_message
    )
    input_df = pd.DataFrame(
        {"condition": list(prompts), "prompt": list(prompts.values())}
    ).assign(prompt_ix=lambda df: range(len(df)))
    # Exact upstream tomato cell uses 512 * 4 even though all three prompts fit.
    max_length = 512 * 4
    loader = DataLoader(
        ReconstructableTextDataset(
            input_df["prompt"].tolist(), tokenizer, max_length=max_length,
            prompt_ix=input_df["prompt_ix"].tolist(),
        ),
        batch_size=3,
        shuffle=False,
        collate_fn=stack_collate,
    )
    outputs = run_and_export_states(
        model, tokenizer, run_model_return_states=hook_forward, dl=loader,
        layers_to_keep_acts=LAYERS,
    )
    labeled = (
        flag_message_types(outputs["sample_df"], contents)
        .assign(sample_ix=lambda df: range(len(df)))
        .assign(token_in_prompt_ix=lambda df: df.groupby("prompt_ix").cumcount())
        .merge(input_df[["prompt_ix", "condition"]], on="prompt_ix", how="left")
        .merge(
            pd.DataFrame({"base_message_ix": range(len(message_types)), "original_role": message_types}),
            on="base_message_ix", how="inner",
        )
    )
    hs = outputs["all_hs"].to(torch.float16)
    all_rows = []
    for save_ix, layer in enumerate(LAYERS):
        probe = next(item for item in probes if item["layer_ix"] == layer)
        features = cupy.asarray(
            hs[labeled["sample_ix"].to_numpy(), save_ix, :].to(torch.float32)
        )
        probabilities = cupy_to_numpy(probe["probe"].predict_proba(features))
        logits = cupy_to_numpy(probe["probe"].named_steps["clf"].decision_function(features))
        base = labeled.reset_index(drop=True)[
            ["sample_ix", "prompt_ix", "condition", "base_message_ix", "original_role", "token_in_prompt_ix", "token"]
        ].copy()
        for role_ix, role in enumerate(ROLE_SPACE):
            base[f"prob_{role}"] = probabilities[:, role_ix]
            base[f"logit_{role}"] = logits[:, role_ix]
        all_rows.append(base.assign(layer_ix=layer))
        cupy.get_default_memory_pool().free_all_blocks()
    result = pd.concat(all_rows, ignore_index=True)
    result["token_in_message_ix"] = result.groupby(
        ["layer_ix", "condition", "base_message_ix"]
    ).cumcount()
    result.to_csv(OUTPUT_DIR / "tomato-token-projections.csv.gz", index=False, compression="gzip")
    probability_columns = [f"prob_{role}" for role in ROLE_SPACE]
    summary = (
        result.groupby(["layer_ix", "condition", "original_role"], as_index=False)[probability_columns]
        .mean()
    )
    summary.to_csv(OUTPUT_DIR / "tomato-summary.csv", index=False)

    plot_dir = OUTPUT_DIR / "plots"
    plot_dir.mkdir(exist_ok=True)
    for layer in LAYERS:
        fig, axes = plt.subplots(3, 1, figsize=(13, 8), sharey=True)
        for axis, condition in zip(axes, ["correct_tags", "no_tags", "everything_in_user_tags"], strict=True):
            frame = result[(result.layer_ix == layer) & (result.condition == condition)]
            for role, color in {"user": "#3288bd", "cot": "#f39c12", "assistant": "#36a657"}.items():
                part = frame[frame.original_role == role]
                axis.scatter(part.token_in_prompt_ix, part.prob_cot, s=5, alpha=0.65, color=color, label=role)
            cot_mean = frame.loc[frame.original_role == "cot", "prob_cot"].mean()
            axis.set_title(f"{condition.replace('_', ' ')} — mean CoTness on CoT text: {cot_mean:.1%}")
            axis.set_ylabel("CoTness")
            axis.set_ylim(-0.03, 1.03)
            axis.grid(axis="y", alpha=0.2)
        axes[0].legend(ncol=3, loc="lower right")
        axes[-1].set_xlabel("Token position")
        fig.suptitle(f"Exact upstream tomato conditions — layer {layer}")
        fig.tight_layout()
        fig.savefig(plot_dir / f"tomato-tag-conditions-layer-{layer:02d}.png", dpi=160)
        plt.close(fig)

    write_json(
        OUTPUT_DIR / "tomato-metadata.json",
        {
            "source": str(UPSTREAM_ROOT / "experiments/role-analysis/config/tomato.yaml"),
            "conditions": ["correct_tags", "no_tags", "everything_in_user_tags"],
            "layers": LAYERS,
            "bos_token": tokenizer.bos_token,
            "separators": {"no_tags_between_messages": "\\n", "tagged_between_messages": ""},
            "paper_comparison_layer": 12,
            "neutral_selected_layer": selection["selected_layer"],
        },
    )


def plot_accuracy(selection) -> None:
    accuracy = pd.read_csv(OUTPUT_DIR / "overall-accuracy.csv")
    fig, axis = plt.subplots(figsize=(8, 4.5))
    axis.plot(accuracy.layer_ix, accuracy.accuracy, marker="o")
    axis.axvline(12, color="#f39c12", linestyle="--", label="paper comparison: layer 12")
    axis.axvline(selection["selected_layer"], color="#36a657", linestyle=":", label=f"neutral best: layer {selection['selected_layer']}")
    axis.set(xlabel="Layer", ylabel="Held-out neutral-text accuracy", ylim=(0, 1), xticks=LAYERS)
    axis.grid(alpha=0.2)
    axis.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "plots/neutral-accuracy-by-layer.png", dpi=180)
    plt.close(fig)


def write_checksums() -> None:
    rows = []
    for path in sorted(OUTPUT_DIR.rglob("*")):
        if not path.is_file() or path.name == "sha256sums.txt":
            continue
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(block)
        rows.append(f"{digest.hexdigest()}  {path.relative_to(OUTPUT_DIR)}")
    (OUTPUT_DIR / "sha256sums.txt").write_text("\n".join(rows) + "\n")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=False)
    require_pinned_upstream()
    record_upstream_provenance()
    from utils.pretrained_models import gptoss as upstream_gptoss
    from utils.role_templates import load_chat_template, render_single_message
    from utils.substring_assignments import flag_message_types
    patch_pinned_masking_api(upstream_gptoss)

    metadata = {
        "upstream_commit": UPSTREAM_COMMIT,
        "model": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "c4_revision": C4_REVISION,
        "dolma_revision": DOLMA_REVISION,
        "seed": SEED,
        "requested_passages": N_PASSAGES,
        "source_counts": {"c4": C4_COUNT, "dolma3": DOLMA_COUNT},
        "source_fractions": {"c4": C4_COUNT / N_PASSAGES, "dolma3": DOLMA_COUNT / N_PASSAGES},
        "integer_allocation_note": "250 cannot be divided exactly 25/75; floor C4 and assign remainder to Dolma",
        "max_content_tokens": MAX_CONTENT_TOKENS,
        "layers": LAYERS,
        "roles": ROLE_SPACE,
        "rendered_roles": ALL_RENDERED_ROLES,
        "split_group": "prompt_ix",
        "test_size": 0.1,
        "activation": "pre-MLP (post_attention_layernorm output)",
        "classifier": {"type": "cuML L2 logistic regression", "C": 0.005, "scaling": False, "max_iter": 5000, "linesearch_max_iter": 100},
        "batch_size": BATCH_SIZE,
    }
    write_json(OUTPUT_DIR / "run-metadata.json", metadata)
    mark_phase("loading-model")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, cache_dir=os.environ.get("HF_HOME"),
        add_eos_token=False, add_bos_token=False, padding_side="left",
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, cache_dir=os.environ.get("HF_HOME"),
        attn_implementation="kernels-community/vllm-flash-attn3",
    ).to("cuda:0").eval()
    model.set_experts_implementation("eager")
    patch_pinned_model_api(model)
    validate_pre_mlp_capture(model, tokenizer, upstream_gptoss.run_gptoss_return_topk)

    mark_phase("loading-neutral-data")
    raw_data = load_raw_data()
    with gzip.open(OUTPUT_DIR / "neutral-passages.jsonl.gz", "wt", encoding="utf-8") as handle:
        for passage_ix, row in enumerate(raw_data):
            handle.write(json.dumps({"passage_ix": passage_ix, **row}, ensure_ascii=False) + "\n")
    input_df = build_training_prompts(tokenizer, raw_data, render_single_message)
    manifest = input_df[["prompt_ix", "question_ix", "source", "role"]].copy()
    manifest["content_sha256"] = input_df.question.map(lambda x: hashlib.sha256(x.encode()).hexdigest())
    manifest.to_csv(OUTPUT_DIR / "training-manifest.csv.gz", index=False, compression="gzip")
    max_length = max(len(ids) for ids in tokenizer(input_df.prompt.tolist(), add_special_tokens=False).input_ids)
    loader = DataLoader(
        ReconstructableTextDataset(
            input_df.prompt.tolist(), tokenizer, max_length=max_length,
            prompt_ix=input_df.prompt_ix.tolist(),
        ), batch_size=BATCH_SIZE, shuffle=False, collate_fn=stack_collate,
    )

    mark_phase("extracting-training-pre-mlp-activations", prompts=len(input_df), max_rendered_tokens=max_length)
    outputs = run_and_export_states(
        model, tokenizer, run_model_return_states=hook_forward, dl=loader,
        layers_to_keep_acts=LAYERS,
    )
    activations = outputs["all_hs"].to(torch.float16)
    torch.save(
        {"layers": LAYERS, "dtype": "float16", "activations": activations},
        OUTPUT_DIR / "training-pre-mlp-activations.pt",
    )
    sample_df = outputs["sample_df"].assign(sample_ix=lambda df: range(len(df)))
    probe_df = (
        label_gptoss_content_roles(sample_df)
        .merge(input_df[["prompt_ix", "role"]].rename(columns={"role": "target_role"}), on="prompt_ix", how="inner")
        .query("is_content == True and role == target_role and role in @ROLE_SPACE")
        .copy()
    )
    probe_df[["sample_ix", "prompt_ix", "token_ix", "token_in_seg_ix", "role"]].to_csv(
        OUTPUT_DIR / "probe-token-index.csv.gz", index=False, compression="gzip"
    )

    mark_phase("training-probes", n_probe_tokens=len(probe_df))
    probes, selection = train_probes(probe_df, activations)
    del activations, outputs
    gc.collect()

    mark_phase("evaluating-tomato", selected_layer=selection["selected_layer"])
    evaluate_tomato(
        model, tokenizer, probes, selection, load_chat_template,
        render_single_message, flag_message_types,
    )
    plot_accuracy(selection)
    mark_phase("checksumming")
    write_checksums()
    mark_phase("complete", selected_layer=selection["selected_layer"])
    print(f"Exact single-seed run complete: {OUTPUT_DIR}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        if OUTPUT_DIR.exists():
            mark_phase("failed", error=repr(error))
        raise
