#!/usr/bin/env python3
"""Resumable GPT-OSS-20B Assistant Axis pilot runner.

The initial commands implement Gate 1 artifact/environment inventory and the
Gate 2 dual-site hook smoke test. Later experiment stages are added as their
preceding gates pass.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import gzip
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import cupy
import cuml
from dotenv import load_dotenv
from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.masking_utils import (
    create_causal_mask as current_create_causal_mask,
    create_sliding_window_causal_mask as current_create_sliding_window_causal_mask,
)


ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")
sys.path.insert(0, str(ROOT))

MODEL_ID = "openai/gpt-oss-20b"
MODEL_REVISION = "6cee5e81ee83917806bbde320786a8fb61efebee"
UPSTREAM_COMMIT = "ec333c40fd43fe991e1ebf66765051b6d7e35784"
STORAGE_ROOT = Path(os.environ.get("ROLE_PROBE_STORAGE_ROOT", "/workspace/role-probe-storage"))
HF_HOME = Path(os.environ.get("HF_HOME", STORAGE_ROOT / "huggingface"))
UPSTREAM_ROOT = Path(
    os.environ.get("ROLE_PROBE_UPSTREAM_ROOT", STORAGE_ROOT / "upstream" / UPSTREAM_COMMIT)
)
PRIOR_RUN = STORAGE_ROOT / "outputs" / "exact-full-pipeline-seed123-v3"
LAYERS = [12, 16]
ASSISTANT_AXIS_COMMIT = "a98961956072224eaf244eb289d6c01700b63795"
ASSISTANT_AXIS_ROOT = STORAGE_ROOT / "upstream" / f"assistant-axis-{ASSISTANT_AXIS_COMMIT}"
PERSONA_FAMILIES = {
    "tutor": "helpful_professional",
    "consultant": "helpful_professional",
    "mediator": "helpful_professional",
    "librarian": "ordinary_occupation",
    "engineer": "ordinary_occupation",
    "bartender": "ordinary_occupation",
    "pirate": "theatrical_fictional",
    "comedian": "theatrical_fictional",
    "genie": "theatrical_fictional",
    "mystic": "spiritual_adversarial",
    "anarchist": "spiritual_adversarial",
    "devils_advocate": "spiritual_adversarial",
}
MICRO_PERSONAS = ["tutor", "librarian", "engineer", "pirate", "mystic", "anarchist"]
QUESTION_IDS = [0, 5]
QUESTION_EXPANSION_IDS = [0, 2, 5, 51, 119, 190]
JUDGE_MODEL = "gpt-4.1-mini"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_text(command: list[str], *, check: bool = True) -> str:
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    if check and result.returncode:
        raise RuntimeError(
            f"Command failed ({result.returncode}): {' '.join(command)}\n{result.stderr}"
        )
    return result.stdout.strip()


def package_version(name: str) -> str | None:
    try:
        module = __import__(name)
        return str(getattr(module, "__version__", "unknown"))
    except Exception as exc:  # pragma: no cover - recorded diagnostic
        return f"ERROR: {type(exc).__name__}: {exc}"


def parse_recorded_checksums(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    if not path.is_file():
        return result
    for raw in path.read_text(encoding="utf-8").splitlines():
        parts = raw.split(maxsplit=1)
        if len(parts) == 2:
            result[parts[1].strip()] = parts[0]
    return result


def artifact_record(
    path: Path,
    *,
    prior_checksums: dict[str, str],
    hash_now: bool,
    authoritative_lost: bool = False,
) -> dict[str, Any]:
    relative = str(path.relative_to(PRIOR_RUN)) if path.is_relative_to(PRIOR_RUN) else None
    record: dict[str, Any] = {
        "path": str(path),
        "exists": path.exists(),
        "readable": os.access(path, os.R_OK),
        "kind": "directory" if path.is_dir() else "file" if path.is_file() else "missing",
        "recorded_sha256": prior_checksums.get(relative or ""),
        "authoritative_status": "lost_do_not_reuse" if authoritative_lost else "candidate_for_reuse",
    }
    if path.is_file():
        record["bytes"] = path.stat().st_size
        try:
            with path.open("rb") as handle:
                record["read_probe_hex"] = handle.read(16).hex()
        except Exception as exc:
            record["read_error"] = repr(exc)
        if hash_now:
            record["observed_sha256"] = sha256_file(path)
            recorded = record["recorded_sha256"]
            record["digest_matches_recorded"] = recorded == record["observed_sha256"] if recorded else None
    elif path.is_dir():
        children = sorted(x for x in path.iterdir() if x.is_file())
        record["file_count"] = len(children)
        record["total_bytes"] = sum(x.stat().st_size for x in children)
    return record


def atomic_stage_dir(run_dir: Path, name: str) -> tuple[Path, Path]:
    run_dir.mkdir(parents=True, exist_ok=True)
    final = run_dir / name
    temporary = run_dir / f".{name}.tmp"
    if final.exists():
        raise FileExistsError(f"Refusing to overwrite completed stage: {final}")
    if temporary.exists():
        raise FileExistsError(f"Incomplete stage exists and needs inspection: {temporary}")
    temporary.mkdir()
    return temporary, final


def command_inventory(args: argparse.Namespace) -> None:
    temporary, final = atomic_stage_dir(args.run_dir, "gate-1-environment")
    prior_checksums = parse_recorded_checksums(PRIOR_RUN / "sha256sums.txt")
    inventory_targets = [
        (PRIOR_RUN / "neutral-passages.jsonl.gz", True, False),
        (PRIOR_RUN / "training-manifest.csv.gz", True, False),
        (PRIOR_RUN / "prompt-split.csv", True, False),
        (PRIOR_RUN / "probe-token-index.csv.gz", True, False),
        (PRIOR_RUN / "role-probes.pkl", False, True),
        (PRIOR_RUN / "training-pre-mlp-activations.pt", False, True),
        (PRIOR_RUN / "heldout-predictions", False, True),
    ]
    artifacts = [
        artifact_record(
            path,
            prior_checksums=prior_checksums,
            hash_now=hash_now,
            authoritative_lost=lost,
        )
        for path, hash_now, lost in inventory_targets
    ]

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        cache_dir=HF_HOME,
        local_files_only=True,
        add_eos_token=False,
        add_bos_token=False,
        padding_side="left",
    )
    config = AutoConfig.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, cache_dir=HF_HOME, local_files_only=True
    )
    tokenizer_payload = {
        "class": type(tokenizer).__name__,
        "vocab_size": len(tokenizer),
        "bos_token": tokenizer.bos_token,
        "eos_token": tokenizer.eos_token,
        "padding_side": tokenizer.padding_side,
        "chat_template": tokenizer.chat_template,
    }
    config_payload = config.to_dict()
    template_path = UPSTREAM_ROOT / "utils/chat_templates/gptoss.j2"
    stat = shutil.disk_usage(STORAGE_ROOT)
    environment = {
        "captured_at": utc_now(),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": sys.version,
        "repository": {
            "commit": run_text(["git", "-C", str(ROOT), "rev-parse", "HEAD"]),
            "branch": run_text(["git", "-C", str(ROOT), "branch", "--show-current"]),
            "status_short": run_text(["git", "-C", str(ROOT), "status", "--short"]),
        },
        "gpu": {
            "query": run_text(
                [
                    "nvidia-smi",
                    "--query-gpu=name,memory.total,memory.free,driver_version",
                    "--format=csv,noheader",
                ]
            ),
            "cuda_runtime": torch.version.cuda,
        },
        "versions": {
            name: package_version(name)
            for name in ["torch", "transformers", "datasets", "numpy", "pandas", "sklearn", "cuml", "cupy"]
        },
        "model": {
            "identifier": MODEL_ID,
            "revision": MODEL_REVISION,
            "hidden_size": config.hidden_size,
            "num_hidden_layers": config.num_hidden_layers,
            "config_sha256": sha256_bytes(
                json.dumps(config_payload, sort_keys=True, separators=(",", ":")).encode()
            ),
            "cached_snapshot": str(
                HF_HOME / f"models--{MODEL_ID.replace('/', '--')}" / "snapshots" / MODEL_REVISION
            ),
        },
        "tokenizer": {
            "sha256": sha256_bytes(
                json.dumps(tokenizer_payload, sort_keys=True, separators=(",", ":")).encode()
            ),
            **{key: value for key, value in tokenizer_payload.items() if key != "chat_template"},
            "chat_template_sha256": sha256_bytes((tokenizer.chat_template or "").encode()),
        },
        "pinned_upstream": {
            "path": str(UPSTREAM_ROOT),
            "expected_commit": UPSTREAM_COMMIT,
            "observed_commit": run_text(["git", "-C", str(UPSTREAM_ROOT), "rev-parse", "HEAD"]),
            "gptoss_template": str(template_path),
            "gptoss_template_sha256": sha256_file(template_path),
        },
        "persistent_storage": {
            "path": str(STORAGE_ROOT),
            "total_bytes": stat.total,
            "used_bytes": stat.used,
            "free_bytes": stat.free,
        },
    }
    write_json(temporary / "environment.json", environment)
    write_json(
        temporary / "artifact-inventory.json",
        {
            "captured_at": utc_now(),
            "prior_run": str(PRIOR_RUN),
            "user_statement_authoritative": (
                "Raw probe objects, coefficient vectors, large activation archive, and raw "
                "held-out logits/probabilities are treated as lost and are not reused even if "
                "a stale readable path is present."
            ),
            "artifacts": artifacts,
            "reuse_decision": {
                "neutral_passages": "reuse only because observed digest matches recorded digest",
                "training_manifest": "reuse only because observed digest matches recorded digest",
                "split_ids": "reuse only because observed digest matches recorded digest",
                "raw_probe_objects": "do_not_reuse_regenerate",
                "activation_archive": "do_not_reuse_regenerate_compactly",
                "heldout_predictions": "do_not_reuse_regenerate",
            },
        },
    )
    if environment["model"]["hidden_size"] != 2880 or environment["model"]["num_hidden_layers"] != 24:
        raise AssertionError("Pinned model configuration does not match the handoff plan")
    for artifact in artifacts[:4]:
        if not artifact.get("digest_matches_recorded"):
            raise AssertionError(f"Reusable artifact digest failed: {artifact['path']}")
    temporary.rename(final)
    print(json.dumps({"stage": "gate-1-environment", "status": "complete", "path": str(final)}, sort_keys=True))


def patch_pinned_masking_api(upstream_module: Any) -> None:
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


def patch_pinned_model_api(model: Any) -> None:
    for layer, attention_type in zip(
        model.model.layers, model.config.layer_types, strict=True
    ):
        layer.attention_type = attention_type


def tensor_comparison(left: torch.Tensor, right: torch.Tensor) -> dict[str, Any]:
    left = left.detach().float().cpu()
    right = right.detach().float().cpu()
    if left.shape != right.shape:
        return {"equal_shape": False, "left_shape": list(left.shape), "right_shape": list(right.shape)}
    flat_left = left.reshape(-1)
    flat_right = right.reshape(-1)
    return {
        "equal_shape": True,
        "shape": list(left.shape),
        "finite_left": bool(torch.isfinite(left).all()),
        "finite_right": bool(torch.isfinite(right).all()),
        "bit_exact": bool(torch.equal(left, right)),
        "max_abs_error": float((left - right).abs().max()),
        "cosine_similarity": float(torch.nn.functional.cosine_similarity(flat_left, flat_right, dim=0)),
    }


@torch.no_grad()
def capture_forward(
    model: Any,
    inputs: dict[str, torch.Tensor],
    sites: tuple[str, ...],
    layers: list[int] | None = None,
):
    layers = layers or LAYERS
    captured: dict[str, dict[int, torch.Tensor]] = {site: {} for site in sites}
    handles = []
    try:
        for layer_ix in layers:
            layer = model.model.layers[layer_ix]
            if "pre_mlp" in sites:
                def pre_hook(_module, _inputs, output, layer_ix=layer_ix):
                    captured["pre_mlp"][layer_ix] = output.detach().cpu()
                handles.append(layer.post_attention_layernorm.register_forward_hook(pre_hook))
            if "block_output" in sites:
                def block_hook(_module, _inputs, output, layer_ix=layer_ix):
                    tensor = output[0] if isinstance(output, tuple) else output
                    captured["block_output"][layer_ix] = tensor.detach().cpu()
                handles.append(layer.register_forward_hook(block_hook))
        outputs = model(**inputs, use_cache=False, return_dict=True)
    finally:
        for handle in handles:
            handle.remove()
    return outputs, captured, len(handles)


def cupy_to_numpy(value: Any) -> np.ndarray:
    return value.get() if hasattr(value, "get") else np.asarray(value)


def centered_coefficients(coef: np.ndarray, n_classes: int) -> np.ndarray:
    coef = np.asarray(coef, dtype=np.float32)
    if coef.shape[0] == 1 and n_classes == 2:
        coef = np.stack([-0.5 * coef[0], 0.5 * coef[0]], axis=0)
    return coef - coef.mean(axis=0, keepdims=True)


def balanced_indices(frame: pd.DataFrame, roles: list[str]) -> np.ndarray:
    counts = frame.groupby("role").size().reindex(roles)
    if counts.isna().any() or int(counts.min()) <= 0:
        raise AssertionError(f"Missing role tokens while balancing: {counts.to_dict()}")
    keep = int(counts.min())
    return np.concatenate(
        [frame.loc[frame.role == role].sort_values(["base_sequence_id", "token_in_seg_ix", "sample_ix"]).index[:keep]
         for role in roles]
    )


def fit_one_probe(
    *,
    features: np.ndarray,
    token_frame: pd.DataFrame,
    roles: list[str],
    train_bases: set[int],
    test_bases: set[int],
    solver_name: str,
) -> tuple[dict[str, Any], dict[str, np.ndarray], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    role_to_id = {role: index for index, role in enumerate(roles)}
    eligible = token_frame[token_frame.role.isin(roles)].copy()
    train_frame = eligible[eligible.base_sequence_id.isin(train_bases)]
    test_frame = eligible[eligible.base_sequence_id.isin(test_bases)]
    train_index = balanced_indices(train_frame, roles)
    test_index = balanced_indices(test_frame, roles)
    train = token_frame.loc[train_index]
    test = token_frame.loc[test_index]
    x_train = np.asarray(features[train.sample_ix.to_numpy()], dtype=np.float32)
    x_test = np.asarray(features[test.sample_ix.to_numpy()], dtype=np.float32)
    y_train = train.role.map(role_to_id).to_numpy(dtype=np.int32)
    y_test = test.role.map(role_to_id).to_numpy(dtype=np.int32)
    warning_messages: list[str] = []
    scaler_mean = np.zeros(x_train.shape[1], dtype=np.float32)
    scaler_scale = np.ones(x_train.shape[1], dtype=np.float32)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        if solver_name == "cuml_exact_prior":
            classifier = cuml.linear_model.LogisticRegression(
                penalty="l2",
                max_iter=5_000,
                linesearch_max_iter=100,
                fit_intercept=True,
                C=5.0e-3,
            )
            classifier.fit(cupy.asarray(x_train), cupy.asarray(y_train))
            pred = cupy_to_numpy(classifier.predict(cupy.asarray(x_test))).astype(np.int32)
            probabilities = cupy_to_numpy(classifier.predict_proba(cupy.asarray(x_test))).astype(np.float64)
            scaled_coef = cupy_to_numpy(classifier.coef_).astype(np.float32)
            intercept = cupy_to_numpy(classifier.intercept_).reshape(-1).astype(np.float32)
            n_iter = getattr(classifier, "n_iter_", None)
            n_iter = int(cupy_to_numpy(n_iter).reshape(-1)[0]) if n_iter is not None else None
            raw_coef = scaled_coef
        elif solver_name == "sklearn_standardized":
            scaler = StandardScaler().fit(x_train)
            scaler_mean = scaler.mean_.astype(np.float32)
            scaler_scale = scaler.scale_.astype(np.float32)
            classifier = SklearnLogisticRegression(
                penalty="l2", C=5.0e-3, fit_intercept=True, solver="lbfgs", max_iter=5_000,
            )
            classifier.fit(scaler.transform(x_train), y_train)
            pred = classifier.predict(scaler.transform(x_test)).astype(np.int32)
            probabilities = classifier.predict_proba(scaler.transform(x_test)).astype(np.float64)
            scaled_coef = classifier.coef_.astype(np.float32)
            intercept_scaled = classifier.intercept_.astype(np.float32)
            raw_coef = scaled_coef / scaler_scale[None, :]
            intercept = intercept_scaled - (scaled_coef * (scaler_mean / scaler_scale)[None, :]).sum(axis=1)
            n_iter = int(np.max(classifier.n_iter_))
        else:
            raise ValueError(solver_name)
        warning_messages = [str(item.message) for item in caught]

    line_search_warning = any("line" in text.lower() and "search" in text.lower() for text in warning_messages)
    converged = (n_iter is None or n_iter < 5_000) and not line_search_warning
    accuracy = float(accuracy_score(y_test, pred))
    balanced = float(balanced_accuracy_score(y_test, pred))
    nll = float(log_loss(y_test, probabilities, labels=list(range(len(roles)))))
    uniform_nll = float(np.log(len(roles)))
    matrix = confusion_matrix(y_test, pred, labels=list(range(len(roles))))
    metrics = {
        "accuracy": accuracy,
        "balanced_accuracy": balanced,
        "nll": nll,
        "uniform_nll": uniform_nll,
        "nll_beats_uniform": nll < uniform_nll,
        "n_train_tokens": int(len(y_train)),
        "n_test_tokens": int(len(y_test)),
        "tokens_per_train_class": int(len(y_train) // len(roles)),
        "tokens_per_test_class": int(len(y_test) // len(roles)),
        "n_iter": n_iter,
        "converged": converged,
        "warnings": warning_messages,
        "coefficient_norms": np.linalg.norm(raw_coef, axis=1).astype(float).tolist(),
        "coefficients_finite": bool(np.isfinite(raw_coef).all() and np.isfinite(intercept).all()),
    }
    arrays = {
        "coefficients_raw": raw_coef,
        "coefficients_scaled": scaled_coef,
        "centered_coefficients_raw": centered_coefficients(raw_coef, len(roles)),
        "intercepts_raw": intercept,
        "scaler_mean": scaler_mean,
        "scaler_scale": scaler_scale,
        "class_labels": np.asarray(roles),
    }
    confusion_rows = [
        {
            "true_role": roles[true_ix],
            "predicted_role": roles[pred_ix],
            "count": int(matrix[true_ix, pred_ix]),
        }
        for true_ix in range(len(roles)) for pred_ix in range(len(roles))
    ]
    per_role_rows = []
    for role_ix, role in enumerate(roles):
        mask = y_test == role_ix
        per_role_rows.append(
            {"role": role, "accuracy": float((pred[mask] == y_test[mask]).mean()), "n": int(mask.sum())}
        )
    position_rows = []
    position_values = test.token_in_seg_ix.to_numpy(dtype=np.int64)
    for position in np.unique(position_values):
        mask = position_values == position
        position_rows.append(
            {"token_in_seg_ix": int(position), "accuracy": float((pred[mask] == y_test[mask]).mean()), "n": int(mask.sum())}
        )
    return metrics, arrays, confusion_rows, per_role_rows, position_rows


def command_role_probes(args: argparse.Namespace) -> None:
    temporary, final = atomic_stage_dir(args.run_dir, "gate-3-role-probes")
    if not (args.run_dir / "gate-2-hook-smoke").is_dir():
        raise RuntimeError("Gate 2 must complete before Gate 3")
    sys.path.insert(0, str(UPSTREAM_ROOT))
    sys.path.insert(0, str(ROOT))
    from demo.simple_test_helpers import ReconstructableTextDataset, label_gptoss_content_roles, stack_collate
    from utils.role_templates import render_single_message

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, cache_dir=HF_HOME, local_files_only=True,
        add_eos_token=False, add_bos_token=False, padding_side="left",
    )
    passages = []
    with gzip.open(PRIOR_RUN / "neutral-passages.jsonl.gz", "rt", encoding="utf-8") as handle:
        for line in handle:
            passages.append(json.loads(line))
            if len(passages) == 50:
                break
    if len(passages) != 50:
        raise AssertionError("Expected 50 reusable neutral passages")
    truncated = tokenizer.batch_decode(
        tokenizer(
            [row["text"] for row in passages], add_special_tokens=False,
            truncation=True, max_length=128, padding=False,
        ).input_ids
    )
    rendered_roles = ["system", "user", "tool", "cot", "assistant"]
    prompt_rows = []
    for base_sequence_id, (text, source_row) in enumerate(zip(truncated, passages, strict=True)):
        for role in rendered_roles:
            prompt_rows.append(
                {
                    "prompt_id": len(prompt_rows),
                    "base_sequence_id": base_sequence_id,
                    "source": source_row["source"],
                    "source_text_sha256": sha256_bytes(source_row["text"].encode()),
                    "truncated_content_sha256": sha256_bytes(text.encode()),
                    "role": role,
                    "prompt": render_single_message("gptoss-20b", role, text),
                }
            )
    prompt_frame = pd.DataFrame(prompt_rows)
    prompt_frame.to_csv(temporary / "prompt-manifest.csv", index=False)
    max_length = max(
        len(ids) for ids in tokenizer(prompt_frame.prompt.tolist(), add_special_tokens=False).input_ids
    )
    dataset = ReconstructableTextDataset(
        prompt_frame.prompt.tolist(), tokenizer, max_length=max_length,
        prompt_ix=prompt_frame.prompt_id.tolist(),
    )
    loader = DataLoader(dataset, batch_size=128, shuffle=False, collate_fn=stack_collate)
    extraction_layers = [8, 10, 12, 14, 16, 18]
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, cache_dir=HF_HOME, local_files_only=True,
        attn_implementation="kernels-community/vllm-flash-attn3",
    ).to("cuda:0").eval()
    model.set_experts_implementation("eager")
    patch_pinned_model_api(model)
    torch.cuda.reset_peak_memory_stats()
    activation_parts: dict[str, dict[int, list[torch.Tensor]]] = {
        site: {layer: [] for layer in extraction_layers} for site in ["pre_mlp", "block_output"]
    }
    token_parts: list[pd.DataFrame] = []
    for batch_ix, batch in enumerate(loader):
        inputs = {
            "input_ids": batch["input_ids"].to(model.device),
            "attention_mask": batch["attention_mask"].to(model.device),
        }
        outputs, captured, handle_count = capture_forward(
            model, inputs, ("pre_mlp", "block_output"), extraction_layers
        )
        if handle_count != len(extraction_layers) * 2:
            raise AssertionError("Unexpected dual-hook count")
        valid = batch["attention_mask"].bool()
        rows = []
        for sequence_ix, prompt_id in enumerate(batch["prompt_ix"]):
            for token_ix in torch.where(valid[sequence_ix])[0].tolist():
                rows.append(
                    {
                        "prompt_ix": int(prompt_id),
                        "token_ix": int(token_ix),
                        "token_id": int(batch["input_ids"][sequence_ix, token_ix]),
                        "token": batch["original_tokens"][sequence_ix][token_ix],
                        "batch_ix": batch_ix,
                    }
                )
        token_parts.append(pd.DataFrame(rows))
        for site in activation_parts:
            for layer_ix in extraction_layers:
                activation_parts[site][layer_ix].append(
                    captured[site][layer_ix][valid].to(torch.float16)
                )
        del outputs, captured, inputs
        torch.cuda.empty_cache()

    token_frame = pd.concat(token_parts, ignore_index=True).assign(sample_ix=lambda frame: range(len(frame)))
    token_frame = label_gptoss_content_roles(token_frame).merge(
        prompt_frame[["prompt_id", "base_sequence_id", "role"]].rename(
            columns={"prompt_id": "prompt_ix", "role": "target_role"}
        ),
        on="prompt_ix", how="left",
    )
    token_frame = token_frame[
        token_frame.is_content & token_frame.role.notna() & (token_frame.role == token_frame.target_role)
    ].copy()
    token_frame[[
        "sample_ix", "prompt_ix", "base_sequence_id", "role", "token_ix", "token_in_seg_ix", "token_id"
    ]].to_csv(temporary / "probe-token-index.csv.gz", index=False, compression="gzip")
    activations = {
        site: {
            layer: torch.cat(parts, dim=0).numpy()
            for layer, parts in by_layer.items()
        }
        for site, by_layer in activation_parts.items()
    }
    if not all(np.isfinite(value).all() for by_layer in activations.values() for value in by_layer.values()):
        raise AssertionError("Non-finite role-probe activations")

    rng = np.random.default_rng(123)
    ordered = rng.permutation(50).tolist()
    pilot_test = set(ordered[:4])
    pilot_train = set(ordered[4:20])
    compact_test = set(ordered[:10])
    compact_train = set(ordered[10:50])
    split_rows = [
        {
            "base_sequence_id": base,
            "pilot_split": "test" if base in pilot_test else "train" if base in pilot_train else "unused",
            "compact_split": "test" if base in compact_test else "train",
        }
        for base in range(50)
    ]
    pd.DataFrame(split_rows).to_csv(temporary / "split-manifest.csv", index=False)

    fit_specs = [
        ("pilot_binary", ["user", "assistant"], pilot_train, pilot_test, extraction_layers),
        ("pilot_plus_tool", ["user", "assistant", "tool"], pilot_train, pilot_test, extraction_layers),
        ("pilot_plus_tool_cot", ["user", "assistant", "tool", "cot"], pilot_train, pilot_test, extraction_layers),
        ("compact_system_user_cot_assistant", ["system", "user", "cot", "assistant"], compact_train, compact_test, [12, 16]),
    ]
    metric_rows: list[dict[str, Any]] = []
    confusion_rows: list[dict[str, Any]] = []
    per_role_rows: list[dict[str, Any]] = []
    position_rows: list[dict[str, Any]] = []
    artifact_arrays: dict[str, np.ndarray] = {}
    invalid_fits: list[dict[str, Any]] = []
    for probe_name, roles, train_bases, test_bases, layers in fit_specs:
        for site in ["pre_mlp", "block_output"]:
            for layer_ix in layers:
                for solver_name in ["cuml_exact_prior", "sklearn_standardized"]:
                    print(f"Fitting {probe_name} {site} layer {layer_ix} {solver_name}", flush=True)
                    metrics, arrays, confusion, per_role, positions = fit_one_probe(
                        features=activations[site][layer_ix], token_frame=token_frame,
                        roles=roles, train_bases=train_bases, test_bases=test_bases,
                        solver_name=solver_name,
                    )
                    prefix = f"{probe_name}__{site}__layer_{layer_ix:02d}__{solver_name}"
                    for key, value in arrays.items():
                        artifact_arrays[f"{prefix}__{key}"] = value
                    common = {
                        "probe": probe_name, "activation_site": site, "layer": layer_ix,
                        "solver": solver_name, "roles": "|".join(roles),
                    }
                    metric_rows.append({**common, **metrics, "warnings": json.dumps(metrics["warnings"])})
                    confusion_rows += [{**common, **row} for row in confusion]
                    per_role_rows += [{**common, **row} for row in per_role]
                    position_rows += [{**common, **row} for row in positions]
                    if not metrics["converged"] or not metrics["coefficients_finite"] or not metrics["nll_beats_uniform"]:
                        invalid_fits.append({**common, "metrics": metrics})
                    cupy.get_default_memory_pool().free_all_blocks()

    metrics_frame = pd.DataFrame(metric_rows)
    metrics_frame.to_csv(temporary / "probe-metrics.csv", index=False)
    pd.DataFrame(confusion_rows).to_csv(temporary / "confusion-matrices.csv", index=False)
    pd.DataFrame(per_role_rows).to_csv(temporary / "per-role-metrics.csv", index=False)
    pd.DataFrame(position_rows).to_csv(temporary / "accuracy-by-token-position.csv", index=False)
    np.savez_compressed(temporary / "probe-coefficients.npz", **artifact_arrays)
    with np.load(temporary / "probe-coefficients.npz") as reloaded:
        if set(reloaded.files) != set(artifact_arrays):
            raise AssertionError("Probe artifact key mismatch after reload")
        if not all(np.isfinite(reloaded[key]).all() for key in reloaded.files if reloaded[key].dtype.kind in "fc"):
            raise AssertionError("Non-finite coefficient artifact after reload")

    stable_binary = metrics_frame[
        (metrics_frame.probe == "pilot_binary") & (metrics_frame.solver == "sklearn_standardized")
    ]
    best_by_site = stable_binary.groupby("activation_site").balanced_accuracy.max().to_dict()
    compact_pre16 = metrics_frame[
        (metrics_frame.probe == "compact_system_user_cot_assistant")
        & (metrics_frame.solver == "sklearn_standardized")
        & (metrics_frame.activation_site == "pre_mlp")
        & (metrics_frame.layer == 16)
    ].iloc[0]
    if invalid_fits:
        decision = "stop_numerically_invalid_fit"
    elif best_by_site.get("pre_mlp", 0) < 0.65 and best_by_site.get("block_output", 0) < 0.65:
        decision = "stop_both_sites_below_0.65"
    elif best_by_site.get("block_output", 0) >= 0.80:
        decision = "retain_block_output_as_primary_common_coordinate_system"
    elif best_by_site.get("pre_mlp", 0) >= 0.80 and best_by_site.get("block_output", 0) < 0.65:
        decision = "plan_pre_mlp_assistant_axis_adaptation"
    else:
        decision = "expand_role_pilot_to_50_before_persona_generation"
    expansion_required = float(compact_pre16.balanced_accuracy) < 0.80
    write_json(
        temporary / "gate-decision.json",
        {
            "binary_best_balanced_accuracy_by_site": best_by_site,
            "compact_layer16_pre_mlp_balanced_accuracy": float(compact_pre16.balanced_accuracy),
            "role_pilot_decision": decision,
            "compact_probe_expansion_to_100_required": expansion_required,
            "invalid_fits": invalid_fits,
            "activation_capture": {
                "layers": extraction_layers, "sites": ["pre_mlp", "block_output"],
                "dtype": "float16 temporary in RAM only",
                "temporary_activation_files_removed": False,
                "temporary_activation_files_created": False,
                "recovery": "replay prompt-manifest.csv with pinned model and runner",
            },
            "peak_allocated_gpu_bytes": int(torch.cuda.max_memory_allocated()),
        },
    )
    write_json(
        temporary / "probe-artifact-metadata.json",
        {
            "model": MODEL_ID, "model_revision": MODEL_REVISION,
            "layers": extraction_layers, "activation_sites": ["pre_mlp", "block_output"],
            "prompt_template_commit": UPSTREAM_COMMIT, "seed": 123,
            "feature_dtype": "float32 during fitting", "saved_coefficient_dtype": "float32",
            "solvers": {
                "cuml_exact_prior": {"C": 0.005, "max_iter": 5000, "linesearch_max_iter": 100, "scaled": False},
                "sklearn_standardized": {"C": 0.005, "max_iter": 5000, "solver": "lbfgs", "scaled": True},
            },
            "labels": {name: roles for name, roles, *_ in fit_specs},
            "subset_notice": "Coefficients are subset-derived compact replacements, not the lost full-corpus probes.",
        },
    )
    temporary.rename(final)
    print(json.dumps({"stage": "gate-3-role-probes", "status": "complete", "decision": decision, "path": str(final)}, sort_keys=True))
    if decision.startswith("stop_") or expansion_required:
        raise RuntimeError(f"Gate 3 stop condition: decision={decision}, expansion_required={expansion_required}")


def gate4_dir(run_dir: Path) -> Path:
    return run_dir / "gate-4-assistant-axis"


def read_jsonl_gz(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def append_jsonl_gz(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "at", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def load_gate4_manifest(run_dir: Path) -> pd.DataFrame:
    path = gate4_dir(run_dir) / "prepare-personas" / "persona-manifest.csv"
    if not path.is_file():
        raise RuntimeError("Run prepare-personas before Gate 4 generation")
    return pd.read_csv(path)


def parse_question_ids(value: str) -> list[int]:
    question_ids = [int(item.strip()) for item in value.split(",") if item.strip()]
    if len(question_ids) < 2 or len(question_ids) != len(set(question_ids)):
        raise argparse.ArgumentTypeError("Question IDs must contain at least two unique integers")
    return sorted(question_ids)


def command_prepare_personas(args: argparse.Namespace) -> None:
    root = gate4_dir(args.run_dir)
    root.mkdir(parents=True, exist_ok=True)
    temporary, final = atomic_stage_dir(root, "prepare-personas")
    if run_text(["git", "-C", str(ASSISTANT_AXIS_ROOT), "rev-parse", "HEAD"]) != ASSISTANT_AXIS_COMMIT:
        raise RuntimeError("Official Assistant Axis commit mismatch")
    roles_dir = ASSISTANT_AXIS_ROOT / "data/roles/instructions"
    question_path = ASSISTANT_AXIS_ROOT / "data/extraction_questions.jsonl"
    questions = {
        int(row["id"]): row["question"]
        for row in [json.loads(line) for line in question_path.read_text(encoding="utf-8").splitlines()]
    }
    question_ids = getattr(args, "question_ids", None) or QUESTION_IDS
    missing_questions = sorted(set(question_ids) - set(questions))
    if missing_questions:
        raise RuntimeError(f"Unknown official extraction question IDs: {missing_questions}")
    rows = []
    file_records = []
    conditions = [("default_assistant", "assistant", "default", False)] + [
        (persona, persona, PERSONA_FAMILIES[persona], True) for persona in PERSONA_FAMILIES
    ]
    for persona, role_file_name, family, is_persona in conditions:
        role_path = roles_dir / f"{role_file_name}.json"
        role_data = json.loads(role_path.read_text(encoding="utf-8"))
        instruction = role_data["instruction"][0]["pos"]
        file_records.append(
            {
                "role": persona,
                "path": str(role_path.relative_to(ASSISTANT_AXIS_ROOT)),
                "sha256": sha256_file(role_path),
                "instruction_sha256": sha256_bytes(instruction.encode()),
                "eval_prompt_sha256": sha256_bytes(role_data["eval_prompt"].encode()),
            }
        )
        for question_id in question_ids:
            response_id = f"{persona}__q{question_id}"
            messages = [
                {"role": "system", "content": instruction},
                {"role": "user", "content": questions[question_id]},
            ]
            rows.append(
                {
                    "response_id": response_id,
                    "persona": persona,
                    "role_file": role_file_name,
                    "family": family,
                    "is_persona": is_persona,
                    "question_id": question_id,
                    "question": questions[question_id],
                    "instruction_index": 0,
                    "system_prompt": instruction,
                    "messages_sha256": sha256_bytes(
                        json.dumps(messages, sort_keys=True, separators=(",", ":")).encode()
                    ),
                    "micro_pilot": question_id == 0 and (persona in MICRO_PERSONAS or persona == "default_assistant"),
                }
            )
    frame = pd.DataFrame(rows)
    expected_responses = 13 * len(question_ids)
    if len(frame) != expected_responses or int(frame.micro_pilot.sum()) != 7:
        raise AssertionError("Unexpected 12-persona Gate 4 panel size")
    frame.to_csv(temporary / "persona-manifest.csv", index=False)
    write_json(
        temporary / "upstream-provenance.json",
        {
            "repository": "https://github.com/safety-research/assistant-axis.git",
            "commit": ASSISTANT_AXIS_COMMIT,
            "root": str(ASSISTANT_AXIS_ROOT),
            "question_file": {
                "path": str(question_path.relative_to(ASSISTANT_AXIS_ROOT)),
                "sha256": sha256_file(question_path),
                "selected_ids": question_ids,
            },
            "role_files": file_records,
        },
    )
    write_json(
        temporary / "generation-settings.json",
        {
            "model": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "persona_count": 12,
            "default_condition_count": 1,
            "questions": question_ids,
            "target_generation_budget": expected_responses,
            "micro_generation_count": 7,
            "completion_generation_count": expected_responses - 7,
            "instruction_variant": 0,
            "max_new_tokens": 256,
            "do_sample": False,
            "temperature": None,
            "top_p": None,
            "seed": 123,
            "reasoning_effort": "low",
            "judge_model": JUDGE_MODEL,
            "judge_threshold": 3,
            "user_override": (
                "On 2026-08-27 the user explicitly authorized Gate 4 after review of the "
                "Gate 3 numerical stop, with a limited persona panel, then explicitly "
                "authorized adding more questions after reviewing the two-question result."
            ),
        },
    )
    write_json(
        temporary / "user-override.json",
        {
            "authorized_at": utc_now(),
            "scope": "Gate 4 question expansion only",
            "persona_count": 12,
            "question_ids": question_ids,
            "note": "Gate 5 remains unauthorized after the Gate 3 stop.",
        },
    )
    temporary.rename(final)
    print(json.dumps({"stage": "prepare-personas", "status": "complete", "personas": 12, "questions": question_ids, "responses": expected_responses}, sort_keys=True))


def command_seed_question_expansion(args: argparse.Namespace) -> None:
    source_root = gate4_dir(args.source_run_dir)
    target_root = gate4_dir(args.run_dir)
    manifest = load_gate4_manifest(args.run_dir)
    if sorted(manifest.question_id.unique().tolist()) != QUESTION_EXPANSION_IDS:
        raise RuntimeError("Target manifest is not the preregistered six-question expansion")
    for gate_name in ["gate-1-environment", "gate-2-hook-smoke", "gate-3-role-probes"]:
        source = args.source_run_dir / gate_name
        target = args.run_dir / gate_name
        if target.exists():
            raise FileExistsError(f"Refusing to overwrite seeded gate: {target}")
        shutil.copytree(source, target)
    copied = []
    for relative in [
        "responses.jsonl.gz",
        "judge-raw.jsonl.gz",
        "generation-micro-summary.json",
        "micro-hook-validation.json",
        "micro-pilot-decision.json",
    ]:
        source = source_root / relative
        target = target_root / relative
        if target.exists():
            raise FileExistsError(f"Refusing to overwrite expansion seed: {target}")
        shutil.copy2(source, target)
        copied.append(
            {
                "path": relative,
                "source_sha256": sha256_file(source),
                "target_sha256": sha256_file(target),
            }
        )
    responses = read_jsonl_gz(target_root / "responses.jsonl.gz")
    if len(responses) != 26 or sorted({int(row["question_id"]) for row in responses}) != QUESTION_IDS:
        raise RuntimeError("Source run does not contain the exact completed two-question pilot")
    if not {row["response_id"] for row in responses}.issubset(set(manifest.response_id)):
        raise RuntimeError("Seeded responses do not match the expansion manifest")
    if any(row["source_sha256"] != row["target_sha256"] for row in copied):
        raise AssertionError("A seeded expansion artifact changed during copy")
    write_json(
        target_root / "question-expansion-reuse-provenance.json",
        {
            "created_at": utc_now(),
            "source_run": str(args.source_run_dir),
            "source_questions": QUESTION_IDS,
            "target_questions": QUESTION_EXPANSION_IDS,
            "reused_generation_count": len(responses),
            "new_generation_count": 13 * len(QUESTION_EXPANSION_IDS) - len(responses),
            "copied_artifacts": copied,
        },
    )
    print(json.dumps({"stage": "seed-question-expansion", "status": "complete", "reused": len(responses), "new": 52}, sort_keys=True))


def parse_final_response(tokenizer: Any, generated_ids: list[int]) -> tuple[str, str]:
    raw = tokenizer.decode(generated_ids, skip_special_tokens=False)
    marker = "<|channel|>final<|message|>"
    if marker in raw:
        final = raw.rsplit(marker, 1)[1]
        for end_marker in ["<|return|>", "<|end|>"]:
            if end_marker in final:
                final = final.split(end_marker, 1)[0]
        return final.strip(), raw
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip(), raw


def response_token_indices(tokenizer: Any, token_ids: list[int]) -> list[int]:
    from demo.simple_test_helpers import label_gptoss_content_roles

    frame = pd.DataFrame(
        {
            "prompt_ix": 0,
            "token_ix": np.arange(len(token_ids), dtype=np.int64),
            "token": [tokenizer.decode([token_id], skip_special_tokens=False) for token_id in token_ids],
        }
    )
    labeled = label_gptoss_content_roles(frame)
    return labeled.loc[
        labeled.is_content & (labeled.role == "assistant"), "token_ix"
    ].astype(int).tolist()


def validate_micro_captures(
    model: Any,
    tokenizer: Any,
    generated_rows: list[dict[str, Any]],
    output_path: Path,
) -> None:
    records = []
    for row in generated_rows:
        full_ids = row["full_token_ids"]
        response_indices = response_token_indices(tokenizer, full_ids)
        if not response_indices:
            raise AssertionError(f"No final response tokens for {row['response_id']}")
        inputs = {
            "input_ids": torch.tensor([full_ids], dtype=torch.long, device=model.device),
            "attention_mask": torch.ones((1, len(full_ids)), dtype=torch.long, device=model.device),
        }
        outputs, captured, _ = capture_forward(model, inputs, ("pre_mlp", "block_output"), [12, 16])
        layer_records = {}
        for layer in [12, 16]:
            pre = captured["pre_mlp"][layer][0, response_indices].float()
            block = captured["block_output"][layer][0, response_indices].float()
            layer_records[str(layer)] = {
                "shape_pre_mlp": list(pre.shape),
                "shape_block_output": list(block.shape),
                "finite": bool(torch.isfinite(pre).all() and torch.isfinite(block).all()),
                "sites_bit_identical": bool(torch.equal(pre, block)),
                "mean_cosine": float(torch.nn.functional.cosine_similarity(pre.mean(0), block.mean(0), dim=0)),
            }
        records.append(
            {
                "response_id": row["response_id"],
                "response_token_count": len(response_indices),
                "layers": layer_records,
            }
        )
        del outputs, captured
    if not all(
        item["finite"] and not item["sites_bit_identical"]
        for record in records for item in record["layers"].values()
    ):
        raise AssertionError("Micro-pilot activation capture validation failed")
    write_json(output_path, {"validated_at": utc_now(), "responses": records})


def command_generate_personas(args: argparse.Namespace) -> None:
    root = gate4_dir(args.run_dir)
    manifest = load_gate4_manifest(args.run_dir)
    responses_path = root / "responses.jsonl.gz"
    existing_rows = read_jsonl_gz(responses_path)
    existing = {row["response_id"]: row for row in existing_rows}
    if len(existing) != len(existing_rows):
        raise AssertionError("Duplicate response IDs in saved generation ledger")
    if args.scope == "micro":
        targets = manifest[manifest.micro_pilot.astype(str).str.lower().isin(["true", "1"])]
    else:
        if not (root / "micro-pilot-decision.json").is_file():
            raise RuntimeError("Micro-pilot must pass before completing the panel")
        decision = json.loads((root / "micro-pilot-decision.json").read_text())
        if not decision.get("pass"):
            raise RuntimeError("Micro-pilot did not pass")
        targets = manifest
    targets = targets[~targets.response_id.isin(existing)].copy()
    expected_new = len(targets)
    if args.scope == "micro" and len(existing) == 0 and len(targets) != 7:
        raise AssertionError("Micro-pilot must contain exactly seven new generations")
    if args.scope == "complete" and len(existing) == 7 and len(targets) != len(manifest) - 7:
        raise AssertionError("Complete panel size does not match its prepared manifest")
    if targets.empty:
        print(json.dumps({"stage": "generate", "scope": args.scope, "status": "already-complete"}, sort_keys=True))
        return

    sys.path.insert(0, str(ASSISTANT_AXIS_ROOT))
    from assistant_axis.generation import format_conversation

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, cache_dir=HF_HOME, local_files_only=True,
        add_eos_token=False, add_bos_token=False, padding_side="left",
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, cache_dir=HF_HOME, local_files_only=True,
        attn_implementation="kernels-community/vllm-flash-attn3",
    ).to("cuda:0").eval()
    model.set_experts_implementation("eager")
    patch_pinned_model_api(model)
    torch.manual_seed(123)
    torch.cuda.manual_seed_all(123)

    conversations = []
    prompt_texts = []
    prompt_ids = []
    for row in targets.itertuples(index=False):
        conversation = format_conversation(row.system_prompt, row.question, tokenizer)
        prompt = tokenizer.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=True,
            reasoning_effort="low",
        )
        ids = tokenizer(prompt, add_special_tokens=False).input_ids
        conversations.append(conversation)
        prompt_texts.append(prompt)
        prompt_ids.append(ids)
    encoded = tokenizer(
        prompt_texts, add_special_tokens=False, padding=True, return_tensors="pt"
    ).to(model.device)
    torch.cuda.reset_peak_memory_stats()
    started = utc_now()
    with torch.inference_mode():
        generated = model.generate(
            **encoded,
            max_new_tokens=256,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=True,
        )
    sequence_tail = generated.sequences[:, encoded.input_ids.shape[1]:].detach().cpu().tolist()
    new_rows = []
    for target, conversation, prompt, ids, tail in zip(
        targets.itertuples(index=False), conversations, prompt_texts, prompt_ids, sequence_tail, strict=True
    ):
        trimmed = []
        finish_reason = "length"
        for token_id in tail:
            if token_id == tokenizer.pad_token_id and trimmed:
                break
            trimmed.append(int(token_id))
            if token_id == tokenizer.eos_token_id:
                finish_reason = "eos"
                break
        response, raw_decoded = parse_final_response(tokenizer, trimmed)
        full_ids = list(map(int, ids)) + trimmed
        response_indices = response_token_indices(tokenizer, full_ids)
        record = {
            "response_id": target.response_id,
            "persona": target.persona,
            "family": target.family,
            "is_persona": bool(target.is_persona),
            "question_id": int(target.question_id),
            "question": target.question,
            "system_prompt": target.system_prompt,
            "conversation": conversation,
            "prompt": prompt,
            "prompt_token_ids": list(map(int, ids)),
            "generated_token_ids": trimmed,
            "full_token_ids": full_ids,
            "response": response,
            "raw_generated_decoded": raw_decoded,
            "finish_reason": finish_reason,
            "generated_token_count": len(trimmed),
            "response_token_count": len(response_indices),
            "generation_started_at": started,
            "generation_completed_at": utc_now(),
            "generation_settings": {
                "max_new_tokens": 256,
                "do_sample": False,
                "seed": 123,
                "reasoning_effort": "low",
            },
        }
        if not response or not response_indices or len(trimmed) > 256:
            raise AssertionError(f"Invalid generated response boundary for {target.response_id}")
        new_rows.append(record)
    append_jsonl_gz(responses_path, new_rows)
    all_rows = read_jsonl_gz(responses_path)
    if len({row["response_id"] for row in all_rows}) != len(all_rows):
        raise AssertionError("Generation ledger contains duplicate response IDs after append")
    if args.scope == "micro":
        validate_micro_captures(
            model, tokenizer, new_rows, root / "micro-hook-validation.json"
        )
    write_json(
        root / f"generation-{args.scope}-summary.json",
        {
            "completed_at": utc_now(),
            "scope": args.scope,
            "new_generations": len(new_rows),
            "total_saved_generations": len(all_rows),
            "expected_new_generations": expected_new,
            "peak_allocated_gpu_bytes": int(torch.cuda.max_memory_allocated()),
            "finish_reasons": pd.Series([row["finish_reason"] for row in new_rows]).value_counts().to_dict(),
            "generated_token_counts": [row["generated_token_count"] for row in new_rows],
            "response_token_counts": [row["response_token_count"] for row in new_rows],
        },
    )
    print(json.dumps({"stage": "generate", "scope": args.scope, "new": len(new_rows), "total": len(all_rows)}, sort_keys=True))


def parse_official_judge_score(text_value: str | None) -> int | None:
    if not text_value:
        return None
    import re

    numbers = re.findall(r"\b(\d+)\b", text_value.strip())
    if not numbers:
        return None
    score = int(numbers[0])
    return score if 0 <= score <= 3 else None


def command_score_personas(args: argparse.Namespace) -> None:
    from openai import OpenAI

    root = gate4_dir(args.run_dir)
    manifest = load_gate4_manifest(args.run_dir)
    response_rows = read_jsonl_gz(root / "responses.jsonl.gz")
    responses = {row["response_id"]: row for row in response_rows}
    if args.scope == "micro":
        target_ids = set(
            manifest.loc[
                manifest.micro_pilot.astype(str).str.lower().isin(["true", "1"])
                & manifest.is_persona.astype(str).str.lower().isin(["true", "1"]),
                "response_id",
            ]
        )
        if len(target_ids) != 6 or not target_ids.issubset(responses):
            raise RuntimeError("Six persona micro-pilot responses must exist before scoring")
    else:
        target_ids = set(
            manifest.loc[
                manifest.is_persona.astype(str).str.lower().isin(["true", "1"]), "response_id"
            ]
        )
        expected_persona_responses = int(
            manifest.is_persona.astype(str).str.lower().isin(["true", "1"]).sum()
        )
        if len(responses) != len(manifest) or len(target_ids) != expected_persona_responses:
            raise RuntimeError("The complete prepared response panel must exist before final scoring")
    judge_path = root / "judge-raw.jsonl.gz"
    existing_rows = read_jsonl_gz(judge_path)
    existing = {row["response_id"]: row for row in existing_rows}
    to_score_set = target_ids - set(existing)
    if args.retry_infrastructure_failures:
        retryable = {
            response_id for response_id in target_ids & set(existing)
            if existing[response_id].get("parsed_score") is None
            and existing[response_id].get("error")
        }
        to_score_set |= retryable
    to_score = sorted(to_score_set)
    roles_dir = ASSISTANT_AXIS_ROOT / "data/roles/instructions"
    openai_key = os.getenv("OPENAI_API_KEY", "").strip()
    openrouter_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if openai_key:
        client = OpenAI(api_key=openai_key)
        routed_model = JUDGE_MODEL
        endpoint = "https://api.openai.com/v1"
    elif openrouter_key:
        client = OpenAI(api_key=openrouter_key, base_url="https://openrouter.ai/api/v1")
        routed_model = f"openai/{JUDGE_MODEL}"
        endpoint = "https://openrouter.ai/api/v1"
    else:
        raise RuntimeError("Neither OPENAI_API_KEY nor OPENROUTER_API_KEY is configured")
    new_rows = []
    for response_id in to_score:
        response = responses[response_id]
        role_data = json.loads(
            (roles_dir / f"{response['persona']}.json").read_text(encoding="utf-8")
        )
        prompt = role_data["eval_prompt"].format(
            question=response["question"], answer=response["response"]
        )
        started = utc_now()
        raw_text = None
        error = None
        usage = None
        try:
            result = client.chat.completions.create(
                model=routed_model,
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=10,
                temperature=1,
            )
            raw_text = result.choices[0].message.content if result.choices else None
            if result.usage is not None:
                usage = {
                    "prompt_tokens": result.usage.prompt_tokens,
                    "completion_tokens": result.usage.completion_tokens,
                    "total_tokens": result.usage.total_tokens,
                }
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        new_rows.append(
            {
                "response_id": response_id,
                "persona": response["persona"],
                "family": response["family"],
                "question_id": response["question_id"],
                "judge_model": JUDGE_MODEL,
                "routed_model": routed_model,
                "endpoint": endpoint,
                "judge_prompt_sha256": sha256_bytes(prompt.encode()),
                "raw_output": raw_text,
                "parsed_score": parse_official_judge_score(raw_text),
                "error": error,
                "usage": usage,
                "started_at": started,
                "completed_at": utc_now(),
                "attempt": 1 + sum(row["response_id"] == response_id for row in existing_rows),
            }
        )
    append_jsonl_gz(judge_path, new_rows)
    judge_rows = read_jsonl_gz(judge_path)
    judge = {row["response_id"]: row for row in judge_rows}
    relevant = [judge[response_id] for response_id in sorted(target_ids) if response_id in judge]
    score_frame = pd.DataFrame(
        [
            {
                "response_id": row["response_id"],
                "persona": row["persona"],
                "family": row["family"],
                "question_id": row["question_id"],
                "score": row["parsed_score"],
                "parsed": row["parsed_score"] is not None,
                "included": row["parsed_score"] == 3,
                "error": row["error"],
            }
            for row in judge.values()
        ]
    ).sort_values("response_id")
    score_frame.to_csv(root / "judge-scores.csv", index=False)
    inclusion_rows = []
    for row in response_rows:
        score = judge.get(row["response_id"], {}).get("parsed_score")
        inclusion_rows.append(
            {
                "response_id": row["response_id"],
                "persona": row["persona"],
                "family": row["family"],
                "is_persona": row["is_persona"],
                "question_id": row["question_id"],
                "judge_score": score,
                "included": not row["is_persona"] or score == 3,
                "inclusion_rule": "all_default" if not row["is_persona"] else "official_score_equals_3",
            }
        )
    pd.DataFrame(inclusion_rows).sort_values("response_id").to_csv(
        root / "inclusion-manifest.csv", index=False
    )
    parse_failures = sum(row["parsed_score"] is None for row in relevant)
    accepted_personas = sorted(
        {row["persona"] for row in relevant if row["parsed_score"] == 3}
    )
    if args.scope == "micro":
        formatting = json.loads((root / "micro-hook-validation.json").read_text())
        capture_ok = all(
            item["finite"] and not item["sites_bit_identical"]
            for record in formatting["responses"] for item in record["layers"].values()
        )
        passed = parse_failures == 0 and len(accepted_personas) >= 3 and capture_ok
        write_json(
            root / "micro-pilot-decision.json",
            {
                "evaluated_at": utc_now(),
                "pass": passed,
                "persona_responses": 6,
                "accepted_personas": accepted_personas,
                "accepted_count": len(accepted_personas),
                "parse_failures": parse_failures,
                "activation_capture_ok": capture_ok,
                "next_generation_count": 19 if passed else 0,
            },
        )
        if not passed:
            raise RuntimeError("Gate 4 micro-pilot failed; do not generate remaining responses")
    else:
        persona_with_accepted = score_frame[score_frame.included].persona.nunique()
        parse_failure_rate = parse_failures / len(relevant) if relevant else 1.0
        if parse_failure_rate > 0.02 or persona_with_accepted < 6:
            raise RuntimeError(
                f"Gate 4 judge stop: parse_failure_rate={parse_failure_rate}, accepted_personas={persona_with_accepted}"
            )
        review = score_frame[(score_frame.score != 3) | score_frame.score.isna()].copy()
        for family in sorted(score_frame.family.unique()):
            candidates = score_frame[(score_frame.family == family) & (score_frame.score == 3)]
            if not candidates.empty:
                review = pd.concat([review, candidates.head(1)], ignore_index=True)
        review = review.drop_duplicates("response_id").sort_values("response_id")
        review["response"] = review.response_id.map(lambda key: responses[key]["response"])
        review["review_status"] = "pending_manual_review"
        review.to_csv(root / "manual-review.csv", index=False)
        write_json(
            root / "judge-summary.json",
            {
                "completed_at": utc_now(),
                "judge_model": JUDGE_MODEL,
                "routed_model": routed_model,
                "endpoint": endpoint,
                "persona_judgments": len(relevant),
                "new_judgments_this_command": len(new_rows),
                "parse_failures": parse_failures,
                "parse_failure_rate": parse_failure_rate,
                "accepted_response_count": int(score_frame.included.sum()),
                "personas_with_accepted_response": int(persona_with_accepted),
                "score_counts": {
                    str(key): int(value) for key, value in score_frame.score.value_counts(dropna=False).items()
                },
                "manual_review_count": len(review),
            },
        )
    print(json.dumps({"stage": "score", "scope": args.scope, "new": len(new_rows), "ledger_attempts": len(judge_rows), "unique_responses": len(judge), "accepted_personas": len(accepted_personas)}, sort_keys=True))


def command_extract_personas(args: argparse.Namespace) -> None:
    root = gate4_dir(args.run_dir)
    review_path = root / "manual-review.csv"
    if not review_path.is_file():
        raise RuntimeError("Complete judging and manual review selection before extraction")
    review = pd.read_csv(review_path)
    if review.review_status.astype(str).str.startswith("pending").any():
        raise RuntimeError("Manual review must be completed before extraction")
    responses = {row["response_id"]: row for row in read_jsonl_gz(root / "responses.jsonl.gz")}
    inclusion = pd.read_csv(root / "inclusion-manifest.csv")
    included = inclusion[inclusion.included.astype(str).str.lower().isin(["true", "1"])].copy()
    if len(responses) != len(load_gate4_manifest(args.run_dir)) or included.empty:
        raise RuntimeError("Complete response panel and inclusion manifest are required")
    ids = included.response_id.tolist()
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, cache_dir=HF_HOME, local_files_only=True,
        add_eos_token=False, add_bos_token=False, padding_side="left",
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, cache_dir=HF_HOME, local_files_only=True,
        attn_implementation="kernels-community/vllm-flash-attn3",
    ).to("cuda:0").eval()
    model.set_experts_implementation("eager")
    patch_pinned_model_api(model)
    layers = list(range(8, 21))
    vectors: dict[str, list[torch.Tensor]] = {"pre_mlp": [], "block_output": []}
    metadata_rows = []
    torch.cuda.reset_peak_memory_stats()
    for start in range(0, len(ids), 8):
        batch_ids = ids[start:start + 8]
        sequences = [responses[response_id]["full_token_ids"] for response_id in batch_ids]
        response_indices = [response_token_indices(tokenizer, sequence) for sequence in sequences]
        if any(not indices for indices in response_indices):
            raise AssertionError("Missing response token span during activation replay")
        max_length = max(map(len, sequences))
        padded = []
        masks = []
        offsets = []
        for sequence in sequences:
            offset = max_length - len(sequence)
            offsets.append(offset)
            padded.append([tokenizer.pad_token_id] * offset + sequence)
            masks.append([0] * offset + [1] * len(sequence))
        inputs = {
            "input_ids": torch.tensor(padded, dtype=torch.long, device=model.device),
            "attention_mask": torch.tensor(masks, dtype=torch.long, device=model.device),
        }
        outputs, captured, _ = capture_forward(
            model, inputs, ("pre_mlp", "block_output"), layers
        )
        for batch_ix, response_id in enumerate(batch_ids):
            selected = [offsets[batch_ix] + index for index in response_indices[batch_ix]]
            for site in vectors:
                per_layer = []
                for layer in layers:
                    token_acts = captured[site][layer][batch_ix, selected].float()
                    if not torch.isfinite(token_acts).all():
                        raise AssertionError(f"Non-finite {site} activations: {response_id}, layer {layer}")
                    per_layer.append(token_acts.mean(dim=0, dtype=torch.float32))
                vectors[site].append(torch.stack(per_layer))
            metadata_rows.append(
                {
                    "response_id": response_id,
                    "persona": responses[response_id]["persona"],
                    "family": responses[response_id]["family"],
                    "is_persona": responses[response_id]["is_persona"],
                    "question_id": responses[response_id]["question_id"],
                    "judge_score": included.loc[included.response_id == response_id, "judge_score"].iloc[0],
                    "response_token_count": len(selected),
                    "full_conversation_token_count": len(sequences[batch_ix]),
                }
            )
        del outputs, captured, inputs
        torch.cuda.empty_cache()
    for site, values in vectors.items():
        tensor = torch.stack(values).to(torch.float32)
        path = root / "activations" / f"{site}-response-means.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"response_ids": ids, "layers": layers, "dtype": "float32", "vectors": tensor},
            path,
        )
        reloaded = torch.load(path, map_location="cpu", weights_only=True)
        if reloaded["response_ids"] != ids or tuple(reloaded["vectors"].shape) != tuple(tensor.shape):
            raise AssertionError(f"Activation artifact reload failed for {site}")
        if not torch.isfinite(reloaded["vectors"]).all():
            raise AssertionError(f"Non-finite saved activation artifact for {site}")
    pd.DataFrame(metadata_rows).to_csv(root / "activation-manifest.csv", index=False)
    write_json(
        root / "extraction-summary.json",
        {
            "completed_at": utc_now(),
            "included_responses": len(ids),
            "included_persona_responses": int(sum(responses[key]["is_persona"] for key in ids)),
            "included_default_responses": int(sum(not responses[key]["is_persona"] for key in ids)),
            "layers": layers,
            "activation_sites": ["pre_mlp", "block_output"],
            "accumulation_dtype": "float32",
            "saved_dtype": "float32",
            "token_rule": "final-channel assistant content tokens only; reasoning/analysis excluded",
            "peak_allocated_gpu_bytes": int(torch.cuda.max_memory_allocated()),
        },
    )
    print(json.dumps({"stage": "extract", "status": "complete", "responses": len(ids), "layers": layers}, sort_keys=True))


def vector_cosine(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    denominator = np.linalg.norm(left) * np.linalg.norm(right)
    return float(left @ right / denominator) if denominator > 0 else float("nan")


def orthonormal_row_basis(matrix: np.ndarray, tolerance: float = 1e-8) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.size == 0:
        return np.empty((0, matrix.shape[-1] if matrix.ndim == 2 else 0))
    _, singular, vh = np.linalg.svd(matrix, full_matrices=False)
    rank = int(np.sum(singular > tolerance * singular[0])) if singular.size and singular[0] > 0 else 0
    return vh[:rank]


def fast_pc1(matrix: np.ndarray) -> np.ndarray:
    centered = np.asarray(matrix, dtype=np.float64) - np.asarray(matrix, dtype=np.float64).mean(axis=0)
    gram = centered @ centered.T
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    direction = eigenvectors[:, int(np.argmax(eigenvalues))] @ centered
    norm = np.linalg.norm(direction)
    return direction / norm if norm > 0 else np.zeros(centered.shape[1], dtype=np.float64)


def command_analyze_personas(args: argparse.Namespace) -> None:
    root = gate4_dir(args.run_dir)
    metadata = pd.read_csv(root / "activation-manifest.csv")
    question_ids = sorted(int(value) for value in metadata.question_id.unique())
    inclusion = pd.read_csv(root / "inclusion-manifest.csv")
    judge_summary = json.loads((root / "judge-summary.json").read_text())
    responses = {row["response_id"]: row for row in read_jsonl_gz(root / "responses.jsonl.gz")}
    role_probe_path = args.run_dir / "gate-3-role-probes" / "probe-coefficients.npz"
    role_probe = np.load(role_probe_path)
    analysis_dir = root / "analysis"
    if analysis_dir.exists():
        raise FileExistsError(f"Refusing to overwrite analysis: {analysis_dir}")
    temporary = root / ".analysis.tmp"
    if temporary.exists():
        raise FileExistsError(f"Incomplete analysis exists: {temporary}")
    temporary.mkdir()

    pca_rows = []
    heldout_rows = []
    stability_rows = []
    projection_rows = []
    bootstrap_rows = []
    role_cosine_rows = []
    angle_rows = []
    site_decisions: dict[str, list[dict[str, Any]]] = {"pre_mlp": [], "block_output": []}
    saved_axes = {}
    saved_role_vectors = {}
    saved_relative_vectors = {}
    rng = np.random.default_rng(123)

    for site in ["pre_mlp", "block_output"]:
        artifact = torch.load(
            root / "activations" / f"{site}-response-means.pt",
            map_location="cpu", weights_only=True,
        )
        ids = artifact["response_ids"]
        layers = artifact["layers"]
        values = artifact["vectors"].float().numpy()
        id_to_ix = {response_id: ix for ix, response_id in enumerate(ids)}
        default_ids = metadata.loc[~metadata.is_persona.astype(bool), "response_id"].tolist()
        persona_names = sorted(metadata.loc[metadata.is_persona.astype(bool), "persona"].unique())
        if len(default_ids) != len(question_ids) or len(question_ids) < 2 or len(persona_names) < 6:
            raise RuntimeError("Insufficient default or accepted persona activations")
        default_mean = values[[id_to_ix[key] for key in default_ids]].mean(axis=0)
        role_vector_map = {}
        for persona in persona_names:
            role_ids = metadata.loc[metadata.persona == persona, "response_id"].tolist()
            role_vector_map[persona] = values[[id_to_ix[key] for key in role_ids]].mean(axis=0)
        role_matrix_all = np.stack([role_vector_map[name] for name in persona_names])
        role_mean = role_matrix_all.mean(axis=0)
        axis_all = default_mean - role_mean
        relative_all = role_matrix_all - default_mean[None, :, :]
        saved_axes[site] = torch.from_numpy(axis_all.astype(np.float32))
        saved_role_vectors[site] = {
            "persona_names": persona_names,
            "vectors": torch.from_numpy(role_matrix_all.astype(np.float32)),
        }
        saved_relative_vectors[site] = {
            "persona_names": persona_names,
            "vectors": torch.from_numpy(relative_all.astype(np.float32)),
            "definition": "role_vector - default_vector",
        }

        for layer_position, layer in enumerate(layers):
            role_matrix = role_matrix_all[:, layer_position, :].astype(np.float64)
            axis = axis_all[layer_position].astype(np.float64)
            default_vector = default_mean[layer_position].astype(np.float64)
            axis_unit = axis / np.linalg.norm(axis)
            pca = PCA().fit(role_matrix)
            pc1 = pca.components_[0].copy()
            if pc1 @ axis < 0:
                pc1 *= -1
            axis_pc1 = vector_cosine(axis, pc1)
            ratios = pca.explained_variance_ratio_
            pca_rows.append(
                {
                    "activation_site": site,
                    "layer": layer,
                    "n_personas": len(persona_names),
                    "axis_pc1_cosine": axis_pc1,
                    "pc1_explained_variance_ratio": float(ratios[0]),
                    "pc2_explained_variance_ratio": float(ratios[1]) if len(ratios) > 1 else np.nan,
                    "pc3_explained_variance_ratio": float(ratios[2]) if len(ratios) > 2 else np.nan,
                    "axis_norm": float(np.linalg.norm(axis)),
                    "pc1_alignment": "strong" if abs(axis_pc1) >= 0.8 else "moderate" if abs(axis_pc1) >= 0.5 else "weak",
                }
            )
            for persona, vector in zip(persona_names, role_matrix, strict=True):
                projection_rows.append(
                    {
                        "activation_site": site,
                        "layer": layer,
                        "persona": persona,
                        "family": PERSONA_FAMILIES[persona],
                        "condition": "persona",
                        "axis_projection": float(vector @ axis_unit),
                    }
                )
            projection_rows.append(
                {
                    "activation_site": site,
                    "layer": layer,
                    "persona": "default_assistant",
                    "family": "default",
                    "condition": "default",
                    "axis_projection": float(default_vector @ axis_unit),
                }
            )

            lopo_cosines = []
            for persona_ix, persona in enumerate(persona_names):
                reduced_axis = default_vector - np.delete(role_matrix, persona_ix, axis=0).mean(axis=0)
                cosine = vector_cosine(axis, reduced_axis)
                lopo_cosines.append(cosine)
                stability_rows.append(
                    {
                        "activation_site": site,
                        "layer": layer,
                        "metric": "leave_one_persona_out_cosine",
                        "held_out": persona,
                        "value": cosine,
                    }
                )

            question_axes = {}
            for question_id in question_ids:
                question_default_ids = metadata.loc[
                    (~metadata.is_persona.astype(bool)) & (metadata.question_id == question_id), "response_id"
                ].tolist()
                question_persona = metadata.loc[
                    metadata.is_persona.astype(bool) & (metadata.question_id == question_id)
                ]
                if question_default_ids and question_persona.persona.nunique() >= 3:
                    question_default = values[id_to_ix[question_default_ids[0]], layer_position]
                    question_role_vectors = []
                    for persona in sorted(question_persona.persona.unique()):
                        response_id = question_persona.loc[question_persona.persona == persona, "response_id"].iloc[0]
                        question_role_vectors.append(values[id_to_ix[response_id], layer_position])
                    question_axes[question_id] = question_default - np.mean(question_role_vectors, axis=0)
            pairwise_question_cosines = []
            leave_one_question_out_cosines = []
            if set(question_axes) == set(question_ids):
                first_half = question_ids[::2]
                second_half = question_ids[1::2]
                first_axis = np.mean([question_axes[question_id] for question_id in first_half], axis=0)
                second_axis = np.mean([question_axes[question_id] for question_id in second_half], axis=0)
                split_cosine = vector_cosine(first_axis, second_axis)
                for left_position, left_question in enumerate(question_ids):
                    for right_question in question_ids[left_position + 1:]:
                        pairwise_cosine = vector_cosine(question_axes[left_question], question_axes[right_question])
                        pairwise_question_cosines.append(pairwise_cosine)
                        stability_rows.append(
                            {
                                "activation_site": site,
                                "layer": layer,
                                "metric": "pairwise_question_cosine",
                                "held_out": f"q{left_question}_vs_q{right_question}",
                                "value": pairwise_cosine,
                            }
                        )
                aggregate_question_axis = np.mean(list(question_axes.values()), axis=0)
                for question_id in question_ids:
                    reduced_question_axis = np.mean(
                        [value for key, value in question_axes.items() if key != question_id], axis=0
                    )
                    loqo_cosine = vector_cosine(aggregate_question_axis, reduced_question_axis)
                    leave_one_question_out_cosines.append(loqo_cosine)
                    stability_rows.append(
                        {
                            "activation_site": site,
                            "layer": layer,
                            "metric": "leave_one_question_out_cosine",
                            "held_out": f"q{question_id}",
                            "value": loqo_cosine,
                        }
                    )
            else:
                first_half = question_ids[::2]
                second_half = question_ids[1::2]
                split_cosine = float("nan")
            stability_rows.append(
                {
                    "activation_site": site,
                    "layer": layer,
                    "metric": "split_half_question_cosine",
                    "held_out": "|".join(f"q{value}" for value in first_half) + "_vs_" + "|".join(f"q{value}" for value in second_half),
                    "value": split_cosine,
                }
            )

            family_names = sorted({PERSONA_FAMILIES[name] for name in persona_names})
            heldout_sets = []
            for offset in [0, 1]:
                heldout_sets.append(
                    {
                        sorted([name for name in persona_names if PERSONA_FAMILIES[name] == family])[offset % len([name for name in persona_names if PERSONA_FAMILIES[name] == family])]
                        for family in family_names
                    }
                )
            fold_metrics = []
            for fold_ix, (train_questions, test_questions, heldout_personas) in enumerate(
                [(first_half, second_half, heldout_sets[0]), (second_half, first_half, heldout_sets[1])]
            ):
                train_default_ids = metadata.loc[
                    (~metadata.is_persona.astype(bool)) & metadata.question_id.isin(train_questions), "response_id"
                ].tolist()
                train_persona_rows = metadata[
                    metadata.is_persona.astype(bool)
                    & metadata.question_id.isin(train_questions)
                    & (~metadata.persona.isin(heldout_personas))
                ]
                test_default_ids = metadata.loc[
                    (~metadata.is_persona.astype(bool)) & metadata.question_id.isin(test_questions), "response_id"
                ].tolist()
                test_persona_rows = metadata[
                    metadata.is_persona.astype(bool)
                    & metadata.question_id.isin(test_questions)
                    & (metadata.persona.isin(heldout_personas))
                ]
                if not train_default_ids or not test_default_ids or len(train_persona_rows) < 3 or test_persona_rows.empty:
                    continue
                train_default = values[
                    [id_to_ix[response_id] for response_id in train_default_ids], layer_position
                ].mean(axis=0)
                train_role = np.stack(
                    [values[id_to_ix[key], layer_position] for key in train_persona_rows.response_id]
                )
                fold_axis = train_default - train_role.mean(axis=0)
                fold_unit = fold_axis / np.linalg.norm(fold_axis)
                train_default_score = float(train_default @ fold_unit)
                train_role_score = float((train_role @ fold_unit).mean())
                threshold = 0.5 * (train_default_score + train_role_score)
                test_vectors = [
                    values[id_to_ix[response_id], layer_position] for response_id in test_default_ids
                ] + [
                    values[id_to_ix[key], layer_position] for key in test_persona_rows.response_id
                ]
                truth = np.asarray([1] * len(test_default_ids) + [0] * len(test_persona_rows), dtype=np.int32)
                scores = np.asarray([vector @ fold_unit for vector in test_vectors])
                pred = (scores >= threshold).astype(np.int32)
                auroc = float(roc_auc_score(truth, scores))
                balanced = float(balanced_accuracy_score(truth, pred))
                fold_metrics.append((auroc, balanced))
                heldout_rows.append(
                    {
                        "activation_site": site,
                        "layer": layer,
                        "fold": fold_ix,
                        "train_questions": "|".join(map(str, train_questions)),
                        "test_questions": "|".join(map(str, test_questions)),
                        "heldout_personas": "|".join(sorted(heldout_personas)),
                        "n_train_personas": train_persona_rows.persona.nunique(),
                        "n_test_personas": test_persona_rows.persona.nunique(),
                        "auroc": auroc,
                        "balanced_accuracy": balanced,
                        "threshold": threshold,
                    }
                )

            boot_axis_pc1 = []
            boot_norm = []
            for _ in range(200):
                sampled_ix = rng.integers(0, len(persona_names), size=len(persona_names))
                sampled = role_matrix[sampled_ix]
                boot_axis = default_vector - sampled.mean(axis=0)
                boot_pc1 = fast_pc1(sampled)
                boot_axis_pc1.append(abs(vector_cosine(boot_axis, boot_pc1)))
                boot_norm.append(float(np.linalg.norm(boot_axis)))
            for metric, samples in [("absolute_axis_pc1_cosine", boot_axis_pc1), ("axis_norm", boot_norm)]:
                bootstrap_rows.append(
                    {
                        "activation_site": site,
                        "layer": layer,
                        "metric": metric,
                        "bootstrap_unit": "persona",
                        "replicates": 200,
                        "estimate": abs(axis_pc1) if metric.startswith("absolute") else float(np.linalg.norm(axis)),
                        "ci_low": float(np.quantile(samples, 0.025)),
                        "ci_high": float(np.quantile(samples, 0.975)),
                    }
                )

            if layer in [12, 16]:
                prefix = f"compact_system_user_cot_assistant__{site}__layer_{layer:02d}__sklearn_standardized"
            elif layer in [8, 10, 14, 18]:
                prefix = f"pilot_binary__{site}__layer_{layer:02d}__sklearn_standardized"
            else:
                prefix = None
            role_basis = np.empty((0, axis.shape[0]))
            if prefix and f"{prefix}__centered_coefficients_raw" in role_probe:
                coefficients = role_probe[f"{prefix}__centered_coefficients_raw"].astype(np.float64)
                labels = role_probe[f"{prefix}__class_labels"].astype(str).tolist()
                for role_label, direction in zip(labels, coefficients, strict=True):
                    role_cosine_rows.append(
                        {
                            "activation_site": site,
                            "layer": layer,
                            "role_probe": role_label,
                            "assistant_axis_cosine": vector_cosine(axis, direction),
                        }
                    )
                if "assistant" in labels:
                    assistant_ix = labels.index("assistant")
                    others = np.delete(coefficients, assistant_ix, axis=0)
                    contrast = coefficients[assistant_ix] - others.mean(axis=0)
                    role_cosine_rows.append(
                        {
                            "activation_site": site,
                            "layer": layer,
                            "role_probe": "assistant_vs_other_roles",
                            "assistant_axis_cosine": vector_cosine(axis, contrast),
                        }
                    )
                role_basis = orthonormal_row_basis(coefficients)
                persona_basis = orthonormal_row_basis(pca.components_[:3])
                singular = np.linalg.svd(persona_basis @ role_basis.T, compute_uv=False) if len(role_basis) else []
                for angle_ix, cosine_value in enumerate(np.clip(singular, -1, 1)):
                    angle_rows.append(
                        {
                            "activation_site": site,
                            "layer": layer,
                            "comparison": "persona_pc1_pc3_vs_role_probe_subspace",
                            "angle_index": angle_ix,
                            "angle_degrees": float(np.degrees(np.arccos(cosine_value))),
                        }
                    )
                projection_norm = float(np.linalg.norm(role_basis @ axis_unit)) if len(role_basis) else 0.0
                angle_rows.append(
                    {
                        "activation_site": site,
                        "layer": layer,
                        "comparison": "assistant_axis_vs_role_probe_subspace",
                        "angle_index": 0,
                        "angle_degrees": float(np.degrees(np.arccos(np.clip(projection_norm, 0, 1)))),
                    }
                )
            mean_auroc = float(np.mean([value[0] for value in fold_metrics])) if fold_metrics else float("nan")
            mean_balanced = float(np.mean([value[1] for value in fold_metrics])) if fold_metrics else float("nan")
            default_projection = float(default_vector @ axis_unit)
            mean_role_projection = float((role_matrix @ axis_unit).mean())
            promising = bool(
                np.isfinite(mean_auroc) and mean_auroc >= 0.80
                and np.isfinite(split_cosine) and split_cosine >= 0.80
                and min(lopo_cosines) >= 0.80
                and default_projection > mean_role_projection
            )
            site_decisions[site].append(
                {
                    "layer": layer,
                    "promising": promising,
                    "heldout_auroc_mean": mean_auroc,
                    "heldout_balanced_accuracy_mean": mean_balanced,
                    "split_half_cosine": split_cosine,
                    "pairwise_question_cosine_min": min(pairwise_question_cosines) if pairwise_question_cosines else float("nan"),
                    "pairwise_question_cosine_median": float(np.median(pairwise_question_cosines)) if pairwise_question_cosines else float("nan"),
                    "leave_one_question_out_min_cosine": min(leave_one_question_out_cosines) if leave_one_question_out_cosines else float("nan"),
                    "leave_one_persona_out_min_cosine": min(lopo_cosines),
                    "default_projection_above_persona_mean": default_projection > mean_role_projection,
                    "axis_pc1_cosine": axis_pc1,
                }
            )

    torch.save(saved_axes, temporary / "axis-by-layer.pt")
    torch.save(saved_role_vectors, temporary / "persona-vectors.pt")
    torch.save(saved_relative_vectors, temporary / "persona-vectors-relative-to-default.pt")
    pd.DataFrame(pca_rows).to_csv(temporary / "pca-metrics.csv", index=False)
    pd.DataFrame(heldout_rows).to_csv(temporary / "heldout-separation.csv", index=False)
    pd.DataFrame(stability_rows).to_csv(temporary / "stability.csv", index=False)
    pd.DataFrame(projection_rows).to_csv(temporary / "projection-distribution.csv", index=False)
    pd.DataFrame(bootstrap_rows).to_csv(temporary / "bootstrap-confidence.csv", index=False)
    pd.DataFrame(role_cosine_rows).to_csv(temporary / "role-axis-cosines.csv", index=False)
    pd.DataFrame(angle_rows).to_csv(temporary / "principal-angles.csv", index=False)
    promising_sites = {
        site: [row["layer"] for row in rows if row["promising"]]
        for site, rows in site_decisions.items()
    }
    if promising_sites["pre_mlp"] and promising_sites["block_output"]:
        recommendation = "dual-site result"
    elif promising_sites["block_output"]:
        recommendation = "block-output path"
    elif promising_sites["pre_mlp"]:
        recommendation = "pre-MLP adaptation"
    else:
        recommendation = "stop"
    write_json(
        temporary / "gate-decision.json",
        {
            "completed_at": utc_now(),
            "pilot_personas": 12,
            "question_ids": question_ids,
            "question_split_halves": [question_ids[::2], question_ids[1::2]],
            "accepted_personas": int(metadata.loc[metadata.is_persona.astype(bool), "persona"].nunique()),
            "accepted_persona_responses": int(metadata.is_persona.astype(bool).sum()),
            "judge_parse_failure_rate": judge_summary["parse_failure_rate"],
            "promising_layers_by_site": promising_sites,
            "per_layer": site_decisions,
            "recommendation": recommendation,
            "pilot_warning": f"PCA and confidence intervals are diagnostics from 12 personas and {len(question_ids)} questions, not paper-quality estimates.",
        },
    )
    temporary.rename(analysis_dir)
    print(json.dumps({"stage": "analyze", "status": "complete", "recommendation": recommendation, "promising_layers": promising_sites}, sort_keys=True))


def command_finalize(args: argparse.Namespace) -> None:
    if not (args.run_dir / "gate-3-role-probes").is_dir():
        raise RuntimeError("Gate 3 diagnostics must exist before finalizing")
    rows = []
    for path in sorted(args.run_dir.rglob("*")):
        if not path.is_file() or path.name == "sha256sums.txt":
            continue
        rows.append(
            {
                "sha256": sha256_file(path),
                "path": str(path.relative_to(args.run_dir)),
                "bytes": path.stat().st_size,
            }
        )
    checksum_path = args.run_dir / "sha256sums.txt"
    temporary = args.run_dir / ".sha256sums.tmp"
    temporary.write_text(
        "".join(f"{row['sha256']}  {row['path']}\n" for row in rows),
        encoding="utf-8",
    )
    temporary.replace(checksum_path)
    gate4_decision_path = args.run_dir / "gate-4-assistant-axis" / "analysis" / "gate-decision.json"
    if gate4_decision_path.is_file():
        gate4_decision = json.loads(gate4_decision_path.read_text())
        gate4_root = gate4_dir(args.run_dir)
        manifest = load_gate4_manifest(args.run_dir)
        question_ids = sorted(int(value) for value in manifest.question_id.unique())
        response_count = len(read_jsonl_gz(gate4_root / "responses.jsonl.gz"))
        judge_summary = json.loads((gate4_root / "judge-summary.json").read_text())
        judge_attempts = read_jsonl_gz(gate4_root / "judge-raw.jsonl.gz")
        failed_attempts = sum(
            bool(row.get("error")) and row.get("parsed_score") is None for row in judge_attempts
        )
        reuse_path = gate4_root / "question-expansion-reuse-provenance.json"
        reuse = json.loads(reuse_path.read_text()) if reuse_path.is_file() else {}
        reused_generations = int(reuse.get("reused_generation_count", 0))
        is_question_expansion = len(question_ids) > len(QUESTION_IDS)
        summary = {
            "finalized_at": utc_now(),
            "status": (
                "gate_4_question_expansion_complete_recommendation_stop"
                if is_question_expansion else "gate_4_complete_recommendation_stop"
            ),
            "completed_gates": [1, 2, 3, 4],
            "not_started_gates": [5, 6],
            "reason": (
                f"The explicitly authorized {len(question_ids)}-question Gate 4 pilot completed, "
                "but neither activation site met the held-out separation and split-half "
                "stability criteria."
            ),
            "gate_4_recommendation": gate4_decision["recommendation"],
            "question_ids": question_ids,
            "saved_target_model_generations": response_count,
            "reused_target_model_generations": reused_generations,
            "new_target_model_generations": response_count - reused_generations,
            "successful_judgments": judge_summary["persona_judgments"],
            "new_successful_judgments": judge_summary["new_judgments_this_command"],
            "failed_infrastructure_judge_attempts": failed_attempts,
            "checksum_entries": len(rows),
        }
    else:
        summary = {
            "finalized_at": utc_now(),
            "status": "stopped_after_gate_3_per_operational_stop_condition",
            "completed_gates": [1, 2, 3],
            "not_started_gates": [4, 5, 6],
            "reason": (
                "The exact prior cuML classifier emitted native line-search failures and "
                "two block-output multi-role fits had NLL worse than uniform."
            ),
            "target_model_generations": 0,
            "successful_judgments": 0,
            "failed_infrastructure_judge_attempts": 0,
            "checksum_entries": len(rows),
        }
    write_json(args.run_dir / "run-summary.json", summary)
    # Recompute once so run-summary itself is covered.
    rows = []
    for path in sorted(args.run_dir.rglob("*")):
        if not path.is_file() or path.name == "sha256sums.txt":
            continue
        rows.append((sha256_file(path), str(path.relative_to(args.run_dir)), path.stat().st_size))
    temporary.write_text(
        "".join(f"{digest}  {name}\n" for digest, name, _size in rows),
        encoding="utf-8",
    )
    temporary.replace(checksum_path)
    print(json.dumps({"status": "finalized", "files": len(rows), "path": str(args.run_dir)}, sort_keys=True))


def command_report(args: argparse.Namespace) -> None:
    if args.report_dir.exists():
        raise FileExistsError(f"Refusing to overwrite report directory: {args.report_dir}")
    gate1 = args.run_dir / "gate-1-environment"
    gate2 = args.run_dir / "gate-2-hook-smoke"
    gate3 = args.run_dir / "gate-3-role-probes"
    environment = json.loads((gate1 / "environment.json").read_text())
    hook = json.loads((gate2 / "hook-validation.json").read_text())
    decision = json.loads((gate3 / "gate-decision.json").read_text())
    metrics = pd.read_csv(gate3 / "probe-metrics.csv")
    stable = metrics[metrics.solver == "sklearn_standardized"]
    selected = stable[
        stable.probe.isin(["pilot_binary", "compact_system_user_cot_assistant"])
        & stable.layer.isin([12, 16])
    ][["probe", "activation_site", "layer", "balanced_accuracy", "nll", "uniform_nll", "n_train_tokens", "n_test_tokens"]]
    args.report_dir.mkdir(parents=True)
    copy_names = [
        (gate1 / "environment.json", "environment.json"),
        (gate1 / "artifact-inventory.json", "artifact-inventory.json"),
        (gate2 / "hook-validation.json", "hook-validation.json"),
        (gate3 / "probe-metrics.csv", "probe-metrics.csv"),
        (gate3 / "gate-decision.json", "gate-decision.json"),
        (gate3 / "probe-artifact-metadata.json", "probe-artifact-metadata.json"),
        (gate3 / "native-cuml-warnings.json", "native-cuml-warnings.json"),
    ]
    for source, name in copy_names:
        shutil.copy2(source, args.report_dir / name)
    selected.to_csv(args.report_dir / "selected-metrics.csv", index=False)
    hook_lines = []
    for layer in ["12", "16"]:
        item = hook["comparisons"]["layers"][layer]
        hook_lines.append(
            f"| {layer} | {item['pre_mlp_vs_pinned_custom_forward']['max_abs_error']:.1f} | "
            f"{item['block_output_vs_standard_hook']['max_abs_error']:.1f} | "
            f"{item['pre_mlp_vs_block_output']['cosine_similarity']:.4f} |"
        )
    metric_lines = []
    for row in selected.itertuples(index=False):
        metric_lines.append(
            f"| {row.probe} | {row.activation_site} | {row.layer} | "
            f"{row.balanced_accuracy:.4f} | {row.nll:.4f} | {row.uniform_nll:.4f} |"
        )
    readme = f"""# GPT-OSS-20B Assistant Axis pilot: stopped after Gate 3

Date: 2026-08-27

This run completed environment verification, dual-site hook validation, and the
50-passage compact role-probe regeneration. It then stopped exactly as required
by the handoff plan. No persona, Assistant Axis, CoT-Forgery, steering, target
model generation, or judge call was started.

## Reproducibility

- Run directory: `{args.run_dir}`
- Repository commit at start: `{environment['repository']['commit']}`
- Branch: `{environment['repository']['branch']}`
- Model: `{MODEL_ID}` at `{MODEL_REVISION}`
- GPU: `{environment['gpu']['query']}`
- PyTorch / Transformers: `{environment['versions']['torch']} / {environment['versions']['transformers']}`
- cuML / CuPy: `{environment['versions']['cuml']} / {environment['versions']['cupy']}`
- Activation sites: pre-MLP `post_attention_layernorm` output and decoder-block output
- Role-probe split: grouped by base neutral passage; seed 123
- Fit dtype: float32; saved compact coefficient dtype: float32

The coefficients are subset-derived compact replacements. They are not the lost
full-corpus probes and must not be described as an exact replacement.

## Gate 1

The pinned model has 24 layers and hidden size 2880. The model, tokenizer,
Harmony template, neutral passages, prompt manifest, split IDs, and token index
were checked. Reusable neutral/split artifacts matched their recorded digests.
Raw probe objects, the large activation archive, and held-out predictions were
not reused, even though stale readable paths were present.

## Gate 2

Hooked and unhooked logits were bit-exact. Both activation captures matched
their independent references exactly, all tensors were finite with hidden size
2880, and all hooks were removed after the run.

| Layer | Pre-MLP max error | Block-output max error | Cross-site cosine |
| ---: | ---: | ---: | ---: |
{chr(10).join(hook_lines)}

## Gate 3 stable-baseline results

| Probe | Site | Layer | Balanced accuracy | NLL | Uniform NLL |
| --- | --- | ---: | ---: | ---: | ---: |
{chr(10).join(metric_lines)}

Across layers 8–18, the standardized binary probe reached
{decision['binary_best_balanced_accuracy_by_site']['pre_mlp']:.4f} pre-MLP and
{decision['binary_best_balanced_accuracy_by_site']['block_output']:.4f} at block
output. The compact layer-16 pre-MLP probe reached
{decision['compact_layer16_pre_mlp_balanced_accuracy']:.4f}; therefore the
planned expansion from 50 to 100 passages was not triggered.

## Why the run stopped

The exact prior cuML solver emitted seven native L-BFGS line-search failures.
In addition, two block-output multi-role fits had NLL worse than uniform:

- pilot user/assistant/tool/CoT, layer 12: NLL 1.5333 versus 1.3863 uniform;
- compact system/user/CoT/assistant, layer 12: NLL 2.1722 versus 1.3863 uniform.

The handoff states that line-search warnings are failed numerical fits and lists
role-pilot NLL worse than uniform as an operational stop condition. Gates 4 and
5 were therefore not started, despite the healthy standardized baseline.

## Recommended next action

Debug the exact cuML probability calibration and optimization on the saved
manifest, while retaining the standardized fit as a diagnostic. Do not begin
persona generation until a reviewed Gate 3 rerun no longer triggers the stop
condition.
"""
    (args.report_dir / "README.md").write_text(readme, encoding="utf-8")
    print(json.dumps({"status": "report-created", "path": str(args.report_dir)}, sort_keys=True))


def command_report_gate4(args: argparse.Namespace) -> None:
    if args.report_dir.exists():
        raise FileExistsError(f"Refusing to overwrite report directory: {args.report_dir}")
    root = gate4_dir(args.run_dir)
    analysis = root / "analysis"
    decision = json.loads((analysis / "gate-decision.json").read_text())
    judge = json.loads((root / "judge-summary.json").read_text())
    extraction = json.loads((root / "extraction-summary.json").read_text())
    micro = json.loads((root / "micro-pilot-decision.json").read_text())
    pca = pd.read_csv(analysis / "pca-metrics.csv")
    heldout = pd.read_csv(analysis / "heldout-separation.csv")
    stability = pd.read_csv(analysis / "stability.csv")
    role_cosines = pd.read_csv(analysis / "role-axis-cosines.csv")
    angles = pd.read_csv(analysis / "principal-angles.csv")
    args.report_dir.mkdir(parents=True)
    copy_paths = [
        root / "prepare-personas/persona-manifest.csv",
        root / "prepare-personas/upstream-provenance.json",
        root / "prepare-personas/generation-settings.json",
        root / "prepare-personas/user-override.json",
        root / "generation-micro-summary.json",
        root / "generation-complete-summary.json",
        root / "micro-hook-validation.json",
        root / "micro-pilot-decision.json",
        root / "judge-scores.csv",
        root / "judge-summary.json",
        root / "inclusion-manifest.csv",
        root / "manual-review-summary.json",
        root / "extraction-summary.json",
        analysis / "gate-decision.json",
        analysis / "pca-metrics.csv",
        analysis / "heldout-separation.csv",
        analysis / "stability.csv",
        analysis / "projection-distribution.csv",
        analysis / "bootstrap-confidence.csv",
        analysis / "role-axis-cosines.csv",
        analysis / "principal-angles.csv",
    ]
    for source in copy_paths:
        shutil.copy2(source, args.report_dir / source.name)
    selected = pca[pca.layer.isin([12, 16])].copy()
    split = stability[
        stability.metric == "split_half_question_cosine"
    ][["activation_site", "layer", "value"]].rename(columns={"value": "split_half_cosine"})
    heldout_summary = heldout.groupby(["activation_site", "layer"], as_index=False)[
        ["auroc", "balanced_accuracy"]
    ].mean().rename(columns={"auroc": "heldout_auroc_mean", "balanced_accuracy": "heldout_balanced_accuracy_mean"})
    selected = selected.merge(split, on=["activation_site", "layer"]).merge(
        heldout_summary, on=["activation_site", "layer"]
    )
    selected.to_csv(args.report_dir / "selected-layer-summary.csv", index=False)
    table_lines = []
    for row in selected.itertuples(index=False):
        table_lines.append(
            f"| {row.activation_site} | {row.layer} | {row.axis_pc1_cosine:.3f} | "
            f"{row.pc1_explained_variance_ratio:.3f} | {row.heldout_auroc_mean:.3f} | "
            f"{row.heldout_balanced_accuracy_mean:.3f} | {row.split_half_cosine:.3f} |"
        )
    role_selected = role_cosines[
        role_cosines.layer.isin([12, 16])
        & (role_cosines.role_probe == "assistant_vs_other_roles")
    ]
    role_lines = [
        f"| {row.activation_site} | {row.layer} | {row.assistant_axis_cosine:.3f} |"
        for row in role_selected.itertuples(index=False)
    ]
    angle_selected = angles[
        angles.layer.isin([12, 16])
        & (angles.comparison == "assistant_axis_vs_role_probe_subspace")
    ]
    angle_lines = [
        f"| {row.activation_site} | {row.layer} | {row.angle_degrees:.1f}° |"
        for row in angle_selected.itertuples(index=False)
    ]
    readme = f"""# GPT-OSS-20B limited Assistant Axis pilot (Gate 4)

Date: 2026-08-27

## Outcome

Gate 4 completed with the preregistered limited panel of **12 personas**, not
the upstream collection of 275 personas and not 250 passages. The pilot used
two official extraction questions, producing 26 GPT-OSS responses total. The
result is **stop**: neither pre-MLP nor decoder-block output met all held-out and
stability acceptance criteria at any tested layer.

No Gate 5 CoT-Forgery generation, steering, or optional expansion was run.

## Compute and inclusion

- GPT-OSS generations: 26 (7-response micro-pilot, then 19 missing responses)
- Successful role-adherence judgments: 24
- Initial failed judge attempts: 6 authentication failures with no model output;
  these were preserved and retried once through OpenRouter
- Judge: official prompt and `openai/gpt-4.1-mini`
- Judge parse failures: {judge['parse_failures']}/24
- Score-3 persona responses included: {judge['accepted_response_count']}/24
- Personas with at least one included response: {judge['personas_with_accepted_response']}/12
- Included default responses: {extraction['included_default_responses']}/2
- Extraction layers: 8–20 at pre-MLP and decoder-block output
- Final response cap: 256 generated tokens; all responses reached the cap and
  no completed response was regenerated

The seven-response micro-pilot passed with {micro['accepted_count']}/6 accepted
personas, no parse failure, exact final-channel token boundaries, and finite,
distinct captures at both sites.

## Selected-layer results

| Site | Layer | Axis–PC1 cosine | PC1 EVR | Held-out AUROC | Held-out balanced accuracy | Question split-half cosine |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(table_lines)}

The decisive failure was question split-half stability: it ranged from 0.251 to
0.431 pre-MLP and 0.280 to 0.452 at block output, below the required 0.80 at
every layer. Leave-one-persona-out stability was high (minimum roughly
0.98–0.99), indicating the problem was question dependence rather than a single
dominant persona. Thresholded held-out balanced accuracy was 0.50 in both folds
at every layer, although rank-based AUROC was sometimes above 0.80.

PC1 alignment was weak at layers 12 and 16. Under the handoff interpretation,
that alone would not reject the contrast direction, but the failed split-half
and held-out criteria do.

## Same-site role-probe geometry

| Site | Layer | Cosine with assistant-vs-other role-probe direction |
| --- | ---: | ---: |
{chr(10).join(role_lines)}

| Site | Layer | Angle from Assistant Axis to role-probe subspace |
| --- | ---: | ---: |
{chr(10).join(angle_lines)}

The Assistant direction was nearly orthogonal to the compact role-probe
subspace at layers 12 and 16. These comparisons remain subset-pilot diagnostics,
especially because Gate 3's exact cuML fits had numerical failures.

## Reproducibility and caveats

- Persistent run: `{args.run_dir}`
- Official Assistant Axis commit: `{ASSISTANT_AXIS_COMMIT}`
- Model revision: `{MODEL_REVISION}`
- Instruction variant: official index 0 for every condition
- Questions: official IDs 0 and 5
- Generation: greedy, seed 123, reasoning effort low, 256-token cap
- Activation means: final-channel response tokens only, accumulated/saved float32
- Bootstrap unit: persona, 200 replicates

The pinned official repository's axis tests passed (15/15). Its generation test
module could not be collected because that commit's test imports a removed
`supports_system_prompt` symbol; the exact `format_conversation` function used
here was independently validated.

With 12 personas, 10 accepted personas, and two questions, PCA and uncertainty
are pilot diagnostics—not paper-quality estimates. The appropriate next action
is to review prompt/question dependence before spending on more personas.
"""
    (args.report_dir / "README.md").write_text(readme, encoding="utf-8")
    print(json.dumps({"status": "gate4-report-created", "path": str(args.report_dir)}, sort_keys=True))


def command_report_question_expansion(args: argparse.Namespace) -> None:
    if args.report_dir.exists():
        raise FileExistsError(f"Refusing to overwrite report directory: {args.report_dir}")
    root = gate4_dir(args.run_dir)
    analysis = root / "analysis"
    decision = json.loads((analysis / "gate-decision.json").read_text())
    judge = json.loads((root / "judge-summary.json").read_text())
    extraction = json.loads((root / "extraction-summary.json").read_text())
    reuse = json.loads((root / "question-expansion-reuse-provenance.json").read_text())
    manifest = pd.read_csv(root / "prepare-personas/persona-manifest.csv")
    stability = pd.read_csv(analysis / "stability.csv")
    heldout = pd.read_csv(analysis / "heldout-separation.csv")
    pca = pd.read_csv(analysis / "pca-metrics.csv")
    question_rows = manifest[["question_id", "question"]].drop_duplicates().sort_values("question_id")
    question_lines = [f"- {row.question_id}: {row.question}" for row in question_rows.itertuples(index=False)]

    split = stability[stability.metric == "split_half_question_cosine"][
        ["activation_site", "layer", "value"]
    ].rename(columns={"value": "split_half_cosine"})
    pairwise = stability[stability.metric == "pairwise_question_cosine"]
    lopo = stability[stability.metric == "leave_one_persona_out_cosine"]
    loqo = stability[stability.metric == "leave_one_question_out_cosine"]
    heldout_summary = heldout.groupby(["activation_site", "layer"], as_index=False)[
        ["auroc", "balanced_accuracy"]
    ].mean().rename(columns={"auroc": "heldout_auroc_mean", "balanced_accuracy": "heldout_balanced_accuracy_mean"})
    selected = pca[pca.layer.isin([12, 16])].merge(
        split, on=["activation_site", "layer"]
    ).merge(heldout_summary, on=["activation_site", "layer"])
    args.report_dir.mkdir(parents=True)
    selected.to_csv(args.report_dir / "selected-layer-summary.csv", index=False)
    table_lines = [
        f"| {row.activation_site} | {row.layer} | {row.split_half_cosine:.3f} | "
        f"{row.heldout_auroc_mean:.3f} | {row.heldout_balanced_accuracy_mean:.3f} | "
        f"{row.axis_pc1_cosine:.3f} |"
        for row in selected.itertuples(index=False)
    ]
    site_lines = []
    for site in ["pre_mlp", "block_output"]:
        site_split = split[split.activation_site == site].split_half_cosine
        site_pairwise = pairwise[pairwise.activation_site == site].value
        site_lopo = lopo[lopo.activation_site == site].value
        site_loqo = loqo[loqo.activation_site == site].value
        site_heldout = heldout[heldout.activation_site == site]
        site_lines.append(
            f"| {site} | {site_split.min():.3f}–{site_split.max():.3f} | "
            f"{site_pairwise.median():.3f} | {site_loqo.min():.3f} | {site_lopo.min():.3f} | "
            f"{site_heldout.auroc.mean():.3f} | {site_heldout.balanced_accuracy.mean():.3f} |"
        )
    copy_paths = [
        root / "prepare-personas/persona-manifest.csv",
        root / "prepare-personas/upstream-provenance.json",
        root / "prepare-personas/generation-settings.json",
        root / "prepare-personas/user-override.json",
        root / "question-expansion-reuse-provenance.json",
        root / "generation-complete-summary.json",
        root / "judge-scores.csv",
        root / "judge-summary.json",
        root / "inclusion-manifest.csv",
        root / "manual-review-summary.json",
        root / "extraction-summary.json",
        analysis / "gate-decision.json",
        analysis / "pca-metrics.csv",
        analysis / "heldout-separation.csv",
        analysis / "stability.csv",
        analysis / "projection-distribution.csv",
        analysis / "bootstrap-confidence.csv",
        analysis / "role-axis-cosines.csv",
        analysis / "principal-angles.csv",
    ]
    for source in copy_paths:
        shutil.copy2(source, args.report_dir / source.name)

    readme = f"""# GPT-OSS-20B Assistant Axis six-question expansion

Date: 2026-08-27

## Outcome

The explicitly authorized question expansion is complete. It retained the same
12-persona panel, reused the original 26 generations exactly, and added four
official extraction questions (52 new generations). The result remains
**stop**: neither activation site reached the preregistered 0.80 split-half
cosine at any tested layer.

This result argues against expanding to 50 personas under the current prompt
design. Question choice, rather than persona sampling, remains the dominant
source of instability.

## Questions and preregistered split

{chr(10).join(question_lines)}

- Half A: {decision['question_split_halves'][0]}
- Half B: {decision['question_split_halves'][1]}

The four added questions were selected before generation to cover factual
explanation, interpersonal support, unfamiliar-tool problem solving, and
workplace ethical pressure.

## Compute and inclusion

- Saved responses: 78 ({reuse['reused_generation_count']} reused, {reuse['new_generation_count']} new)
- Persona judgments: {judge['persona_judgments']} ({judge['new_judgments_this_command']} new)
- Judge parse failures: {judge['parse_failures']}
- Included persona responses: {judge['accepted_response_count']}/72
- Personas represented after filtering: {judge['personas_with_accepted_response']}/12
- Included activation records: {extraction['included_responses']}
- Manual review: 28 selected cases; zero score overrides or boundary failures

## Stability summary across layers 8–20

| Site | Three-vs-three cosine range | Median pairwise-question cosine | Min leave-one-question-out | Min leave-one-persona-out | Mean held-out AUROC | Mean held-out balanced accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(site_lines)}

The three-question-vs-three-question comparison improved over the original
single-question comparison, but remained below 0.80. Low pairwise-question
cosines show that the individual question axes are not measuring one common
direction. High leave-one-persona-out stability shows that adding personas is
unlikely to repair that disagreement.

## Selected layers

| Site | Layer | Three-vs-three cosine | Held-out AUROC | Held-out balanced accuracy | Axis–PC1 cosine |
| --- | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(table_lines)}

## Recommendation

Do not expand to 50 or 250 personas yet. First revise the estimator or identify
a semantically coherent question subset, then validate that choice on untouched
questions. Gate 5 CoT-Forgery generation and steering remain unstarted.

## Reproducibility

- Persistent run: `{args.run_dir}`
- Source two-question run: `{reuse['source_run']}`
- Official Assistant Axis commit: `{ASSISTANT_AXIS_COMMIT}`
- Model revision: `{MODEL_REVISION}`
- Generation: greedy, seed 123, lowest reasoning effort, 256-token cap
- Activation sites: pre-MLP and decoder-block output, layers 8–20
- Response statistic: final-channel token mean, accumulated and saved float32
"""
    (args.report_dir / "README.md").write_text(readme, encoding="utf-8")
    print(json.dumps({"status": "question-expansion-report-created", "path": str(args.report_dir)}, sort_keys=True))


def command_hook_smoke(args: argparse.Namespace) -> None:
    temporary, final = atomic_stage_dir(args.run_dir, "gate-2-hook-smoke")
    if not (args.run_dir / "gate-1-environment").is_dir():
        raise RuntimeError("Gate 1 must complete before Gate 2")
    if run_text(["git", "-C", str(UPSTREAM_ROOT), "rev-parse", "HEAD"]) != UPSTREAM_COMMIT:
        raise RuntimeError("Pinned upstream commit mismatch")
    sys.path.insert(0, str(UPSTREAM_ROOT))
    from utils.pretrained_models import gptoss as upstream_gptoss
    from utils.role_templates import load_chat_template

    patch_pinned_masking_api(upstream_gptoss)
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        cache_dir=HF_HOME,
        local_files_only=True,
        add_eos_token=False,
        add_bos_token=False,
        padding_side="left",
    )
    tokenizer.chat_template = load_chat_template(
        str(UPSTREAM_ROOT / "utils/chat_templates"), "gptoss-20b"
    )
    messages = [
        {"role": "system", "content": "You are ChatGPT, a large language model trained by OpenAI."},
        {"role": "user", "content": "Briefly explain why leaves look green."},
    ]
    prompt = (tokenizer.bos_token or "") + tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)

    torch.cuda.reset_peak_memory_stats()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        cache_dir=HF_HOME,
        local_files_only=True,
        attn_implementation="kernels-community/vllm-flash-attn3",
    ).to("cuda:0").eval()
    model.set_experts_implementation("eager")
    patch_pinned_model_api(model)
    inputs = {key: value.to(model.device) for key, value in encoded.items()}

    with torch.no_grad():
        baseline = model(**inputs, use_cache=False, return_dict=True)
        dual_outputs, dual, dual_handle_count = capture_forward(
            model, inputs, ("pre_mlp", "block_output")
        )
        pre_outputs, pre_reference, pre_handle_count = capture_forward(model, inputs, ("pre_mlp",))
        block_outputs, block_reference, block_handle_count = capture_forward(
            model, inputs, ("block_output",)
        )
        upstream = upstream_gptoss.run_gptoss_return_topk(
            model, **inputs, return_hidden_states=True
        )

    comparisons: dict[str, Any] = {
        "baseline_vs_dual_logits": tensor_comparison(baseline.logits, dual_outputs.logits),
        "baseline_vs_pre_reference_logits": tensor_comparison(baseline.logits, pre_outputs.logits),
        "baseline_vs_block_reference_logits": tensor_comparison(baseline.logits, block_outputs.logits),
        "layers": {},
    }
    for layer_ix in LAYERS:
        upstream_pre = upstream["all_pre_mlp_hidden_states"][layer_ix].reshape(
            encoded.input_ids.shape[0], encoded.input_ids.shape[1], -1
        )
        comparisons["layers"][str(layer_ix)] = {
            "pre_mlp_vs_standard_hook": tensor_comparison(
                dual["pre_mlp"][layer_ix], pre_reference["pre_mlp"][layer_ix]
            ),
            "pre_mlp_vs_pinned_custom_forward": tensor_comparison(
                dual["pre_mlp"][layer_ix], upstream_pre
            ),
            "block_output_vs_standard_hook": tensor_comparison(
                dual["block_output"][layer_ix], block_reference["block_output"][layer_ix]
            ),
            "sites_are_distinct": not torch.equal(
                dual["pre_mlp"][layer_ix], dual["block_output"][layer_ix]
            ),
            "pre_mlp_vs_block_output": tensor_comparison(
                dual["pre_mlp"][layer_ix], dual["block_output"][layer_ix]
            ),
        }

    result = {
        "captured_at": utc_now(),
        "model": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "layers": LAYERS,
        "activation_sites": {
            "pre_mlp": "model.model.layers[i].post_attention_layernorm output",
            "block_output": "model.model.layers[i] output after attention and MLP residual updates",
        },
        "prompt": prompt,
        "decoded_prompt": tokenizer.decode(encoded.input_ids[0]),
        "token_ids": encoded.input_ids[0].tolist(),
        "token_count": int(encoded.input_ids.shape[1]),
        "hook_counts": {
            "dual": dual_handle_count,
            "pre_reference": pre_handle_count,
            "block_reference": block_handle_count,
            "remaining_after_runs": sum(len(module._forward_hooks) for module in model.modules()),
        },
        "comparisons": comparisons,
        "peak_allocated_gpu_bytes": int(torch.cuda.max_memory_allocated()),
    }
    write_json(temporary / "hook-validation.json", result)

    required = [
        comparisons["baseline_vs_dual_logits"],
        comparisons["baseline_vs_pre_reference_logits"],
        comparisons["baseline_vs_block_reference_logits"],
    ]
    for layer_ix in LAYERS:
        metrics = comparisons["layers"][str(layer_ix)]
        required += [
            metrics["pre_mlp_vs_standard_hook"],
            metrics["pre_mlp_vs_pinned_custom_forward"],
            metrics["block_output_vs_standard_hook"],
        ]
        if not metrics["sites_are_distinct"]:
            raise AssertionError(f"Activation sites are bit-identical at layer {layer_ix}")
        for site in ["pre_mlp_vs_standard_hook", "block_output_vs_standard_hook"]:
            if metrics[site]["shape"][-1] != 2880:
                raise AssertionError(f"Unexpected hidden size at layer {layer_ix}: {site}")
    if not all(x.get("finite_left") and x.get("finite_right") for x in required):
        raise AssertionError("Non-finite values found during hook validation")
    if not all(x.get("bit_exact") for x in required):
        raise AssertionError("A required hook/custom-forward equality was not bit exact")
    if result["hook_counts"]["remaining_after_runs"]:
        raise AssertionError("Forward hooks remained installed after validation")
    temporary.rename(final)
    print(json.dumps({"stage": "gate-2-hook-smoke", "status": "complete", "path": str(final)}, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in [
        "inventory", "hook-smoke", "role-probes",
        "extract", "analyze", "finalize",
    ]:
        child = subparsers.add_parser(name)
        child.add_argument("--run-dir", type=Path, required=True)
    prepare = subparsers.add_parser("prepare-personas")
    prepare.add_argument("--run-dir", type=Path, required=True)
    prepare.add_argument("--question-ids", type=parse_question_ids, default=QUESTION_IDS)
    seed_expansion = subparsers.add_parser("seed-question-expansion")
    seed_expansion.add_argument("--source-run-dir", type=Path, required=True)
    seed_expansion.add_argument("--run-dir", type=Path, required=True)
    generate = subparsers.add_parser("generate")
    generate.add_argument("--run-dir", type=Path, required=True)
    generate.add_argument("--scope", choices=["micro", "complete"], required=True)
    score = subparsers.add_parser("score")
    score.add_argument("--run-dir", type=Path, required=True)
    score.add_argument("--scope", choices=["micro", "complete"], required=True)
    score.add_argument("--retry-infrastructure-failures", action="store_true")
    report = subparsers.add_parser("report")
    report.add_argument("--run-dir", type=Path, required=True)
    report.add_argument("--report-dir", type=Path, required=True)
    report_gate4 = subparsers.add_parser("report-gate4")
    report_gate4.add_argument("--run-dir", type=Path, required=True)
    report_gate4.add_argument("--report-dir", type=Path, required=True)
    report_questions = subparsers.add_parser("report-question-expansion")
    report_questions.add_argument("--run-dir", type=Path, required=True)
    report_questions.add_argument("--report-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "inventory":
        command_inventory(args)
    elif args.command == "hook-smoke":
        command_hook_smoke(args)
    elif args.command == "role-probes":
        command_role_probes(args)
    elif args.command == "prepare-personas":
        command_prepare_personas(args)
    elif args.command == "seed-question-expansion":
        command_seed_question_expansion(args)
    elif args.command == "generate":
        command_generate_personas(args)
    elif args.command == "score":
        command_score_personas(args)
    elif args.command == "extract":
        command_extract_personas(args)
    elif args.command == "analyze":
        command_analyze_personas(args)
    elif args.command == "finalize":
        command_finalize(args)
    elif args.command == "report":
        command_report(args)
    elif args.command == "report-gate4":
        command_report_gate4(args)
    elif args.command == "report-question-expansion":
        command_report_question_expansion(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
