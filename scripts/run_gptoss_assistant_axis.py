#!/usr/bin/env python3
"""Resumable GPT-OSS-20B Assistant Axis pilot runner.

The initial commands implement Gate 1 artifact/environment inventory and the
Gate 2 dual-site hook smoke test. Later experiment stages are added as their
preceding gates pass.
"""

from __future__ import annotations

import argparse
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
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, log_loss
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
        "".join(f"{row['sha256']}  {row['path']}  {row['bytes']} bytes\n" for row in rows),
        encoding="utf-8",
    )
    temporary.replace(checksum_path)
    write_json(
        args.run_dir / "run-summary.json",
        {
            "finalized_at": utc_now(),
            "status": "stopped_after_gate_3_per_operational_stop_condition",
            "completed_gates": [1, 2, 3],
            "not_started_gates": [4, 5, 6],
            "reason": (
                "The exact prior cuML classifier emitted native line-search failures and "
                "two block-output multi-role fits had NLL worse than uniform."
            ),
            "target_model_generations": 0,
            "judge_calls": 0,
            "checksum_entries": len(rows),
        },
    )
    # Recompute once so run-summary itself is covered.
    rows = []
    for path in sorted(args.run_dir.rglob("*")):
        if not path.is_file() or path.name == "sha256sums.txt":
            continue
        rows.append((sha256_file(path), str(path.relative_to(args.run_dir)), path.stat().st_size))
    temporary.write_text(
        "".join(f"{digest}  {name}  {size} bytes\n" for digest, name, size in rows),
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
    for name in ["inventory", "hook-smoke", "role-probes", "finalize"]:
        child = subparsers.add_parser(name)
        child.add_argument("--run-dir", type=Path, required=True)
    report = subparsers.add_parser("report")
    report.add_argument("--run-dir", type=Path, required=True)
    report.add_argument("--report-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "inventory":
        command_inventory(args)
    elif args.command == "hook-smoke":
        command_hook_smoke(args)
    elif args.command == "role-probes":
        command_role_probes(args)
    elif args.command == "finalize":
        command_finalize(args)
    elif args.command == "report":
        command_report(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
