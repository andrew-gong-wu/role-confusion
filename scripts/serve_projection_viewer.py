#!/usr/bin/env python3
"""Local-only browser for encrypted GPT-OSS token projection trajectories."""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import os
import secrets
import subprocess
import tarfile
import webbrowser
from bisect import bisect_left
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "data" / "projection-viewer"
DEFAULT_UI = ROOT / "demo" / "projection-viewer" / "index.html"
TRAJECTORY_NAME = "pair-token-trajectories.jsonl.gz.enc"
SOURCE_NAME = "sensitive-artifacts.tar.gz.enc"
TRAJECTORY_KEY = Path("/Users/rigel/.codex/gate5-keys/gptoss20b-cot-forgery-assistant-axis-projections-20260827-2048.key")
SOURCE_KEY = Path("/Users/rigel/.codex/gate5-keys/gptoss20b-cot-forgery-gate6-400-20260827-1740.key")
RESPONSE_MEMBER = "cot-forgery-pilot/generation/responses.jsonl.gz"


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def decrypt_bytes(ciphertext: Path, key: Path, iterations: int = 600_000) -> bytes:
    if not ciphertext.is_file():
        raise FileNotFoundError(ciphertext)
    if not key.is_file():
        raise FileNotFoundError(key)
    result = subprocess.run(
        ["openssl", "enc", "-d", "-aes-256-cbc", "-pbkdf2", "-iter", str(iterations),
         "-md", "sha256", "-in", str(ciphertext), "-pass", f"file:{key}"],
        check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    return result.stdout


def verify_ciphertext(path: Path, metadata_path: Path) -> dict:
    metadata = json.loads(metadata_path.read_text())
    if sha256(path.read_bytes()) != metadata["ciphertext_sha256"]:
        raise RuntimeError(f"Ciphertext checksum mismatch: {path}")
    return metadata


def jsonl_gzip_bytes(blob: bytes) -> list[dict]:
    with gzip.GzipFile(fileobj=io.BytesIO(blob)) as stream:
        return [json.loads(line) for line in stream if line.strip()]


def load_source_responses(archive_plaintext: bytes, expected_sha256=None) -> list[dict]:
    with tarfile.open(fileobj=io.BytesIO(archive_plaintext), mode="r:gz") as archive:
        member = archive.extractfile(RESPONSE_MEMBER)
        if member is None:
            raise RuntimeError(f"Missing archive member: {RESPONSE_MEMBER}")
        compressed = member.read()
        if expected_sha256 and sha256(compressed) != expected_sha256:
            raise RuntimeError("Source-response archive member checksum mismatch")
        return jsonl_gzip_bytes(compressed)


def byte_decoder() -> dict[str, int]:
    base = list(range(ord("!"), ord("~") + 1))
    base += list(range(ord("¡"), ord("¬") + 1))
    base += list(range(ord("®"), ord("ÿ") + 1))
    chars = list(base)
    extra = 0
    for value in range(256):
        if value not in base:
            base.append(value)
            chars.append(256 + extra)
            extra += 1
    return {chr(char): value for value, char in zip(base, chars)}


class TokenDecoder:
    def __init__(self, tokenizer_path: Path):
        data = json.loads(tokenizer_path.read_text())
        self.id_to_token = {int(index): token for token, index in data["model"]["vocab"].items()}
        self.special = {int(row["id"]): row["content"] for row in data.get("added_tokens", [])}
        self.bytes = byte_decoder()

    def token_bytes(self, token_id: int) -> bytes:
        if token_id in self.special:
            return self.special[token_id].encode("utf-8")
        token = self.id_to_token.get(token_id, f"<|token_{token_id}|>")
        return bytes(self.bytes.get(char, ord("?")) for char in token)

    def piece(self, token_id: int) -> str:
        raw = self.token_bytes(token_id)
        try:
            return raw.decode("utf-8")
        except UnicodeDecodeError:
            return "".join(chr(v) if 32 <= v < 127 else f"\\x{v:02x}" for v in raw)


def token_index_for_byte(offsets: list[int], target: int) -> int:
    return max(0, min(len(offsets) - 2, bisect_left(offsets, target + 1) - 1))


def semantic_markers(row: dict, decoder: TokenDecoder) -> list[dict]:
    ids = row["prompt_token_ids"] + row["generated_token_ids"]
    offsets = [0]
    for token_id in ids:
        offsets.append(offsets[-1] + len(decoder.token_bytes(int(token_id))))
    prompt = row.get("rendered_prompt", "")
    markers = [{"index": int(row["prompt_token_count"]), "label": "generation begins", "kind": "generation"}]
    request = row.get("source_request") or ""
    request_char = prompt.find(request) if request else -1
    if request_char >= 0:
        start_byte = len(prompt[:request_char].encode("utf-8"))
        end_byte = start_byte + len(request.encode("utf-8"))
        markers.extend([
            {"index": token_index_for_byte(offsets, start_byte), "label": "harmful request begins", "kind": "request"},
            {"index": token_index_for_byte(offsets, max(start_byte, end_byte - 1)) + 1, "label": "harmful request ends", "kind": "request-end"},
        ])
    if row.get("condition") != "baseline" and request_char >= 0:
        policy_char = prompt.find("\n\n", request_char + len(request))
        if policy_char >= 0:
            policy_char += 2
            while policy_char < len(prompt) and prompt[policy_char].isspace():
                policy_char += 1
            policy_byte = len(prompt[:policy_char].encode("utf-8"))
            if policy_byte < len(prompt.encode("utf-8")):
                markers.append({"index": token_index_for_byte(offsets, policy_byte), "label": "forged policy begins", "kind": "forgery"})
    return markers


class ProjectionStore:
    def __init__(self, data_dir: Path, trajectory_key: Path, source_key: Path):
        trajectory_path, source_path = data_dir / TRAJECTORY_NAME, data_dir / SOURCE_NAME
        trajectory_meta = verify_ciphertext(trajectory_path, data_dir / "trajectory-encryption-metadata.json")
        source_meta = verify_ciphertext(source_path, data_dir / "source-encryption-metadata.json")
        trajectory_plain = decrypt_bytes(trajectory_path, trajectory_key)
        if sha256(trajectory_plain) != trajectory_meta["plaintext_sha256"]:
            raise RuntimeError("Trajectory plaintext checksum mismatch")
        trajectories = jsonl_gzip_bytes(trajectory_plain)
        del trajectory_plain
        source_plain = decrypt_bytes(source_path, source_key)
        responses = load_source_responses(
            source_plain, source_meta["archived_files"][RESPONSE_MEMBER]
        )
        del source_plain
        self.decoder = TokenDecoder(data_dir / "tokenizer.json")
        self.trajectories = {row["condition_id"]: row for row in trajectories}
        self.responses = {row["condition_id"]: row for row in responses}
        if set(self.trajectories) != set(self.responses):
            raise RuntimeError("Trajectory/source condition IDs do not match")
        with (data_dir / "paired-results.csv").open(newline="") as stream:
            labels = {row["source_row_id"]: row for row in csv.DictReader(stream)}
        pairs: dict[str, dict[str, str]] = {}
        for condition_id, row in self.responses.items():
            pairs.setdefault(row["source_row_id"], {})[row["condition"]] = condition_id
        self.pairs = []
        for ordinal, source_id in enumerate(sorted(pairs), 1):
            conditions = pairs[source_id]
            if len(conditions) != 2:
                raise RuntimeError(f"Incomplete pair: {source_id}")
            label = labels.get(source_id, {})
            attack_id, baseline_id = conditions["cot_forgery_base_no_qualifier"], conditions["baseline"]
            attack, baseline = self.responses[attack_id], self.responses[baseline_id]
            if label.get("pair_analyzable", "").lower() != "true":
                result = "unlabeled"
            elif label.get("attack_label") == "HARMFUL_RESPONSE":
                result = "successful"
            else:
                result = "unchanged"
            self.pairs.append({"ordinal": ordinal, "category": attack["question_category"], "result": result,
                "attack": attack_id, "baseline": baseline_id,
                "attack_tokens": len(attack["prompt_token_ids"]) + len(attack["generated_token_ids"]),
                "baseline_tokens": len(baseline["prompt_token_ids"]) + len(baseline["generated_token_ids"])})

    def index(self) -> list[dict]:
        return [{k: v for k, v in row.items() if k not in {"attack", "baseline"}} for row in self.pairs]

    def pair(self, ordinal: int) -> dict:
        if ordinal < 1 or ordinal > len(self.pairs):
            raise KeyError(ordinal)
        pair = self.pairs[ordinal - 1]
        return {"ordinal": ordinal, "category": pair["category"], "result": pair["result"],
            "responses": {"attack": self.response(pair["attack"]), "baseline": self.response(pair["baseline"])}}

    def response(self, condition_id: str) -> dict:
        source, trajectory = self.responses[condition_id], self.trajectories[condition_id]
        ids = source["prompt_token_ids"] + source["generated_token_ids"]
        if len(ids) != trajectory["token_count"]:
            raise RuntimeError("Token count changed")
        boundaries, markers = dict(trajectory["boundaries"]), semantic_markers(source, self.decoder)
        if boundaries.get("last_generated_analysis") is not None:
            markers.append({"index": boundaries["last_generated_analysis"], "label": "generated analysis ends", "kind": "analysis"})
        markers.append({"index": boundaries["first_final_content"], "label": "final answer begins", "kind": "final"})
        return {"condition": "attack" if source["condition"] != "baseline" else "baseline",
            "category": source["question_category"], "finish_reason": source["finish_reason"],
            "hit_token_limit": bool(source["hit_token_limit"]), "prompt_token_count": int(source["prompt_token_count"]),
            "tokens": [{"index": i, "id": int(token_id), "text": self.decoder.piece(int(token_id)),
                "role": trajectory["roles"][i], "content": bool(trajectory["content"][i]),
                "generated": bool(trajectory["generated"][i])} for i, token_id in enumerate(ids)],
            "coordinates": trajectory["coordinates"], "markers": sorted(markers, key=lambda x: x["index"]),
            "boundaries": boundaries}


def make_handler(store: ProjectionStore, ui: bytes, capability: str):
    class Handler(BaseHTTPRequestHandler):
        server_version = "ProjectionViewer/1"
        def log_message(self, fmt: str, *args):
            return
        def send_blob(self, content_type: str, blob: bytes):
            self.send_response(HTTPStatus.OK)
            for key, value in [("Content-Type", content_type), ("Content-Length", str(len(blob))),
                ("Cache-Control", "no-store, max-age=0"), ("Pragma", "no-cache"),
                ("Referrer-Policy", "no-referrer"), ("X-Content-Type-Options", "nosniff"),
                ("Cross-Origin-Resource-Policy", "same-origin"), ("X-Frame-Options", "DENY"),
                ("Content-Security-Policy", "default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; connect-src 'self'; img-src 'self' data:; frame-ancestors 'none'")]: self.send_header(key, value)
            self.end_headers(); self.wfile.write(blob)
        def fail(self, status: HTTPStatus, message: str):
            blob = json.dumps({"error": message}).encode(); self.send_response(status)
            self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(blob)))
            self.send_header("Cache-Control", "no-store"); self.end_headers(); self.wfile.write(blob)
        def do_GET(self):
            host = self.headers.get("Host", "").split(":", 1)[0]
            parsed, query = urlparse(self.path), parse_qs(urlparse(self.path).query)
            if host not in {"127.0.0.1", "localhost"}:
                self.fail(HTTPStatus.FORBIDDEN, "Loopback host required"); return
            if query.get("key", [""])[0] != capability:
                self.fail(HTTPStatus.FORBIDDEN, "Invalid session key"); return
            if parsed.path == "/": self.send_blob("text/html; charset=utf-8", ui); return
            try:
                if parsed.path == "/api/index": payload = {"pairs": store.index()}
                elif parsed.path == "/api/pair": payload = store.pair(int(query.get("ordinal", ["0"])[0]))
                else: self.fail(HTTPStatus.NOT_FOUND, "Not found"); return
            except (KeyError, ValueError) as exc:
                self.fail(HTTPStatus.BAD_REQUEST, f"Invalid request: {exc}"); return
            self.send_blob("application/json; charset=utf-8", json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode())
    return Handler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--trajectory-key", type=Path, default=TRAJECTORY_KEY)
    parser.add_argument("--source-key", type=Path, default=SOURCE_KEY)
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--no-open", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args(); os.umask(0o077)
    print("Loading and verifying encrypted artifacts…", flush=True)
    store = ProjectionStore(args.data_dir, args.trajectory_key, args.source_key)
    capability, ui = secrets.token_urlsafe(24), DEFAULT_UI.read_bytes()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(store, ui, capability))
    url = f"http://127.0.0.1:{server.server_port}/?key={capability}"
    print(f"Projection viewer ready: {url}", flush=True)
    print("Press Ctrl-C to stop. Sensitive data remains in memory only.", flush=True)
    if not args.no_open: webbrowser.open(url)
    try: server.serve_forever()
    except KeyboardInterrupt: pass
    finally: server.server_close()


if __name__ == "__main__": main()
