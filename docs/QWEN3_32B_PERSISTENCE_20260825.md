# Qwen3-32B persistence handoff (2026-08-25)

The Cambria host's root filesystem contained the final Qwen3-32B shards and
the downloaded assistant persona axis. Before GPU shutdown, the artifacts were
made recoverable from the persistent `/workspace` volume without committing
model weights to Git.

## Persistent sources

- `/workspace/role-confusion-qwen3-32b-assistant-axis-20260825/model`
  contains model metadata, complete shards 1-12, prefixes of shards 13-16, and
  an empty shard-17 placeholder. Size at handoff: approximately 58 GB.
- `/workspace/qwen3-32b-completion-20260825/model-tails` contains the exact
  missing bytes for shards 13-16 and all bytes of shard 17. Size: approximately
  4.23 GB.
- `/workspace/qwen3-32b-completion-20260825/axis/assistant_axis.pt` contains the
  complete 64-by-5120 assistant persona axis.
- `/workspace/role-probe-storage` contains the partner probe environment,
  downloaded data, and outputs. Its rebuildable 7.3 GB `uv-cache` directory was
  removed—with user approval—to make room for the completion bundle. The
  virtual environment, Hugging Face cache, outputs, logs, and source files were
  not altered.

Every partial-shard-plus-tail stream was byte-compared with the live,
load-tested full shard before shutdown; all five comparisons passed. The axis
copy also passed a byte comparison. Exact full-shard sizes and SHA-256 digests,
tail digests, and the axis digest are embedded in
`scripts/restore_qwen3_32b_persistent.py`.

## Restore on a future host

The destination must not already exist and must have at least 66 GB free:

```bash
python scripts/restore_qwen3_32b_persistent.py \
  --output-dir /root/qwen3-32b-restored
```

The script creates only new files, reconstructs all 17 shards, copies the
tokenizer/configuration files and full persona axis, and verifies every final
size and SHA-256 digest. The restored paths are:

- `/root/qwen3-32b-restored/model`
- `/root/qwen3-32b-restored/axis/assistant_axis.pt`

## Git and local backup state

All probe code and compact outputs are committed on `codex/andre-work`. Model
weights remain excluded by `.gitignore` and were not pushed. A local ignored
backup directory also contains the full axis and an interrupted partial copy of
shard 1; it is not needed for restoration and should not be mistaken for the
authoritative persistent bundle:

`artifacts/qwen3-32b-assistant-axis-20260825`
