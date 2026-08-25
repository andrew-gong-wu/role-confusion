# External artifacts

Large model and activation-vector files are stored outside Git. The repository
ignores `*.pt` and `*.safetensors`; do not force-add them.

## Qwen3-32B

- Hugging Face repository: `Qwen/Qwen3-32B`
- Snapshot revision: `9216db5781bf21249d130ec9da846c4624c16137`
- Snapshot: 27 files, 65,540,298,478 bytes
- Weight index SHA-256:
  `bed42c6c55274bc08a1f616bceb3bcb84b3f02cb6584c573bd18c6519291ecd0`
- Verified remote load path:
  `/root/role-confusion-qwen3-32b-assistant-axis-20260825-ready/model`

The verified load path resolves all 17 weight shards. Shards 1-12 are symlinks
to files on `/workspace`; shards 13-17 are regular files on the root volume.
Keep both locations available while using this path.

## Lu et al. (2026) Assistant Axis

- Hugging Face dataset: `lu-christina/assistant-axis-vectors`
- Dataset revision: `3b3b788432ad33e3a28d9ff08e88a530c0740814`
- Dataset file: `qwen-3-32b/assistant_axis.pt`
- Size: 656,986 bytes
- SHA-256:
  `a207fe7a36563280b7b29010880aa0082bd8e3113c141cb4a2eed6b46c140211`
- Verified remote path:
  `/root/role-confusion-qwen3-32b-assistant-axis-20260825-ready/axis/assistant_axis.pt`

These files were downloaded from the published repositories. No axis
recomputation, model inference, or new experiment was run.
