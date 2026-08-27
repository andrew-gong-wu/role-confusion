# GPT-OSS-20B Assistant Axis pilot: stopped after Gate 3

Date: 2026-08-27

This run completed environment verification, dual-site hook validation, and the
50-passage compact role-probe regeneration. It then stopped exactly as required
by the handoff plan. No persona, Assistant Axis, CoT-Forgery, steering, target
model generation, or judge call was started.

## Reproducibility

- Run directory: `/workspace/role-probe-storage/outputs/gptoss20b-assistant-axis-20260827-1426`
- Repository commit at start: `2a149717685012dac17a6be45c6d0b604a19ba58`
- Branch: `codex/partner-work`
- Model: `openai/gpt-oss-20b` at `6cee5e81ee83917806bbde320786a8fb61efebee`
- GPU: `NVIDIA H100 80GB HBM3, 81559 MiB, 81079 MiB, 580.126.09`
- PyTorch / Transformers: `2.9.1+cu128 / 5.16.1`
- cuML / CuPy: `25.10.00 / 14.2.0`
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
| 12 | 0.0 | 0.0 | 0.4398 |
| 16 | 0.0 | 0.0 | 0.4813 |

## Gate 3 stable-baseline results

| Probe | Site | Layer | Balanced accuracy | NLL | Uniform NLL |
| --- | --- | ---: | ---: | ---: | ---: |
| pilot_binary | pre_mlp | 12 | 0.9946 | 0.0545 | 0.6931 |
| pilot_binary | pre_mlp | 16 | 0.9989 | 0.0192 | 0.6931 |
| pilot_binary | block_output | 12 | 0.9914 | 0.0593 | 0.6931 |
| pilot_binary | block_output | 16 | 0.9925 | 0.0278 | 0.6931 |
| compact_system_user_cot_assistant | pre_mlp | 12 | 0.7529 | 0.7106 | 1.3863 |
| compact_system_user_cot_assistant | pre_mlp | 16 | 0.9009 | 0.2833 | 1.3863 |
| compact_system_user_cot_assistant | block_output | 12 | 0.7375 | 0.7603 | 1.3863 |
| compact_system_user_cot_assistant | block_output | 16 | 0.8722 | 0.3654 | 1.3863 |

Across layers 8–18, the standardized binary probe reached
1.0000 pre-MLP and
0.9979 at block
output. The compact layer-16 pre-MLP probe reached
0.9009; therefore the
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
