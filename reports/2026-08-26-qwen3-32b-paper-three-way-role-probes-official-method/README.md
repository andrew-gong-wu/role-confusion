# Qwen3-32B paper-style three-way role probes

This report contains a three-way `user` / `assistant` / `tool` role probe for
dense Qwen3-32B, following the UCAT role space in the authors' full
reproduction notebook and the nested-role controls in Appendix G of Ye, Cui,
and Hadfield-Menell (2026), arXiv:2603.12277v6.

## Construction

- Seed 123; 250 neutral bases (62 C4 and 188 Dolma3), at most 1,024 tokens.
- Qwen-native wrappers:
  - user: `<|im_start|>user\n[CONTENT]<|im_end|>`
  - assistant final: content after a closed, variable-length `<think>` block
  - tool: `<|im_start|>user\n<tool_response>\n[CONTENT]...`
- Assistant thought filler came from another neutral base. Matching filler was
  prepended before the user and tool role tags to control absolute position.
- Tags and filler were excluded from training.
- The first 32 content positions were excluded, matching the authors' nested
  reasoning-model path. One short base therefore contributed no retained rows
  in any class: 747 rendered prompts and 410,064 token examples remained.

## Fitting

- Fresh three-way cuML multinomial logistic regression at decoder-block output.
- Layers 0, 4, ..., 60; hidden size 5,120.
- L2 penalty, no feature scaling, `C=0.1`, 5,000 maximum iterations. These are
  the authors' published Qwen3-30B-A3B probe settings, transferred to dense
  Qwen3-32B.
- Seeded 90/10 split over rendered prompts, as in the authors' notebook.
- Selected layer: maximum held-out token accuracy, lower layer breaking ties.

## Result

Layer 48 was selected:

| Metric | Value |
|---|---:|
| Held-out accuracy | 0.379560 |
| Balanced held-out accuracy | 0.392036 |
| Held-out NLL | 1.245826 |
| Train tokens | 368,068 |
| Held-out tokens | 41,996 |

Per-class recall at layer 48:

| Role | Recall | Held-out tokens |
|---|---:|---:|
| User | 0.335379 | 17,759 |
| Assistant | 0.648702 | 11,671 |
| Tool | 0.192026 | 12,566 |

The balanced score exceeds three-way chance (0.3333), but the raw score is
below the held-out majority-class baseline (0.4229), tool recall is poor, and
NLL is worse than uniform prediction (`ln(3) = 1.0986`). cuML emitted L-BFGS
line-search early-stop warnings. This probe therefore does **not** meet the
paper's high in-distribution-accuracy validity criterion and should not be
treated as a validated measurement instrument without further work.

## Vector comparison

At layer 48, cosine similarity with the downloaded Qwen3-32B assistant-persona
axis was 0.013798 for user, -0.019718 for assistant, and 0.025058 for tool.
These are descriptive only given the failed validity criterion.

The arrays are finite. Shapes are `(16, 3, 5120)` for all-layer centered
coefficients and `(3, 5120)` for selected-layer role directions. Maximum
absolute centering residual is `2.98e-8`.

## Reproduction locations

- Runner: `scripts/run_qwen_paper_tag_role_probes.py`
- Persistent output:
  `/workspace/role-probe-storage/outputs/qwen3-32b-paper-three-way-role-probes-seed123-v3-20260826`
- Persistent log:
  `/workspace/role-probe-storage/logs/qwen3-32b-paper-three-way-role-probes-seed123-v3-20260826.log`

The paper's released Qwen experiment targets Qwen3-30B-A3B. This is an
adaptation to the available dense Qwen3-32B checkpoint, not a claim to
reproduce the paper's Qwen numerical results.
