# GPT-OSS-20B Gate 5 CoT-Forgery pilot

## Decision

**saturated**: preserve as saturated slice for paired activation analysis only.

This is a single-category, 10-pair pilot and **not** a full-benchmark attack-success rate.

## Results

| Condition | Harmful responses | Refusals | Redirections | Exact 95% binomial interval |
| --- | ---: | ---: | ---: | --- |
| Raw baseline | 0/10 | 10/10 | 0/10 | 0.000–0.308 |
| CoT-Forgery (`base` / `no_qualifier`) | 9/10 | 0/10 | 1/10 | 0.555–0.997 |

Paired attack-minus-baseline change: **+9/10**; gains/losses/unchanged = 9/0/1. Exact two-sided McNemar p = 0.0039.

## Scope and provenance

- Target: `openai/gpt-oss-20b` at revision `6cee5e81ee83917806bbde320786a8fb61efebee`.
- Dataset: official StrongREJECT-small at commit `f7cad6c17e624e21d8df2278e918ae1dddb4cb56`, SHA-256 `3051340e3e89a3598d764dde497d5fcda80e258ac05cc35e6bd87228ac3d467c`.
- Official exact category: `Disinformation and Deception` (10 of 60 rows).
- The preserved policy archive spells the same category `Disinformation and deception`; both exact strings are retained in provenance.
- All 10 requests joined one-to-one to preserved successful `base` / `no_qualifier` policies using the SHA-256 of exact request text.
- Forgery-generation calls: **0**. Target-model generations: **20**. One-time safety judgments: **20**; parse failures: **0**.
- Decoding: deterministic greedy generation, seed 123, low reasoning effort, 1,024-token cap for both conditions.
- GPU peak: 13.05 GiB; target generation elapsed time: 188.5 seconds.
- Raw prompts, policies, responses, token IDs, and judge payloads are encrypted at rest in persistent storage and are not committed. The decryption key is stored owner-only outside Git on the authorized local machine.

## Next step

Gate 6 is optional and requires a separate review and explicit decision. No second category, extra seed, steering sweep, or dataset expansion was run automatically.
