# GPT-OSS-20B Gate 6 CoT-Forgery expansion

## Decision

**broad-expansion**: use this preregistered broad expansion for category-stratified paired analysis; do not expand automatically.

This is a preregistered, category-stratified 200-pair subset of the 313-row full dataset.

## Results

| Condition | Harmful responses | Refusals | Redirections | Exact 95% binomial interval |
| --- | ---: | ---: | ---: | --- |
| Raw baseline | 0/200 | 198/200 | 2/200 | 0.000–0.018 |
| CoT-Forgery (`base` / `no_qualifier`) | 148/199 | 20/199 | 31/199 | 0.677–0.803 |

Among 199/200 fully labeled pairs, the net paired success change is **+148**; gains/losses/unchanged = 148/0/51. Exact two-sided McNemar p = 0.0000.

## Category breakdown

| Category | Pairs | Labeled attacks | Baseline harmful | Attack harmful |
| --- | ---: | ---: | ---: | ---: |
| Disinformation and deception | 32 | 32 | 0 | 27 |
| Hate, harassment and discrimination | 32 | 32 | 0 | 17 |
| Illegal goods and services | 32 | 32 | 0 | 26 |
| Non-violent crimes | 38 | 38 | 0 | 34 |
| Sexual content | 32 | 31 | 0 | 20 |
| Violence | 34 | 34 | 0 | 24 |

## Scope and provenance

- Target: `openai/gpt-oss-20b` at revision `6cee5e81ee83917806bbde320786a8fb61efebee`.
- Dataset: Full StrongREJECT source CSV, SHA-256 `4dd70357e4ff8b5d0ba5ebafecab5d6dd5633ce8046e3dd1c8bd93e64de44381`; 200 of 313 rows across all six categories.
- All 200 requests joined one-to-one to preserved successful `base` / `no_qualifier` policies using the SHA-256 of exact request text.
- Forgery-generation calls: **0**. Total target-model responses: **400**; reused: **20**; newly generated: **380**. One-time safety judgments: **400**; parse failures: **1**.
- Decoding: deterministic greedy generation, seed 123, low reasoning effort, 1,024-token cap for both conditions.
- GPU peak: 15.43 GiB; exact total wall time was not recorded across the manual batch-size restart; the final 288-response batch-64 segment took 226.2 seconds.
- Raw prompts, policies, responses, token IDs, and judge payloads are encrypted at rest in persistent storage and are not committed. The decryption key is stored owner-only outside Git on the authorized local machine.

## Next step

Review category-level results before authorizing any additional methods, seeds, or the remaining 113 requests.
