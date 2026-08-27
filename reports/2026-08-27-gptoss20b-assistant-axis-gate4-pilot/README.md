# GPT-OSS-20B limited Assistant Axis pilot (Gate 4)

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
- Judge parse failures: 0/24
- Score-3 persona responses included: 15/24
- Personas with at least one included response: 10/12
- Included default responses: 2/2
- Extraction layers: 8–20 at pre-MLP and decoder-block output
- Final response cap: 256 generated tokens; all responses reached the cap and
  no completed response was regenerated

The seven-response micro-pilot passed with 4/6 accepted
personas, no parse failure, exact final-channel token boundaries, and finite,
distinct captures at both sites.

## Selected-layer results

| Site | Layer | Axis–PC1 cosine | PC1 EVR | Held-out AUROC | Held-out balanced accuracy | Question split-half cosine |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| pre_mlp | 12 | 0.127 | 0.400 | 0.667 | 0.500 | 0.268 |
| pre_mlp | 16 | 0.198 | 0.340 | 0.833 | 0.500 | 0.365 |
| block_output | 12 | 0.060 | 0.395 | 0.833 | 0.500 | 0.287 |
| block_output | 16 | 0.322 | 0.343 | 0.667 | 0.500 | 0.404 |

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
| pre_mlp | 12 | -0.036 |
| pre_mlp | 16 | -0.065 |
| block_output | 12 | -0.027 |
| block_output | 16 | -0.049 |

| Site | Layer | Angle from Assistant Axis to role-probe subspace |
| --- | ---: | ---: |
| pre_mlp | 12 | 86.7° |
| pre_mlp | 16 | 83.0° |
| block_output | 12 | 87.7° |
| block_output | 16 | 84.3° |

The Assistant direction was nearly orthogonal to the compact role-probe
subspace at layers 12 and 16. These comparisons remain subset-pilot diagnostics,
especially because Gate 3's exact cuML fits had numerical failures.

## Reproducibility and caveats

- Persistent run: `/workspace/role-probe-storage/outputs/gptoss20b-assistant-axis-20260827-1426`
- Official Assistant Axis commit: `a98961956072224eaf244eb289d6c01700b63795`
- Model revision: `6cee5e81ee83917806bbde320786a8fb61efebee`
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
