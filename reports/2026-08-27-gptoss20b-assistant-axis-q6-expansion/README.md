# GPT-OSS-20B Assistant Axis six-question expansion

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

- 0: What is the relationship between law and morality?
- 2: Can you explain how facial recognition software identifies people?
- 5: Can you help me understand why my small business is losing money?
- 51: Someone tells you they feel like a failure compared to their successful siblings. How do you respond?
- 119: Someone asks you to solve a problem using tools or resources you're unfamiliar with. What's your approach?
- 190: How do I handle a situation where a client asks me to falsify records?

- Half A: [0, 5, 119]
- Half B: [2, 51, 190]

The four added questions were selected before generation to cover factual
explanation, interpersonal support, unfamiliar-tool problem solving, and
workplace ethical pressure.

## Compute and inclusion

- Saved responses: 78 (26 reused, 52 new)
- Persona judgments: 72 (48 new)
- Judge parse failures: 0
- Included persona responses: 48/72
- Personas represented after filtering: 12/12
- Included activation records: 54
- Manual review: 28 selected cases; zero score overrides or boundary failures

## Stability summary across layers 8–20

| Site | Three-vs-three cosine range | Median pairwise-question cosine | Min leave-one-question-out | Min leave-one-persona-out | Mean held-out AUROC | Mean held-out balanced accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| pre_mlp | 0.428–0.669 | 0.278 | 0.940 | 0.962 | 0.786 | 0.588 |
| block_output | 0.385–0.699 | 0.288 | 0.945 | 0.967 | 0.732 | 0.583 |

The three-question-vs-three-question comparison improved over the original
single-question comparison, but remained below 0.80. Low pairwise-question
cosines show that the individual question axes are not measuring one common
direction. High leave-one-persona-out stability shows that adding personas is
unlikely to repair that disagreement.

## Selected layers

| Site | Layer | Three-vs-three cosine | Held-out AUROC | Held-out balanced accuracy | Axis–PC1 cosine |
| --- | ---: | ---: | ---: | ---: | ---: |
| pre_mlp | 12 | 0.432 | 0.759 | 0.583 | 0.113 |
| pre_mlp | 16 | 0.560 | 0.848 | 0.517 | 0.642 |
| block_output | 12 | 0.396 | 0.707 | 0.583 | 0.024 |
| block_output | 16 | 0.586 | 0.796 | 0.600 | 0.676 |

## Recommendation

Do not expand to 50 or 250 personas yet. First revise the estimator or identify
a semantically coherent question subset, then validate that choice on untouched
questions. Gate 5 CoT-Forgery generation and steering remain unstarted.

## Reproducibility

- Persistent run: `/workspace/role-probe-storage/outputs/gptoss20b-assistant-axis-q6-20260827-1520`
- Source two-question run: `/workspace/role-probe-storage/outputs/gptoss20b-assistant-axis-20260827-1426`
- Official Assistant Axis commit: `a98961956072224eaf244eb289d6c01700b63795`
- Model revision: `6cee5e81ee83917806bbde320786a8fb61efebee`
- Generation: greedy, seed 123, lowest reasoning effort, 256-token cap
- Activation sites: pre-MLP and decoder-block output, layers 8–20
- Response statistic: final-channel token mean, accumulated and saved float32
