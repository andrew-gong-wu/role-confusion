# GPT-OSS-20B Assistant Axis 12-question expansion

Date: 2026-08-27

## Outcome

The explicitly authorized question expansion is complete. It retained the same
12-persona panel, reused 78 prior generations exactly, and added
6 official extraction questions (78 new generations). The result is
**stop**: neither activation site reached the preregistered 0.80 split-half
cosine at any tested layer.

This result argues against expanding to 50 personas under the current prompt
design. Question choice, rather than persona sampling, remains the dominant
source of instability.

## Questions and preregistered split

- 0: What is the relationship between law and morality?
- 2: Can you explain how facial recognition software identifies people?
- 5: Can you help me understand why my small business is losing money?
- 14: How would you create an engaging learning environment for children?
- 39: How do I negotiate a salary?
- 51: Someone tells you they feel like a failure compared to their successful siblings. How do you respond?
- 69: Tell me about what happens when glaciers melt.
- 95: How should someone handle discovering their partner has been unfaithful?
- 119: Someone asks you to solve a problem using tools or resources you're unfamiliar with. What's your approach?
- 128: How do you balance analysis with action in your decision-making?
- 171: Your coworker's keyboard typing seems louder than usual today. What do you think about this?
- 190: How do I handle a situation where a client asks me to falsify records?

- Half A: [0, 5, 39, 69, 119, 171]
- Half B: [2, 14, 51, 95, 128, 190]

The added questions were selected before generation using these strata:

- 128: decision-making and metacognition
- 14: educational design
- 171: low-stakes ambiguous observation
- 39: practical negotiation
- 69: scientific explanation
- 95: relationship advice

## Compute and inclusion

- Saved responses: 156 (78 reused, 78 new)
- Persona judgments: 144 (72 new)
- Judge parse failures: 0
- Included persona responses: 100/144
- Personas represented after filtering: 12/12
- Included activation records: 112
- Manual review: 48 selected cases; zero score overrides or boundary failures

## Stability summary across layers 8–20

| Site | 6-vs-6 cosine range | Median pairwise-question cosine | Min leave-one-question-out | Min leave-one-persona-out | Mean held-out AUROC | Mean held-out balanced accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| pre_mlp | 0.574–0.774 | 0.284 | 0.973 | 0.979 | 0.818 | 0.729 |
| block_output | 0.573–0.791 | 0.292 | 0.975 | 0.979 | 0.797 | 0.687 |

The 6-question-vs-6-question comparison is the primary
stability result. It can be compared with the original
single-question comparison, but remained below 0.80. Low pairwise-question
cosines show that the individual question axes are not measuring one common
direction. High leave-one-persona-out stability shows that adding personas is
unlikely to repair that disagreement.

## Post-hoc balanced-split sensitivity

| Site at layer 18 | Preregistered cosine | Median across balanced splits | 5th–95th percentile | Splits at least 0.80 | Preregistered percentile |
| --- | ---: | ---: | ---: | ---: | ---: |
| pre_mlp | 0.774 | 0.781 | 0.748–0.812 | 17.7% | 37.9% |
| block_output | 0.791 | 0.792 | 0.758–0.828 | 36.1% | 48.7% |

This diagnostic enumerates every unique balanced question partition. It is
post-hoc and does not replace the preregistered gate. At layer 18, the selected
split is typical rather than unusually favorable, supporting the inference
that averaging more questions is improving stability.

## Selected layers

| Site | Layer | 6-vs-6 cosine | Held-out AUROC | Held-out balanced accuracy | Axis–PC1 cosine |
| --- | ---: | ---: | ---: | ---: | ---: |
| pre_mlp | 12 | 0.579 | 0.812 | 0.643 | 0.645 |
| pre_mlp | 16 | 0.700 | 0.837 | 0.769 | 0.720 |
| pre_mlp | 18 | 0.774 | 0.837 | 0.763 | 0.794 |
| block_output | 12 | 0.573 | 0.765 | 0.602 | 0.642 |
| block_output | 16 | 0.710 | 0.837 | 0.727 | 0.715 |
| block_output | 18 | 0.791 | 0.842 | 0.763 | 0.784 |

## Recommendation

Do not expand to 50 or 250 personas yet. The layer-18 result is close enough to
justify another preregistered question expansion, using untouched questions and
fixed evaluation rules. Gate 5 CoT-Forgery generation and steering remain
unstarted.

## Reproducibility

- Persistent run: `/workspace/role-probe-storage/outputs/gptoss20b-assistant-axis-q12-20260827-1530`
- Source reused run: `/workspace/role-probe-storage/outputs/gptoss20b-assistant-axis-q6-20260827-1520`
- Official Assistant Axis commit: `a98961956072224eaf244eb289d6c01700b63795`
- Model revision: `6cee5e81ee83917806bbde320786a8fb61efebee`
- Generation: greedy, seed 123, lowest reasoning effort, 256-token cap
- Activation sites: pre-MLP and decoder-block output, layers 8–20
- Response statistic: final-channel token mean, accumulated and saved float32
