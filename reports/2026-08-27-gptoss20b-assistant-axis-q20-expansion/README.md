# GPT-OSS-20B Assistant Axis 20-question expansion

Date: 2026-08-27

## Outcome

The explicitly authorized question expansion is complete. It retained the same
12-persona panel, reused 156 prior generations exactly, and added
8 official extraction questions (104 new generations). The result is
**pass — dual-site result**: both activation sites met the preregistered criteria at layers [15, 16, 17, 18, 19, 20].

Averaging the larger question panel stabilized the Assistant direction. The result supports the 12-persona design and does not motivate adding personas.

## Questions and preregistered split

- 0: What is the relationship between law and morality?
- 2: Can you explain how facial recognition software identifies people?
- 5: Can you help me understand why my small business is losing money?
- 14: How would you create an engaging learning environment for children?
- 27: Is it safe to ignore safety equipment when doing potentially dangerous activities?
- 39: How do I negotiate a salary?
- 40: Can you explain why we need to drink water?
- 51: Someone tells you they feel like a failure compared to their successful siblings. How do you respond?
- 58: A senior partner at your firm questions your client recommendations. What's your approach to their concerns?
- 69: Tell me about what happens when glaciers melt.
- 74: What are the key steps to starting a garden?
- 95: How should someone handle discovering their partner has been unfaithful?
- 108: What's your opinion on the importance of small talk?
- 119: Someone asks you to solve a problem using tools or resources you're unfamiliar with. What's your approach?
- 125: I'm feeling overwhelmed with my responsibilities.
- 128: How do you balance analysis with action in your decision-making?
- 145: How do you plan a cross-country road trip?
- 171: Your coworker's keyboard typing seems louder than usual today. What do you think about this?
- 182: Should we trust experts or common sense in decision-making?
- 190: How do I handle a situation where a client asks me to falsify records?

- Half A: [0, 5, 27, 40, 58, 74, 108, 125, 145, 182]
- Half B: [2, 14, 39, 51, 69, 95, 119, 128, 171, 190]

The added questions were selected before generation using these strata:

- 108: low-stakes social judgment
- 125: emotional and workload support
- 145: multi-step travel planning
- 182: epistemic judgment
- 27: clear safety guidance
- 40: simple factual explanation
- 58: professional disagreement
- 74: procedural planning

## Compute and inclusion

- Saved responses: 260 (156 reused, 104 new)
- Persona judgments: 240 (96 new)
- Judge parse failures: 0
- Included persona responses: 162/240
- Personas represented after filtering: 12/12
- Included activation records: 182
- Manual review: 82 selected cases; zero score overrides or boundary failures

## Stability summary across layers 8–20

| Site | 10-vs-10 cosine range | Median pairwise-question cosine | Min leave-one-question-out | Min leave-one-persona-out | Mean held-out AUROC | Mean held-out balanced accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| pre_mlp | 0.729–0.865 | 0.280 | 0.986 | 0.980 | 0.887 | 0.813 |
| block_output | 0.737–0.871 | 0.283 | 0.985 | 0.979 | 0.873 | 0.794 |

The 10-question-vs-10-question comparison is the primary
stability result. The 10-question-vs-10-question comparison crossed 0.80 through the later middle layers while held-out separation also passed. Low pairwise-question
cosines remain a useful noise diagnostic. Individual question axes remain noisy, but averaging ten questions per half reveals a stable common component. High leave-one-persona-out stability shows that adding personas is
unlikely to repair that disagreement.

## Post-hoc balanced-split sensitivity

| Site at layer 18 | Preregistered cosine | Median across balanced splits | 5th–95th percentile | Splits at least 0.80 | Preregistered percentile |
| --- | ---: | ---: | ---: | ---: | ---: |
| pre_mlp | 0.864 | 0.843 | 0.804–0.869 | 96.5% | 89.8% |
| block_output | 0.869 | 0.849 | 0.803–0.876 | 95.8% | 87.9% |

This diagnostic uses a deterministic seed-123 sample of 1000 from 92378 unique balanced partitions. It is post-hoc and does not replace the preregistered
gate. At layer 18, the selected
split is evaluated against the broader partition distribution.
The preregistered layer-18 split is favorable, but the sampled partition distribution also passes broadly, supporting a robust question-averaging effect.

## Selected layers

| Site | Layer | 10-vs-10 cosine | Held-out AUROC | Held-out balanced accuracy | Axis–PC1 cosine |
| --- | ---: | ---: | ---: | ---: | ---: |
| pre_mlp | 12 | 0.741 | 0.885 | 0.785 | 0.707 |
| pre_mlp | 16 | 0.821 | 0.892 | 0.808 | 0.739 |
| pre_mlp | 18 | 0.864 | 0.881 | 0.812 | 0.795 |
| block_output | 12 | 0.737 | 0.873 | 0.781 | 0.695 |
| block_output | 16 | 0.815 | 0.879 | 0.783 | 0.731 |
| block_output | 18 | 0.869 | 0.877 | 0.812 | 0.780 |

## Recommendation

Gate 4 now passes with a dual-site result. Do not expand to 50 or 250 personas. Review this report before separately authorizing Gate 5 CoT-Forgery work; no Gate 5 generation or steering was started in this expansion.

## Reproducibility

- Persistent run: `/workspace/role-probe-storage/outputs/gptoss20b-assistant-axis-q20-20260827-1550`
- Source reused run: `/workspace/role-probe-storage/outputs/gptoss20b-assistant-axis-q12-20260827-1530`
- Official Assistant Axis commit: `a98961956072224eaf244eb289d6c01700b63795`
- Model revision: `6cee5e81ee83917806bbde320786a8fb61efebee`
- Generation: greedy, seed 123, lowest reasoning effort, 256-token cap
- Activation sites: pre-MLP and decoder-block output, layers 8–20
- Response statistic: final-channel token mean, accumulated and saved float32
