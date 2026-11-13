# Publication and semantic scope

The supported claim is backend substitution at the AI-facing interface:
whether an unchanged downstream pipeline can consume an AODT-derived contract
and what downstream utility/runtime it yields.

It is not a claim that AODT physically reproduces Colosseum. Identical column
names do not establish semantic equivalence.

## `ul_sinr`

The frozen `ul_sinr` column is retained only for interface reproducibility.
Its semantic role is `aodt_em_proxy_ul_sinr_db`, derived with all other UEs
treated as cochannel-active and with no actual PRB/activity mask. Absolute CIR
normalization is not independently verified. Therefore:

- classification: `AI_CONTRACT_COMPATIBLE_ONLY`;
- Colosseum semantic equivalence: `NOT_YET_VERIFIED_SEMANTIC_EQUIVALENT`;
- physical distribution comparison: prohibited; and
- native RAN/PHY SINR wording: prohibited.

Future RAN-mode work should preserve native pre/post-equalization telemetry and
independently establish the Colosseum estimator before mapping either field.

## Statistical units

UE/slot rows are correlated observations, not independent repetitions. Means,
standard deviations, and confidence intervals across runs require independent
dataset generation/model-training seeds. Do not select seeds after inspecting
outcomes.
