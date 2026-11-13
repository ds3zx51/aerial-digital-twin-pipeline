# BU refactor audit

## Evidence boundaries

`BU00–BU13` establish scenario-specific physical propagation/RSS evidence. `BU14` performs the
semantic interface mapping into the historical 32-column contract. `BU15` checks downstream AI
usability while keeping `ai_server/AdvTabDDPM` frozen. These claims must not be collapsed.

## Preserved unchanged

- Existing B/A/legacy scripts and historical result directories.
- `scripts/aodt/run_uav_00BU_07BU.py`, `uav_core.py`, `uav_runtime.py`, and `run_uav_08BU.py`.
- `scripts/adapter/legacy_aodt_contract_adapter.py` and its exact 32-column order.
- `ai_server/AdvTabDDPM` implementation and existing model/training artifacts.
- Existing Tokyo, Rome/ColO-RAN, and other candidate workflows.

## Scientific source decisions

The final journal PDF controls the coverage setup. Its 3.3 GHz SISO coverage override supersedes
the generic Table 2 3.4 GHz 4x4 ray-tracing profile for this branch. The generic profile and raw
MAT carrier metadata remain recorded for provenance and are never silently discarded.

The exact supplementary foliage data and formal AODT scene are absent, so the implementation
does not invent them. `paper_reconstruction` fails closed; `osm_only_smoke` is permanently marked
non-formal.
