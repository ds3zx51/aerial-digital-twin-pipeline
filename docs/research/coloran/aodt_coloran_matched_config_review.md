# AODT_COLORAN_MATCHED configuration review

Status: configuration design only, 2026-08-12. The backend scope is the open-source NVIDIA AODT client, worker, and native telemetry only; external RF backends are excluded. No GIS scene, AODT worker, EM, RAN, adapter, or AdvTabDDPM job was executed. The existing Tokyo 8-RU/50-UE branch and all historical results were left unchanged. The target repository has no Git metadata, so `AODT_COLORAN_MATCHED` is an isolated profile under `configs/coloran/`, not a Git branch.

## 1. Exact ColO-RAN baseline provenance

The intended upstream family is `wineslab/colosseum-oran-coloran-dataset/rome_static_medium`. Its documented family profile is seven BSs on source scenario nodes 1, 8, 15, 22, 29, 36, and 43; 42 static UEs; six UEs per BS; and two UEs per BS for each of eMBB, MTC, and URLLC.

The local AdvTabDDPM baseline is materially narrower and has unresolved ancestry. The recovered package contains 50 CSV files and 12,800 rows, with `sched0`, `sched1`, and `sched2`, but only `tr0` and `exp0`. It contains filenames `bs0` through `bs7`, IMSI suffixes 6000 through 6049, and `num_ues=50`. Those facts conflict with the public 7-BS/42-UE family and public repetition naming. Consequently, the local package is not claimed to be an exact copy or exact subset of the public corpus.

The exact files, row counts, inclusion flags, and SHA-256 hashes are recorded in `coloran_advtabddpm_source_inventory.csv`. The recovered lineage, field order, transformations, split sizes, and artifact hashes are recorded in `coloran_advtabddpm_provenance.md`.

## 2. Parameters matched exactly in the design

These are preregistered target values, not executed results:

- 7 serving RUs, 42 UEs, 6 UEs per RU.
- Static UEs, with six deterministic uniform-in-area samples within a 50 m disk around each RU.
- Two eMBB, two MTC, and two URLLC UEs per RU as experiment metadata.
- Source BS node identifiers `[1, 8, 15, 22, 29, 36, 43]` retained as lineage identifiers.
- Source LTE targets: 980 MHz DL, 1020 MHz UL, 10 MHz, 50 LTE PRBs.
- Source scheduler identifiers 0/RR, 1/WF, and 2/PF.
- All 28 documented source RBG allocation tuples, `tr0` through `tr27`, are preserved in the reference profile without pretending that AODT enforces them.
- A 250 ms SCOPE KPI aggregation target is source-backed by the inspected `metrics_period_secs=0.25` path, not selected from a plot.

## 3. Parameters matched approximately

- The scene is classified `GIS_GROUNDED_APPROXIMATE_COLORAN_ROME`. Exact ColO-RAN BS coordinates were not recovered.
- The proposed central-Rome bounding box, OSM source, terrain source, CRS, and provenance plan are scene-generation inputs only. RU coordinates remain unresolved and are not invented.
- Deterministic UE coordinates are local ENU offsets relative to each future RU position. They reproduce the documented radial rule, not the original absent UE coordinates.
- AODT `mcs`, `nPrb`, `tbs`, and `outcome` can support candidate interval aggregates, but exact SCOPE inclusion, counter, HARQ, and resource-grid semantics still require a trace test.
- AODT `preEqSinr` is the closest native measurement boundary found, but it is not yet semantically equivalent to SCOPE `ul_sinr`.

## 4. LTE/NR differences

ColO-RAN is LTE. The installed AODT RAN path is NR. `50 LTE PRBs` is therefore not copied into an AODT NR configuration. The installed artifacts expose bandwidth and subcarrier-spacing controls but do not establish the accepted 10 MHz/15 kHz NR grid or its configured PRB count.

| Source ColO-RAN value | Nearest AODT candidate | Technical reason | Expected semantic difference | Publication limitation |
|---|---|---|---|---|
| LTE, 10 MHz, 50 LTE PRBs | NR, 10 MHz and 15 kHz candidates; PRB count unresolved | LTE and NR resource grids and scheduler units differ | Grant sizes, control overhead, MCS/TBS behavior, and RBG meaning may differ | Do not claim resource-grid matching or directly compare `slice_prb`/granted PRBs as equal units |
| LTE turbo iterations | No AODT field | NR does not use LTE turbo coding for the data channel | Decoder telemetry has a different coding chain | `ul_turbo_iters` is unavailable and must not be zero-filled |
| LTE CQI/PMI/RI reports | No equivalent native fields; only scheduled layers is present | Telemetry interface does not expose these reports | Missing link-adaptation state | These contract fields remain unavailable, not reconstructed from MCS or SINR |

## 5. FDD/TDD differences

The source uses separate 980 MHz DL and 1020 MHz UL carriers. The inspected AODT schema exposes one carrier frequency per RU and the worker exposes TDD patterns; it does not expose a native FDD or separate direction-specific carrier setting.

The following options are retained without selecting one based on output similarity:

- **Option A:** one documented single-carrier NR RAN run. This permits scheduled RAN telemetry but retains a declared FDD-to-TDD and carrier mismatch.
- **Option B:** two direction-specific EM propagation runs, 980 MHz for DL and 1020 MHz for UL. This preserves direction-specific propagation frequencies but cannot replace scheduler/PHY RAN telemetry.
- **Option C:** declare the FDD-to-TDD difference non-alignable and scope the study as AI-contract/backend compatibility rather than radio-system reproduction.

## 6. Scene and geography limitation

The plan uses a central-Rome geographic context and independently sourced GIS inputs. It is not RF1017 and does not reuse the Tokyo geometry. Because the original seven geographic BS coordinates are absent, no exact matched-scene claim is allowed. Before scene generation, every selected BS needs a source identifier, retrieval date, license, original coordinates, transformation record, and input hash.

## 7. Scheduler support

PF is `SUPPORTED_AND_VERIFIED` at configuration-artifact level because the installed worker explicitly uses `Scheduler Mode=PF`. RR is `SUPPORTED_BUT_UNVERIFIED`: a candidate is requested, but the accepted worker token and behavior were not verified by execution or implementation source. WF is `NOT_PUBLICLY_EXPOSED`; no WF configuration was generated and it must not be approximated with PF or RR.

The native client YAML has no scheduler key. Candidate YAMLs therefore do not invent one. Scheduler selection remains an external worker-configuration prerequisite whose exact file and hash must be snapshotted per run.

## 8. Slice support

Native network slicing, per-slice PRB/RBG masks, and per-slice scheduler policies were not found in the installed public client/worker surface. The result is:

`COLORAN_SLICE_ALLOCATION_NOT_REPRODUCIBLE_NATIVELY`

Traffic class and slice ID may be retained as experiment metadata, but intended allocations cannot be written into the AI-facing `slice_prb` field unless the executed scheduler actually enforces and reports them.

## 9. Traffic and queue support

No native per-UE packet-arrival input, offered-load profile, queue state, packet counter, requested-PRB counter, or traffic-class control was found. The currently feasible design is `POST_HOC_TRAFFIC_COMPATIBILITY_MODEL`. It can deterministically maintain external queues and drain them using AODT TBS/outcome telemetry, but it cannot influence scheduler decisions and is not full ColO-RAN equivalence.

The source traffic profile is also blocked: the paper and public README agree on eMBB but conflict on MTC and URLLC rates. The exact runtime TGEN/job configuration tied to the local baseline was not recovered. The reference profile retains all values as `SOURCE_CONFLICT_REQUIRES_RESOLUTION`.

## 10. AI-contract mapping status

The historical field order is documented in `coloran_ai_contract_mapping.csv`. Native/derived candidates are limited to telemetry-supported scheduled metrics and explicit metadata. Buffers, packets, CQI, RSSI, PHR, requested PRBs, PMI, RI, `ul_n`, and turbo iterations remain unavailable or non-equivalent and must remain missing with reason codes.

`ul_sinr` is specifically not a verified cross-backend physical field. SCOPE reports a 250 ms scheduled-PUSCH channel-estimator statistic with export censoring. AODT `preEqSinr` is classified `VERIFIED_CLOSEST_MEASUREMENT`, `postEqSinr` is `PLAUSIBLE_BUT_UNVERIFIED`, and the legacy all-UE EM proxy is `NOT_COMPARABLE`. None may be silently written as ColO-RAN `ul_sinr`.

## 11. Blocking unknowns

1. Exact transformation lineage from the public 7-BS/42-UE corpus to the local 8-BS/50-UE/`exp0` package.
2. Exact runtime traffic-generator settings for MTC and URLLC.
3. Exact geographic positions and antenna/heights/powers for the seven ColO-RAN sites.
4. Supported AODT 10 MHz/15 kHz NR grid, FFT size, and accepted carrier range near 1 GHz.
5. A valid single-carrier choice or an explicit decision among FDD/TDD Options A-C.
6. Exact RR worker token and behavior.
7. Lack of native WF, slicing, per-slice allocation, traffic arrivals, queues, and several contract KPIs.
8. SCOPE-to-AODT SINR estimator, inclusion, weighting, and reporting-window equivalence.
9. AODT RU/UE transmit powers, antenna configuration, and georeferenced RU coordinates.

The candidate YAMLs deliberately contain `null` for unresolved physical inputs and use `_NOT_READY` simulation IDs. They are parseable design artifacts, not executable configurations.

## 12. One-cell pilot readiness

The topology and deterministic six-UE layout are ready as a design. Execution is not ready. Before a one-cell RR diagnostic, resolve the carrier/grid/FFT and radio-power fields, install or generate the approximate Rome scene, set a sourced RU position, verify the RR worker token, snapshot `config_ran.json`, and choose the FDD/TDD handling policy. The pilot must preserve raw telemetry and may not populate unavailable AI-contract fields.

## 13. Full-run readiness

The full 7-RU/42-UE configs are skeletons only. In addition to all pilot gates, a full run requires seven independently sourced RU positions, scene validation, verified multi-cell capacity, an approved traffic/queue compatibility scope, and a decision about how to handle the non-reproducible slice allocations. Only RR and PF candidates exist; WF is excluded.

The current local baseline does not justify running all `tr0`-`tr27` and `exp1`-`exp5` combinations. Its observed input contains only `tr0`/`exp0`, and scheduler/traffic provenance is confounded. The final experiment matrix must be frozen only after that lineage is resolved.

## 14. Estimated number of required runs

The minimum technical sequence is one one-cell RR diagnostic, followed by one full RR and one full PF smoke run: **3 AODT runs**. This is a functionality estimate, not a publication statistics plan.

For publication-level backend statistics, let `K` be the preregistered number of independent seeds/model-training runs after a power or precision analysis. The minimum planned AODT count is `1 pilot + 2 schedulers x K full runs`. If `K=5` is adopted only as a planning placeholder, that is 11 AODT runs. It must not be represented as statistically sufficient without the analysis, and UE/slot rows must not be counted as independent runs. A comparable ColO-RAN independent-run set is currently absent from the local `exp0` package.

## 15. Runtime and cost from measured local evidence

The only recovered local timing evidence is from the historical Tokyo 8-RU/50-UE EM branch: 256 EM steps required 607.6907348632812 worker seconds; collection required 1501.516 seconds and validation 12.421 seconds. This is measured evidence for that different EM workload only. It is not extrapolated into Rome RAN runtime, per-run cost, or monetary cost.

AODT RAN runtime, scene-generation runtime, GPU instance type, hourly price, and monetary cost are unavailable. No cost estimate is reported. Future runs must record wall-clock stage timestamps, usable-sample counts, hardware/instance identity, price source and date, and configuration/data/code hashes.

## Configuration artifacts and gates

- `configs/coloran/coloran_rome_static_medium_reference.yaml`: authoritative target and unresolved conflicts.
- `configs/coloran/coloran_rome_scene_plan.yaml`: approximate GIS plan; no scene generated.
- `configs/coloran/coloran_ue_layout_candidate.csv`: deterministic local ENU offsets for 42 UEs.
- `configs/coloran/aodt_coloran_one_cell_pilot.yaml`: intentionally non-executable one-cell design.
- `configs/coloran/aodt_coloran_rr_candidate.yaml`: intentionally non-executable full RR design; external worker verification required.
- `configs/coloran/aodt_coloran_pf_candidate.yaml`: intentionally non-executable full PF design.
- No WF config exists because WF support was not verified.

## Final recommendation

REQUIRES_EXTERNAL_COLORAN_COMPATIBILITY_LAYER
