# AdvTabDDPM ColO-RAN provenance audit — AODT_COLORAN_MATCHED source profile

Status: configuration-design inspection, updated 2026-08-12 against `${REPO_ROOT}`. No simulation, training, adapter, or pipeline execution was performed. The candidate backend scope is the open-source NVIDIA AODT client, worker, and native telemetry only.

## Corrected baseline identity

The earlier RF1017 audit is retained but is now marked `NOT_THE_ADVTabDDPM_BASELINE`. The publication candidate is the ColO-RAN `rome_static_medium` family, not TRACTOR/RF1017. No RF scenario numeric ID was found in an authoritative artifact; `coloran_rf_scenario_id = UNRESOLVED`.

The official ColO-RAN repository describes `rome_static_medium` as seven base stations on Colosseum nodes 1, 8, 15, 22, 29, 36, and 43, with 42 static UEs, six per BS, two per slice, uniformly placed within 50 m. The paper reports 0.98 GHz DL, 1.02 GHz UL, 10 MHz/50 LTE PRBs, three slices, and RR/WF/PF scheduling. These facts define the intended family; they do not by themselves prove that the local evaluation files are an unmodified subset of it.

## Authoritative versus local evidence

| Item | Official ColO-RAN family | Local AdvTabDDPM package | Audit conclusion |
|---|---|---|---|
| BS count/indexing | 7, `bs1`–`bs7` | filenames include `bs0`–`bs7` | conflict |
| UE count | 42; six/BS | `num_ues=50`; IMSI suffixes 6000–6049 | conflict |
| traffic configurations | `tr0`–`tr27` | only `tr0` | strict subset in name only |
| repetitions | public tree uses `exp1` and later | only `exp0` | conflict |
| input files | raw SCOPE metrics; no `traffic_type` | 50 already flattened/labeled files | transformation provenance missing |
| rows | repository-wide experiment corpus | 50 × 256 = 12,800 | local evaluation subset |

Therefore the strongest defensible statement is: **the local data has ColO-RAN/SCOPE-like schema and naming and may be derived from a related preparation, but exact ancestry from the official `wineslab/colosseum-oran-coloran-dataset/rome_static_medium` corpus is not verified.** A publication must not call the local 50-file package an exact copy or exact subset until an upstream manifest or transformation script reconciles the conflicts.

## Exact local lineage

1. `data/colosseum/sched*_tr*_exp*_bs*_slices_bs*_*_*_metrics.csv` is globbed and lexically sorted by `colosseum_dataset_csv_to_npy.py`.
2. All 50 matched CSVs are concatenated after selecting one label, 26 numeric, and five categorical columns. The label `traffic_type` already exists; this script does not create it.
3. A proportional 70/15/15 split uses `random_state=42`, producing actual train/validation/test sizes 8,960/1,920/1,920.
4. `make_colosseum_22.py` reduces 26+5 inputs to 20 numeric plus three categorical columns. It drops numeric `power_multiplier`, `tx_errors downlink (%)`, `ul_rssi`, `dl_pmi`, `dl_ri`, `ul_n`, and categorical `IMSI`, `slicing_enabled`.
5. Diffusion configuration points to `data/colosseum_22`. Its actual arrays are 8,960/1,920/1,920; the `info.json` sizes 76,209/16,330/16,331 are stale and must not be reported.
6. Classifier preprocessing loads the 20+3 arrays, concatenates them, removes `slice_id`, subtracts the first timestamp within each split, forms length-four sliding windows separately by label, and min–max normalizes using training windows only.
7. The July evaluation loads `colosseum_dataset_globalnorm.pkl` (SHA-256 `B0324E5A6CABB34F842E65D5BAF712E99DB5A2CA72EBBEB6D49A1AD5231EE635`) and `colosseum_global_cols_maxmin.pkl` (SHA-256 `289689736ADA059A7E1CAD2599F00099EEACF7E57474A880914402AF2520D544`). It evaluates an existing diffusion checkpoint; the package has no training manifest binding that checkpoint to these exact input hashes.

Merged historical AI-contract CSV SHA-256: `0EDADFC03F8BF7CFD4A8580268A9DA396669019E865E0A1F4EEAF2955A65CC87`.

The per-file paths and hashes are in `advtabddpm_coloran_source_inventory.csv`; every listed file was included.

### Local value audit

All 12,800 local rows use `power_multiplier=1`. The observed MCS ranges are DL 0–24 and UL 0–22. By label: class 0/eMBB has `slice_prb=60`, DL 0–22, UL 0–21; class 1/mMTC has `slice_prb=20`, DL 0–24, UL 0–22; class 2/URLLC has `slice_prb=15`, DL 0–17, UL 0–15. These `slice_prb` values are not the official `tr0=(2,13,2)` RBG tuple and further demonstrate an undocumented flattening/configuration step.

For every local file and every row, `traffic_type` is internally consistent with the filename traffic token: 0=eMBB, 1=mMTC, 2=URLLC. This supports that it is a source-traffic label rather than a row-wise inference from radio KPIs. However, the code/artifact that assigned it was not recovered, so creation lineage remains unresolved.

## Exact historical AI contract

Order is authoritative because it is hard-coded by the conversion script:

`traffic_type`, `Timestamp`, `num_ues`, `slice_prb`, `power_multiplier`, `dl_mcs`, `dl_n_samples`, `dl_buffer [bytes]`, `tx_brate downlink [Mbps]`, `tx_pkts downlink`, `tx_errors downlink (%)`, `dl_cqi`, `ul_mcs`, `ul_n_samples`, `ul_buffer [bytes]`, `rx_brate uplink [Mbps]`, `rx_pkts uplink`, `rx_errors uplink (%)`, `ul_rssi`, `ul_sinr`, `phr`, `sum_requested_prbs`, `sum_granted_prbs`, `dl_pmi`, `dl_ri`, `ul_n`, `ul_turbo_iters`, `IMSI`, `RNTI`, `slicing_enabled`, `slice_id`, `scheduling_policy`.

Classification precedence is: E for label; B/C for control/metadata; F where AODT lacks the field; G for the specifically tempting but unverified native mapping; otherwise A/D for native or postprocessed SCOPE KPIs. Each field below has exactly one A–G classification.

| # | column | class | ColO-RAN source and exact semantics | unit | temporal aggregation | AODT candidate / status | evidence and required action |
|---:|---|---|---|---|---|---|---|
| 0 | traffic_type | E CLASS_LABEL | Label already present in local flattened file | class ID | constant per file | traffic-profile metadata / unresolved lineage | No inspected script creates it and public raw CSV lacks it; recover label-generation manifest; never infer from KPI values |
| 1 | Timestamp | C EXPERIMENT_METADATA | CSV timestamp | ms | one row/report | slot/batch time / plausible | Align clocks and window boundaries explicitly |
| 2 | num_ues | C EXPERIMENT_METADATA | configured UE population | count | constant per run | config count / direct | Reconcile local 50 versus official 42 first |
| 3 | slice_prb | B RAN_CONTROL_CONFIGURATION | flattened slice resource value | unresolved | constant/config dependent | none / unresolved | Recover transformation and whether value is PRB or RBG-derived |
| 4 | power_multiplier | B RAN_CONTROL_CONFIGURATION | SCOPE power-control multiplier | dimensionless | config value | RU/UE dBm / not same parameter | Recover runtime setting; do not convert by assumption |
| 5 | dl_mcs | A NATIVE_RAN_KPI | mean scheduled DL MCS | index | report window | telemetry `mcs` / aggregation unverified | Define valid scheduled rows and averaging |
| 6 | dl_n_samples | A NATIVE_RAN_KPI | number of contributing DL PHY samples | count | report window | scheduled-record count / unverified | Match validity rules |
| 7 | dl_buffer [bytes] | F NOT_AVAILABLE_IN_AODT | DL MAC buffer occupancy | bytes | snapshot/report | none | Do not fabricate; approve missingness policy |
| 8 | tx_brate downlink [Mbps] | D DERIVED_DURING_PREPROCESSING | MAC byte rate divided by elapsed time | Mbit/s | report window | successful TBS sum / plausible | Verify retransmission and counter boundary |
| 9 | tx_pkts downlink | F NOT_AVAILABLE_IN_AODT | DL MAC packet counter | count | report window | none | A TB is not automatically a packet |
| 10 | tx_errors downlink (%) | D DERIVED_DURING_PREPROCESSING | MAC error percentage with export zero/floor | % | report window | failed-TB fraction / plausible | Verify numerator/denominator semantics |
| 11 | dl_cqi | F NOT_AVAILABLE_IN_AODT | UE-reported DL CQI | index | report window | none | Do not infer from MCS or SINR |
| 12 | ul_mcs | A NATIVE_RAN_KPI | cumulative mean scheduled-PUSCH MCS | index | 250 ms | UL telemetry `mcs` / aggregation unverified | Match inclusion and reset rules |
| 13 | ul_n_samples | A NATIVE_RAN_KPI | valid scheduled-PUSCH sample count | count | 250 ms | scheduled UL record count / unverified | Determine failed/outcome-row inclusion |
| 14 | ul_buffer [bytes] | F NOT_AVAILABLE_IN_AODT | UL MAC buffer occupancy | bytes | snapshot/report | none | Do not fabricate |
| 15 | rx_brate uplink [Mbps] | D DERIVED_DURING_PREPROCESSING | MAC received-byte rate divided by elapsed time | Mbit/s | report window | successful UL TBS sum / plausible | Verify HARQ and counter semantics |
| 16 | rx_pkts uplink | F NOT_AVAILABLE_IN_AODT | UL MAC packet counter | count | report window | none | TB count is not packet count |
| 17 | rx_errors uplink (%) | D DERIVED_DURING_PREPROCESSING | MAC error percentage with zero/floor | % | report window | failed-TB fraction / plausible | Verify exact error boundary |
| 18 | ul_rssi | F NOT_AVAILABLE_IN_AODT | exported SCOPE field; inspected `cc_worker` passes literal zero to `metrics_ul` | unresolved | 250 ms | none | Treat local physical meaning as unverified, not a received-power comparator |
| 19 | ul_sinr | G AODT_NATIVE_EQUIVALENT_UNVERIFIED | scheduled PUSCH `chest_res.snr_db` cumulative mean; NaN→0; valid floor 0.1 | dB | 250 ms | `preEqSinr` closest; `postEqSinr` plausible | Do not rename or compare distributions until estimator and aggregation equivalence are proven |
| 20 | phr | F NOT_AVAILABLE_IN_AODT | MAC power-headroom report | source encoding unresolved | report window | none | Do not infer from UE power |
| 21 | sum_requested_prbs | F NOT_AVAILABLE_IN_AODT | sum of requested/assigned PRBs | PRB | report window; reset | none | Granted resources cannot substitute for requests |
| 22 | sum_granted_prbs | D DERIVED_DURING_PREPROCESSING | sum of actually allocated PRBs | PRB | report window; reset | sum `nPrb` / plausible | First prove LTE/NR grid and window match |
| 23 | dl_pmi | F NOT_AVAILABLE_IN_AODT | reported DL PMI | index | report window | none | Do not synthesize |
| 24 | dl_ri | F NOT_AVAILABLE_IN_AODT | reported DL RI | index | report window | `layers` / not same parameter | Scheduled layers do not establish RI |
| 25 | ul_n | F NOT_AVAILABLE_IN_AODT | SCOPE PHY `ul.n`; exact meaning undocumented | unknown | report window | none | Recover source definition or exclude physical interpretation |
| 26 | ul_turbo_iters | F NOT_AVAILABLE_IN_AODT | cumulative mean PUSCH turbo iterations | count | 250 ms | none | Decode outcome is not iteration count |
| 27 | IMSI | C EXPERIMENT_METADATA | subscriber identifier | identifier | constant per UE | explicit UE-ID map | Preserve a deterministic mapping |
| 28 | RNTI | C EXPERIMENT_METADATA | runtime LTE RNTI | identifier | connection/runtime | AODT `ue_id` / unverified | Do not equate identifiers without a manifest |
| 29 | slicing_enabled | B RAN_CONTROL_CONFIGURATION | SCOPE slicing flag | boolean | config value | none | Contract-only field unless native slicing support is established |
| 30 | slice_id | B RAN_CONTROL_CONFIGURATION | SCOPE slice assignment | class ID | constant/config | external metadata only | Never claim external labels are native AODT slices |
| 31 | scheduling_policy | B RAN_CONTROL_CONFIGURATION | 0 RR, 1 WF, 2 PF | class ID | config value | cuMAC native scheduler / not selectable in inspected client | Block strict match or explicitly reframe study |

## ColO-RAN configuration reconstruction

### Verified family-level facts

- Topology: 7 BS, 42 UE, 6 UE/BS, 3 slices, 2 UE/BS/slice.
- Placement: `rome_static_medium`, static, uniform within 50 m of the associated BS.
- Radio: LTE FDD, DL 0.98 GHz, UL 1.02 GHz, 10 MHz, 50 PRBs.
- Scheduling: codes 0/1/2 correspond to RR/WF/PF.
- Slicing allocations are expressed as RBG tuples, each summing to 17 RBG: `tr0=(2,13,2)`, `tr1=(4,11,2)`, `tr2=(6,9,2)`, `tr3=(8,7,2)`, `tr4=(10,5,2)`, `tr5=(12,3,2)`, `tr6=(14,1,2)`, `tr7=(2,11,4)`, `tr8=(4,9,4)`, `tr9=(6,7,4)`, `tr10=(8,5,4)`, `tr11=(10,3,4)`, `tr12=(12,1,4)`, `tr13=(2,9,6)`, `tr14=(4,7,6)`, `tr15=(6,5,6)`, `tr16=(8,3,6)`, `tr17=(10,1,6)`, `tr18=(2,7,8)`, `tr19=(4,5,8)`, `tr20=(6,3,8)`, `tr21=(8,1,8)`, `tr22=(2,5,10)`, `tr23=(4,3,10)`, `tr24=(6,1,10)`, `tr25=(2,3,12)`, `tr26=(4,1,12)`, `tr27=(2,1,14)`.
- Nominal KPI reporting period: 250 ms, supported by SCOPE `metrics_period_secs=0.25` and public raw timestamp spacing.
- The inspected SCOPE Colosseum eNB configuration sets `pusch_max_mcs=16` and leaves `pdsch_max_mcs` commented/default; generic scheduler fallback is 28. Because local UL values reach 22, this configuration cannot be asserted as the exact runtime configuration of the local package.
- SCOPE initializes a UE power multiplier to 1.0, reads it from a per-slice UE configuration, substitutes 1.0 for zero, and applies it as the DL `rho_a` amplitude multiplier. Its channel/power estimation path divides linear transmit power by the multiplier squared. This is not equivalent to setting an AODT RU or UE power in dBm.

### Traffic discrepancy

The paper's slice-based profile reports eMBB 4 Mbit/s/UE CBR, MTC 44.6 kbit/s/UE Poisson, and URLLC 89.3 kbit/s/UE Poisson. The repository README instead describes eMBB 4 Mbps CBR, MTC 30 packets/s × 125 bytes (30 kbit/s), and URLLC 10 packets/s × 125 bytes (10 kbit/s). The executable traffic-generator configurations were not found in the inspected artifacts. Exact packet size, inter-arrival process, and rate are therefore `UNRESOLVED_COLORAN_CONFIG`; the candidate does not choose whichever description is convenient.

### Other unresolved fields

- Exact RF scenario numeric ID and exact BS/UE coordinates.
- Per-run transmit-power multiplier and its conversion to absolute power.
- Exact origin/transformation that added `traffic_type` and changed 42/7/exp1+ into 50/8/exp0.
- Checkpoint-to-dataset hash manifest.

## Publication consequence

The prior RF1017 topology and geometry must be removed from all matched-configuration tables. A new AODT scene may be called only `GIS_GROUNDED_APPROXIMATE_COLORAN_ROME`; it cannot be described as exact ColO-RAN geometry. A one-cell pilot remains technically useful only after the blocking baseline identity and traffic settings are resolved or explicitly reframed as a contract-compatibility pilot rather than a matched reproduction.

## Evidence locations

- Local evaluation: `${REPO_ROOT}\results\colosseum_advtabddpm_eval_20260712T102059Z_6e2c765b9f75`
- Official dataset repository snapshot: commit `bd86629d07d5fbfb778ebe3afd9d0b05e5191c6b`
- SCOPE source snapshot: commit `10a8a77360eff8ba7b097f0e59083781d774491b`
- Installed AODT client source: commit `f3378f57b2905a154d8b4b9d3a7d4b2483ab28a3` (dirty worktree; no claim that the commit is a signed 1.5.1 release tag)

Final recommendation: `WAIT_FOR_BLOCKING_COLORAN_CONFIG`.
