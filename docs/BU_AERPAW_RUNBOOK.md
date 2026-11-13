# BU AERPAW Lake Wheeler runbook

All commands run from the repository's `scripts\remote\uav_aodt` directory.
Every downstream stage takes explicit `--input BUxx=<immutable result directory>` mappings.
Never substitute a newest/latest directory. `RESULT_NAME` below is a prefix; the runner appends
UTC time and a UUID and refuses overwrite.

Wrappers use `BU_PYTHON`, then `AODT_PYTHON`, then `python` from `PATH`. Set
`BU_PYTHON` explicitly when the required packages are installed in another environment.
BU02/BU03/BU13 require the packages in `scripts\aodt\bu_aerpaw\requirements.txt`.

| Stage | Purpose | Command skeleton | Inputs | Outputs / expected RESULT_NAME | Formal gate | GPU/AODT | Internet | Common failure |
|---|---|---|---|---|---|---|---|---|
| BU00 | Audit/freeze legacy state | `00BU_legacy_audit.bat` | repository | legacy inventory; `bu00_legacy_audit_*` | never modifies legacy | no | no | repository missing |
| BU01 | Hash/register AERPAW sources | `01BU_source_registry.bat --source <Dataset21.zip>` | raw source ZIP/dir | source registry; `bu01_source_registry_*` | raw MAT for LW1–LW5 | no | no | incomplete/missing source |
| BU02 | Normalize raw dual-USRP data | `02BU_measurement_normalization.bat --input BU01=<run>` | pinned BU01 | raw measurements, 917 GPS and 3530-clock route; `bu02_measurement_normalization_*` | exact counts, finite, no calibration | no | no | MAT schema/count drift |
| BU03 | Define WGS84/local ENU contract | `03BU_geospatial_contract.bat --input BU02=<run>` | pinned BU02 | geospatial contract; `bu03_geospatial_contract_*` | deterministic round trip; formal vertical datum deferred to scene | no | no | round-trip failure |
| BU04 | Register/build scene profile | smoke: `04BU_scene_contract.bat --input BU03=<run> --scene-profile osm_only_smoke`; formal: add `--scene-bundle <dir> --foliage-geojson <file>` | pinned BU03 for Lake Wheeler OSM AOI; formal scene bundle/foliage for publication profile | registration/request; `bu04_scene_contract_*` | nonempty foliage + CRS/origin/axes/vertical datum/terrain | scene tooling only with builder `--execute` | OSM/terrain download only when executing | missing/empty/template foliage or scene manifest |
| BU05 | Build canonical GPX and 3D route | `05BU_trajectory_builder.bat --input BU02=<run> --input BU04=<run>` | pinned BU02/BU04 | CSV waypoints + audit GPX; `bu05_trajectory_builder_*` | 3530 samples and formal terrain heights | no | no | incomplete terrain samples |
| BU06 | Build AODT 1.5.1 EM config | `06BU_aodt_config_builder.bat --input BU04=<run> --input BU05=<run>` | pinned BU04/BU05 | execution spec/config manifest; `bu06_aodt_config_builder_*` | EM, 3.3 GHz, SISO, 5 RU + 1 UAV, 44 dBm | no for spec; native binding for final config | no | scene URL unresolved/profile drift |
| BU07 | Formal pre-flight gate | `07BU_formal_gate.bat --input BU01=<run> ... --input BU06=<run>` | pinned BU01–BU06 | gate report; `bu07_formal_gate_*` | every formal check passes | no | no | smoke/candidate/Tokyo/placeholder detected |
| BU08 | Package/upload immutable bundle | `08BU_immutable_bundle.bat --input BU01=<run> ... --input BU07=<run>` | pinned BU01–BU07 | tar.gz, hashes, tower table; `bu08_immutable_bundle_*` | passed BU07 | no; upload optional with `--execute` | only for actual upload | destination exists or formal gate absent |
| BU09 | Engineering AODT EM smoke | `09BU_aodt_em_smoke.bat --input BU08=<run> --execute --smoke-evidence <json> --command <remote command>` | pinned BU08, representative route subset | smoke evidence; `bu09_aodt_em_smoke_*` | all eight smoke checks; never publication evidence | yes | yes | five links/altitude/export/readback failure |
| BU10 | Full AODT EM replay | `10BU_full_aodt_em_replay.bat --input BU08=<run> --input BU09=<run> --execute --native-output <path> --command <remote command>` | pinned BU08/BU09 | native output registry; `bu10_full_aodt_em_replay_*` | smoke passed and native evidence preserved | yes | yes | worker/scene/channel export failure |
| BU11 | Derive measurement-comparable RSS | `11BU_native_rss_parser.bat --input BU10=<run> --native-output <csv> --semantic-contract <json>` | pinned BU10 + actual native schema declaration | raw RSS, provenance, semantic audit; `bu11_rss_extraction_*` | power/antenna/endpoint/no-path semantics explicit | no | no | guessed/incomplete semantics or position readback absent |
| BU12 | Outer-align AODT and dual USRP | `12BU_measurement_alignment.bat --input BU02=<run> --input BU11=<run>` | pinned BU02/BU11 | matched + both unmatched tables; `bu12_measurement_alignment_*` | every row retained/reason-coded | no | no | invalid tolerance/schema |
| BU13 | Physical validation/Fig. 15 views | `13BU_physical_validation.bat --input BU12=<run>` | pinned BU12 | three plot views, CDF, metrics, claims; `bu13_physical_validation_*` | methodological completeness only; no arbitrary accuracy threshold | no | no | too few matched samples/dependency missing |
| BU14 | Bind validated channel to legacy contract | `14BU_advtabddpm_adapter.bat --input BU13=<run> --pinned-bu13-run-id <id> --execute --adapter-output <csv> --command <adapter command>` | explicitly pinned BU13 | exact 32-column dataset + field provenance; `bu14_advtabddpm_adapter_*` | exact schema and source class for every field | no | no | unpinned input/schema drift |
| BU15 | Run frozen AdvTabDDPM consumer | `15BU_frozen_consumer.bat --input BU14=<run> --pinned-bu14-run-id <id> --execute --command <consumer command>` | explicitly pinned BU14 | consumer logs/core before-after hashes; `bu15_frozen_consumer_*` | AdvTabDDPM source tree unchanged | training may require GPU | no unless command does | source mutation/unpinned input/consumer failure |

## Parameter decision

- Formal coverage config: final-paper coverage override, 3.3 GHz SISO, 44 dBm.
- Physical link: UAV transmitter to LW1–LW5 receivers. AODT maps reciprocal tower RUs to one
  UAV UE and records the reciprocity assumption.
- Route clock: 3530 LW1 measurement timestamps. The 917 raw GPS samples remain separately
  preserved. Raw altitude is unchanged; simulation height is `max(raw, 0.5 m)`.
- GPX is audit/visualization only. AODT execution uses explicit 3D georeferenced waypoints.
- Scene: 1080 x 2130 m, tower height 10 m AGL, maximum two reflections, scattering and
  diffraction disabled, paper materials and foliage constants recorded in the parameter contract.
- OSM smoke AOI is derived from the raw Dataset21 route plus LW1–LW5 in local ENU and centered
  inside the paper's 1080 x 2130 m dimensions. A slippy-map URL's display center is never used as
  experiment coordinates. Source attribution: OpenStreetMap contributors.
- AODT CIR power already includes configured RU transmit power under the accepted semantic
  contract. BU11 adds 0 dB and represents no-path as null.

## CURRENTLY MISSING EXTERNAL INPUTS

The public release does not include the Dataset 21 ZIP with raw dual-USRP MAT files for
LW1–LW5. Obtain it from an authorized source and register it explicitly at BU01. The release
also does not include:

1. The paper's supplementary/manual foliage GeoJSON (the included empty file is a template and
   intentionally fails the formal gate).
2. A verified AODT Lake Wheeler USD scene bundle incorporating terrain, buildings, OSM forest
   trees and the paper's satellite-guided supplemental foliage.
3. A formal scene manifest proving CRS, origin, axis order, units, vertical datum, scene hash,
   dependency hashes, and terrain elevations for all 3530 UAV points plus the five towers.
4. The exact paper antenna radiation pattern. The baseline explicitly uses a SISO isotropic
   model and does not claim the unavailable pattern is reproduced.
5. After AODT execution, a native-schema semantic declaration/readback proving power units,
   antenna inclusion, transmit-power inclusion, endpoint mapping, timestamps and no-path encoding.

The public AERPAW scene candidate in `assets/external` is useful only for smoke work: it is
not verified as the paper's AODT scene, lacks LW4/LW5 tower meshes, and is not a substitute for
the missing formal foliage/terrain evidence.

Consequently a formal AODT config and formal terrain-resolved 3D trajectory cannot yet be
generated. BU00–BU03 and a BU04 `osm_only_smoke` request can run locally; BU07 will correctly
block until the external scene evidence is supplied.
