# Stage reference

| Stage | Output class | Online? |
|---|---|---:|
| 00B–05B | evidence, inputs, scene/topology/config, preflight | no |
| 06B–07B | AODT execution and raw collection | yes |
| 08B | static raw validation | no |
| 09B–10B | derived trace and optional legacy AdvTabDDPM example adapter | no |
| 11B–13B | bundled AdvTabDDPM workflow on the self-hosted AI Server | remote/GPU |
| 00BU–05BU | UAV evidence, GPX/topology/config, preflight | no |
| 06BU–07BU | mobile AODT execution and raw collection | yes |
| 08BU | UAV raw/trajectory validation | no |
| 09BU | no-path-aware AODT CIR channel trace | no |
| 10BU | strict 54-UE admission for the optional legacy AdvTabDDPM example | no |
| 11BU | upload admitted data and prepare an isolated bundled-project copy | remote |
| 12BU | train/evaluate with the unchanged downstream command | remote/GPU |
| 13BU | download the immutable evaluation package and checksum | remote |
| AERPAW BU00–BU03 | legacy audit, source registry, measurement normalization, geospatial contract | no |
| AERPAW BU04–BU08 | scene/trajectory/config contracts, formal gate, immutable bundle | optional scene build/upload |
| AERPAW BU09–BU10 | engineering smoke and full AODT EM replay | yes/GPU |
| AERPAW BU11–BU13 | native RSS semantics, alignment, and physical validation | no |
| AERPAW BU14–BU15 | pinned 32-column adapter and frozen AdvTabDDPM consumer | explicit command/GPU as needed |

The `static` and `uav` directories under `scripts/remote` each contain fourteen
stage entry points: 00 through 13 inclusive. A 10BU admission failure is a
terminal scientific gate; 11BU cannot consume outage-imputed or RF-floor data.

The additional `uav_aodt` directory contains sixteen AERPAW Lake Wheeler stage
entry points, `BU00` through `BU15`. It is intentionally separate from the
general `uav` track and requires explicit immutable upstream directories.
