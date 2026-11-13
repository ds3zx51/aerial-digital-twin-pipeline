# AERPAW Lake Wheeler BU00–BU15

This code directory is the fail-closed replacement for the legacy UAV BU branch.
It keeps the original scripts untouched and separates measured evidence, geospatial conversion,
AODT-native output, calibration, comparison, and the frozen 32-column AdvTabDDPM consumer.

The formal execution order is:

`BU00 -> BU01 -> ... -> BU07 -> BU08 -> BU09 -> BU10 -> BU11 -> BU12 -> BU13 -> BU14 -> BU15`

Every stage creates a unique immutable result directory and writes `run_manifest.json`,
`summary.json`, `stdout.log`, `stderr.log`, `input_sha256.txt`, and `output_sha256.txt`.
Stages never discover "the latest" upstream run. Supply each upstream run explicitly.

## Paper-controlled baseline

The baseline is the final journal paper *Analysis and Prediction of Coverage and Channel Rank
for UAV Networks in Rural Scenarios With Foliage* (IEEE OJVT, DOI
`10.1109/OJVT.2025.3585537`). The coverage experiment uses 3.3 GHz SISO, 44 dBm, a UAV
transmitter and LW1–LW5 receivers. The generic ray-tracing table's 3.4 GHz 4x4 MIMO setup is
recorded but is not silently substituted for the coverage profile.

The repository does not contain the paper's exact supplementary foliage GeoJSON nor a verified
AODT USD scene bundle containing it. Therefore `paper_reconstruction` must stop at BU04/BU07
until those artifacts are supplied. `osm_only_smoke` is useful for plumbing tests only and can
never pass the formal gate.

## Quick start (local, no remote execution)

Use the wrappers in `scripts\remote\uav_aodt`; implementation code lives in
`scripts\aodt\bu_aerpaw`, configuration in `configs\aerpaw_lake_wheeler`, scene contracts in
`scenes\aerpaw_lake_wheeler`, documentation in `docs`, and tests in `tests`.

```powershell
scripts\remote\uav_aodt\00BU_legacy_audit.bat
scripts\remote\uav_aodt\01BU_source_registry.bat --source C:\path\to\Dataset21.zip
```

Use `--help` for stage-specific options. BU09, BU10 and BU15 require both `--execute` and an
explicit command. Merely running a stage never starts AODT or training.
