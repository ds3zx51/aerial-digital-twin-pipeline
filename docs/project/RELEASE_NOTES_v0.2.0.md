# aerial-digital-twin-pipeline v0.2.0

This release adds the AERPAW Lake Wheeler validation workflow from `BU00`
through `BU15` while retaining the existing static and general UAV tracks.

## Highlights

- End-to-end fail-closed stages for source registration, dual-USRP measurement
  normalization, geospatial contracts, scene and trajectory preparation, AODT
  EM replay, native RSS parsing, physical validation, dataset adaptation, and
  frozen-consumer verification.
- Immutable per-stage outputs with explicit upstream run selection, SHA-256
  inventories, manifests, summaries, and logs.
- A pinned Lake Wheeler OpenStreetMap smoke-environment snapshot with ODbL
  attribution and a reproducibility manifest.
- A formal gate that prevents the OSM-only smoke profile from being presented
  as the paper's verified terrain and foliage reconstruction.
- Twenty offline contract tests covering source hashes, tower/trajectory
  semantics, AODT configuration, no-path handling, alignment, calibration
  separation, schema provenance, and frozen-source behavior.

## Release boundaries

Raw AERPAW Dataset 21 measurements, credentials, generated experiment results,
private deployment settings, and formal AODT scene bundles are not included.
The bundled AdvTabDDPM test snapshot retains its separate licensing boundary;
credit for its earlier upstream implementation belongs to `@0425Ning`.

See `docs/BU_AERPAW_RUNBOOK.md` for prerequisites, commands, scientific gates,
and the currently required external inputs.
