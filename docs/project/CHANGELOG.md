# Changelog

## 0.2.0 - 2026-08-19

- Added the separate AERPAW Lake Wheeler `BU00`–`BU15` fail-closed workflow.
- Added an auditable OpenStreetMap smoke-environment snapshot, source hash,
  bounding box, ODbL attribution, and AODT handoff metadata.
- Added measurement normalization, geospatial and scene contracts, immutable
  stage manifests, native RSS parsing, physical validation, a pinned
  32-column adapter boundary, and a frozen AdvTabDDPM consumer check.
- Added 20 offline contract tests for the new AERPAW workflow.
- Kept raw Dataset 21 measurements, credentials, generated results, and formal
  scene bundles outside the release.

## Unreleased

- Prepared an AODT-only source-release layout.
- Restored AODT-native static stage dependencies and portable configuration.
- Limited the public UAV workflow to `00BU`–`08BU`.
- Classified the historical AI-contract `ul_sinr` as an EM proxy and prohibited
  physical cross-backend distribution claims.
- Added CI, repository hygiene checks, documentation, and third-party/data
  boundaries.
