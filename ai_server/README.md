# Self-hosted AI Server

This directory contains one deployable AI-side reference application used by
stages `11B`-`13B` and `11BU`-`13BU`. AdvTabDDPM is an example, not a required
AI backend. Developers may replace it with another AI project plus the adapter,
environment, runner, and provenance manifest required by that project.

The bundled reference snapshot's exact runtime, dependency, configuration,
provenance status, and integrity hashes are documented in
[`VERSION.md`](VERSION.md) and [`version_manifest.json`](version_manifest.json).
It is identified as `aodt-advtabddpm-reference-2026.08`; this is a local
integration profile, not an official upstream AdvTabDDPM release number.

## Contents

- `AdvTabDDPM/`: test-project source, fixed classifier checkpoint, diffusion
  checkpoints, and pinned Python requirements.
- `deploy/`: dataset materializer and the remote prepare/train/evaluate runner.
- `environment.yml`: reference Conda environment for the self-hosted GPU host.
- `model_manifest.csv`: SHA-256 and byte sizes for the checked-in checkpoints.
- `VERSION.md` / `version_manifest.json`: human- and machine-readable version
  specifications for the reference example.

Generated datasets and experiment results are not part of this directory.
Stage 11 uploads the passing 10B/10BU dataset to the AI Server; it does not
alter the checked-in model configuration or classifier threshold.

The project must be cloned on the AI Server at the path configured as
`REMOTE_ROOT`. Installation and IP configuration are documented in the root
README.
