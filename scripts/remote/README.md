# Stage entry points

The remote launchers are separated by experiment track and expose one file per stage:

- `static/00B_inspect_inputs.bat` through `static/13B_download_results.bat`
- `uav/00BU_inspect_inputs.bat` through `uav/13BU_download_results.bat`
- `uav_aodt/00BU_legacy_audit.bat` through `uav_aodt/15BU_frozen_consumer.bat`

Every filename uses the old stage-oriented convention: numeric stage prefix
followed by a short action name. Static and UAV use matching action names where
their semantics are the same; stage 01 differs because UAV validates its GPX
inputs while static prepares the locked scenario inputs.

Stages 00-10 invoke local validation or post-processing code. Stages 11-13 use
the self-hosted AI Server configured in `remote_settings.bat`. Copy
`remote_settings.example.bat` to that ignored filename and fill in your own
values. `REMOTE_HOST` is the AI Server SSH IP/DNS name and `REMOTE_PORT` is its
SSH port (normally 22). They are unrelated to the AODT Worker gRPC endpoint.

The repository must already be cloned at `REMOTE_ROOT` on the AI Server, and
`REMOTE_PYTHON` must point to the installed AdvTabDDPM environment. Stage 11
uploads only the admitted AODT dataset and invokes
`ai_server/deploy/aodt_advtabddpm_remote.sh`; stage scripts contain no fixed IP.

The per-stage entry points reuse shared Python modules, but no entry point silently
advances into the next numbered stage. A failing 10BU admission cannot be uploaded by
11BU, and no outage/RF-floor imputation is accepted.

The `uav_aodt` directory is the separate AERPAW Lake Wheeler evidence pipeline.
It requires explicit immutable upstream run mappings and fails closed when formal
scene or foliage evidence is unavailable. See `docs/BU_AERPAW_RUNBOOK.md`.
