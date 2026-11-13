# Reproducibility

## Invariants

- A run gets a new UTC timestamped directory.
- Existing stage/result directories are never reused or overwritten.
- Input, artifact, and adapter files are hashed where the workflow consumes
  them.
- Online stages fail closed when endpoint, credentials, scene, worker assets,
  or acknowledgements are missing.
- CSV rows represent UE/slot observations, not independent experimental runs.

## Offline checks

```powershell
python -m unittest discover -s tests -v
python tools\validate_shared_multi_uav_gpx.py
python tools\validate_bu_scenario.py configs\scenarios\bu\tokyo_BU_multi_uav.json
```

Integration tests require `AODT_CLIENT_ROOT` and `AODT_SCENE_ARCHIVE`.
Ordinary CI never starts AODT, contacts a worker, or launches a GPU
AdvTabDDPM job.

## External software

Record the AODT checkout commit, worker/runtime version, scene identifier,
Python version, device type, environment variables by name (never secret
values), and source result paths for each experiment. The AdvTabDDPM test
snapshot is checked in under `ai_server/AdvTabDDPM`; preserve the config and
checkpoint hashes from `ai_server/model_manifest.csv` in result metadata.

## Coordinate units

Scenario coordinates are OpenUSD stage units. For the frozen Tokyo metadata,
`meters_per_unit` is 0.01. Position validation applies this conversion before
comparing static node coordinates to AODT-reported metres. Do not label raw
scenario coordinates as physical metres without conversion.
