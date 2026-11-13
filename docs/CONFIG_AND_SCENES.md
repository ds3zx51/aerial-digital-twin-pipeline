# Configuration and scene placement

This document defines where each experiment configuration and scene input
belongs, which format is accepted, and whether it is stored in Git. Paths under
`configs/` are relative to the repository root.

## Quick placement table

| Input | Location | Format | Commit to Git? |
|---|---|---|---|
| Static experiment scenario | `configs/scenario_layout_tokyo_docomo.json` | UTF-8 JSON | Yes |
| UAV experiment scenario | `configs/scenarios/bu/<scenario_id>.json` | UTF-8 JSON | Yes |
| Worker asset references | `configs/aodt/<profile>_assets.yml` | UTF-8 YAML mapping | Yes |
| Geodetic scene contract | `configs/geospatial/<scene_id>_aodt_scene_contract.json` | UTF-8 JSON | Yes |
| UAV design and canonical ENU trace | `configs/trajectories/shared/<trajectory_id>_design.json` and `<trajectory_id>_enu.json` | UTF-8 JSON | Yes |
| UAV GPX and trajectory manifest | `configs/gpx/shared/<trajectory_id>/` | GPX 1.1 XML plus UTF-8 JSON manifest | Yes |
| Authorized AODT scene archive | `assets/scenes/<scene_id>/<scene_id>.zip` | ZIP containing OpenUSD and metadata | No |
| Worker-expanded scene | AODT Worker object-store prefix, for example `test_data/maps/<scene_id>/` | Expanded OpenUSD scene tree | No |
| Worker AODT asset definitions | Paths named by the asset YAML, for example `test_data/assets/*.json` | AODT Worker JSON assets | No |
| Generated results | `results/aodt_static/` or `results/aodt_uav/` | Generated artifacts | No |

Do not place credentials, vendor scene archives, raw CIR output, or generated
results under `configs/`.

## 1. Static scenario configuration

The checked-in static reference is:

```text
configs/scenario_layout_tokyo_docomo.json
```

It is a UTF-8 JSON object with exactly these top-level keys:

```json
{
  "meta": {},
  "bs": [],
  "ue": [],
  "experiment_definition": {}
}
```

The checked-in publication profile has its own locked node counts, ordered
identities, radio definition, and topology validator. The AODT adapter maps
scenario node ID `n` to AODT node ID `n + 1`. Coordinates in this JSON are
OpenUSD scene-stage units; interpret them using the recorded scene-unit
conversion rather than assuming the raw values are metres.

To inspect another JSON without overwriting the reference, put it under
`configs/scenarios/static/<scenario_id>.json` and pass its path explicitly:

```bat
scripts\remote\static\00B_inspect_inputs.bat ^
  --scenario configs\scenarios\static\<scenario_id>.json ^
  --scene-archive assets\scenes\<scene_id>\<scene_id>.zip
```

A new topology or scene is a new profile. Update its validators, geospatial
contract, trajectory bindings, and provenance together. Do not assume that
passing `--scenario` alone makes a new profile valid through stage 13.

## 2. UAV scenario and trajectory configuration

Store each UAV scenario as:

```text
configs/scenarios/bu/<scenario_id>.json
```

The bundled example is `configs/scenarios/bu/tokyo_BU_multi_uav.json`. It uses
repository-relative links to:

```text
configs/trajectories/shared/<trajectory_id>_design.json
configs/trajectories/shared/<trajectory_id>_enu.json
configs/gpx/shared/<trajectory_id>/trajectory_manifest.json
configs/gpx/shared/<trajectory_id>/uav_<id>_<trajectory_name>.gpx
```

UAV count is profile-specific. It must come from the scenario and trajectory
manifest; it is not a universal pipeline constant. The number of `aerial_ues`
entries, GPX records, canonical ENU routes, manifest records, Worker GPX
objects, and runtime node mappings must agree. A profile-specific validator
may impose an additional registered count, so changing the count requires a
new validated profile rather than adding or deleting one GPX file in place.

Each GPX file must be GPX 1.1 XML. Every `trkpt` must contain latitude,
longitude, `<ele>`, and an ISO-8601 UTC `<time>`. Sample count and interval are
declared by the profile manifest and must agree across every route. The
manifest binds the design JSON, canonical ENU JSON, GPX files, source static
scenario, geodetic contract, and derived artifacts with SHA-256 hashes.

Never edit only one GPX or hand-update one hash. Regenerate the entire bound
payload in staging, validate it, review the diff, and publish it together:

```powershell
python tools\generate_shared_multi_uav_gpx.py --project-root . --staging-root "C:\path\to\empty-staging-dir"
python tools\validate_shared_multi_uav_gpx.py --artifact-root "C:\path\to\empty-staging-dir" --reference-project-root .
python tools\validate_bu_scenario.py configs\scenarios\bu\<scenario_id>.json
```

The bundled generator is an example-profile generator. For a new UAV count or
trajectory design, create or update the generator and validator as part of the
new profile. It does not start AODT or an AI job.

## 3. Local scene archive

Place a lawfully obtained scene archive under the ignored local directory:

```text
assets/
  scenes/
    <scene_id>/
      <scene_id>.zip
```

Set its absolute path in the Client shell:

```powershell
$env:AODT_SCENE_ARCHIVE = "$PWD\assets\scenes\<scene_id>\<scene_id>.zip"
```

For the checked-in Tokyo profile, the ZIP must contain these exact entries at
minimum, without an extra enclosing directory:

```text
tokyo/sim/master.usd
tokyo/sim/master_tiles/tile_0_0.usd
tokyo/sim/master_metadata.json
tokyo/sim/master_input_manifest.json
tokyo/sim/vegetation.geojson
```

It must also contain every OpenUSD layer, texture, and referenced dependency
needed to compose the scene. JSON and GeoJSON must be valid UTF-8. The pipeline
hashes and inventories the archive, but Git ignores it because scene
redistribution rights are separate from the source-code license.

## 4. Worker-side scene and AODT assets

The local ZIP is validation evidence; this repository does not automatically
upload it. Before an online run, provision the same expanded scene on the AODT
Worker using an authorized upstream AODT asset workflow. The reference Worker
prefix is `test_data/maps/tokyo`, so the Worker must resolve paths such as:

```text
test_data/maps/tokyo/sim/master.usd
test_data/maps/tokyo/sim/vegetation.geojson
```

`configs/aodt/aodt_static_assets.yml` is a YAML mapping of these Worker-side
AODT asset categories:

```yaml
du: test_data/assets/du.json
ru: test_data/assets/gnb.json
ue: test_data/assets/ue.json
materials: test_data/assets/materials.json
scatterers: test_data/assets/car_small.json
vegetation_materials: test_data/assets/vegetation_materials.json
vegetation_assets: test_data/assets/street_tree.json
scenario: test_data/assets/scenario.json
panels: test_data/assets/panel.json
```

These are Worker paths, not Windows Client or repository-relative paths.
Verify every referenced JSON before using `--ack-worker-assets-resolved`.

For a UAV run, provision every GPX listed by the selected manifest under the
Worker prefix supplied with `--worker-gpx-prefix`. Only use
`--ack-worker-gpx-resolved` after confirming that the Worker copies and
committed manifest hashes agree.

## 5. Client environment and invocation

The runners read these variables directly from the Client process:

```powershell
$env:AODT_CLIENT_ROOT = "C:\path\to\aerial-omniverse-digital-twin"
$env:AODT_SCENE_ARCHIVE = "$PWD\assets\scenes\<scene_id>\<scene_id>.zip"
$env:DT_SERVER_ADDRESS = "<AODT_WORKER_IP>:50051"
```

`.env.example` is a template only; the batch files do not automatically load
a `.env` file. Set variables in the shell, operating-system environment, or an
untracked local wrapper.

Static example:

```bat
scripts\remote\static\05B_run_preflight.bat --ack-worker-assets-resolved
scripts\remote\static\06B_run_aodt.bat --ack-worker-assets-resolved
```

UAV example:

```bat
scripts\remote\uav\05BU_run_preflight.bat --ack-worker-assets-resolved --ack-worker-gpx-resolved
scripts\remote\uav\06BU_run_aodt.bat --ack-worker-assets-resolved --ack-worker-gpx-resolved
```

Acknowledgement flags are operator assertions, not upload commands or
substitutes for inspecting the actual Worker files.
