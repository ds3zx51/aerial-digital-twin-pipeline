# Configuration directory

All committed experiment definitions use UTF-8 text formats:

- [`scenario_layout_tokyo_docomo.json`](scenario_layout_tokyo_docomo.json): checked-in static reference profile.
- [`scenarios/static/`](scenarios/static/README.md): new static profile JSON files; always pass `--scenario`.
- [`scenarios/bu/`](scenarios/bu/): UAV profile JSON files.
- [`aodt/`](aodt/): YAML mappings to AODT Worker-side JSON assets.
- [`geospatial/`](geospatial/): scene anchors, units, bounds, and coordinate contracts in JSON.
- [`trajectories/shared/`](trajectories/shared/): trajectory design and canonical ENU JSON.
- [`gpx/shared/`](gpx/shared/): GPX 1.1 route files and hash manifests.

Scene ZIP files do not belong here. Store local authorized scene archives in
the ignored `assets/scenes/<scene_id>/` directory and provision their expanded
copies separately on the AODT Worker.

Open the complete [English placement and format guide](../docs/CONFIG_AND_SCENES.md)
or [繁體中文規範](../docs/CONFIG_AND_SCENES.zh-TW.md) before adding a profile.
Do not overwrite an existing reference profile.
