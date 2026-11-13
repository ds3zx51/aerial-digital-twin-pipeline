# Third-party notices

This repository is an independent integration workflow. Third-party components
retain their own copyright and license terms.

The repository's root Apache-2.0 license applies only to project-authored
material except where otherwise noted. It does not override this table.

| Component | Source | Release boundary |
|---|---|---|
| NVIDIA Aerial Omniverse Digital Twin | <https://github.com/NVIDIA/aerial-omniverse-digital-twin> | External checkout/runtime. The upstream repository identifies NVIDIA copyright and Apache License 2.0. NVIDIA credentials, cloud services, and scene assets are not included. Preserve upstream copyright, license, trademark, and attribution notices whenever upstream material is redistributed. |
| AdvTabDDPM test snapshot | <https://github.com/0425Ning/AdvTabDDPM> | Source plus four fixed checkpoints are included under `ai_server/` for this test workflow. The reviewed snapshot contains no standalone license file; public redistribution must remain blocked until the copyright holders confirm an applicable license. Generated data/results are excluded. |
| Colosseum datasets and RF scenarios | <https://www.colosseum.net/> | Not included. Obtain from an authorized source and follow its terms. |
| Tokyo AODT scene | User/NVIDIA-provided research input | Geometry and mesh derivatives are omitted pending redistribution authorization. Metadata does not grant rights to the scene. |
| CellMapper/map evidence | <https://www.cellmapper.net/> | Screenshots/source artifacts are not included. The frozen topology must not be represented as authoritative geographic ground truth. |
| OpenStreetMap Lake Wheeler smoke snapshot | <https://www.openstreetmap.org/> | The `.osm` source under `scenes/aerpaw_lake_wheeler/source/osm/` is © OpenStreetMap contributors and available under ODbL 1.0. Preserve attribution and the accompanying source manifest. It is a smoke-test input, not formal paper-scene evidence. |

Generated results, raw CIR tensors, and new model-training outputs are excluded.
The four intentionally checked-in AI inputs are enumerated in
`ai_server/model_manifest.csv`.
