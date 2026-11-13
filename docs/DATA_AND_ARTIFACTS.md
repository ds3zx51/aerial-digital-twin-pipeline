# Data and artifact policy

The source repository includes pipeline code, the deployable AdvTabDDPM test
project with its fixed checkpoints, small deterministic trajectory artifacts,
configuration contracts, and lightweight evidence.

Excluded by default:

- `results/` and `logs/`;
- raw CIR/telemetry tensors;
- Colosseum datasets;
- generated AdvTabDDPM pickle/NumPy datasets and new training outputs;
- AODT runtime files and cloud credentials; and
- Tokyo OpenUSD/XML/PLY scene geometry and derivatives.

Store large reproducibility artifacts in a separately governed archive with a
manifest containing SHA-256, byte size, media type, source run ID, creation
timestamp, applicable license/access terms, and the source Git commit. Never
assume that generated scene/model/data artifacts inherit the project source
license.

Checked-in AI checkpoints are enumerated with hashes in
`ai_server/model_manifest.csv`; they are inputs, not new experimental results.
The checked-in geospatial JSON is metadata needed to validate coordinates; it
does not include or grant rights to the underlying scene.
