# Static scenario profiles

Store new static scenario definitions here as UTF-8 JSON:

```text
configs/scenarios/static/<scenario_id>.json
```

Pass the file with `--scenario`. The checked-in Tokyo reference remains at
`configs/scenario_layout_tokyo_docomo.json` for reproducibility. A different
scene or topology requires its own validator, geospatial contract, provenance,
and downstream profile; adding a JSON file alone is not end-to-end admission.
