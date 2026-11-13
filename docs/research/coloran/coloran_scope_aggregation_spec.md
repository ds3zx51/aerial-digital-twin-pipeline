# SCOPE-compatible aggregation specification

Status: preregistration candidate only; no aggregation was run.

## Source interval evidence

The interval is not inferred only from plotted timestamps. The inspected SCOPE `enb.conf` sets `metrics_period_secs=0.25`, `enb.cc` invokes `save_ue_metrics_to_csv` at that period, and public/local metrics timestamps advance nominally by 250 ms. The target window is therefore 250 ms, subject to preservation of actual timestamp boundaries and partial-window flags.

## Raw-data rule

Raw AODT telemetry is immutable. Every derived row must retain its input run ID, batch, slot range, RU, UE, link, configuration hash, worker `config_ran.json` hash, and aggregation-spec version. Do not collapse across independent batches/runs.

## Candidate windows

- Group by `(source_run_id, batch_id, ru_id, ue_id)` and non-overlapping 250-ms wall-time windows.
- Obtain slot duration from the executed RAN configuration; do not assume it from LTE or from the example script.
- Preserve scheduled and unscheduled state separately. Absence of a row is not a zero-valued PHY measurement.
- Mark first/last partial windows; exclude or include them only by a preregistered rule.
- Preserve DL and UL records separately before producing one AI-facing row.

## Formulas

- `*_n_samples`: count of preregistered included scheduled records, explicitly not claimed identical to SCOPE PHY samples.
- `*_mcs`: arithmetic mean of MCS across those records; retain count.
- candidate throughput: `8 × sum(tbs for outcome=1) / window_seconds / 1e6` Mbit/s.
- candidate TB error percentage: `100 × count(outcome=0) / count(scheduled records)`.
- `sum_granted_prbs`: sum `nPrb`; unit must be labelled NR PRB until grid equivalence is proven.
- SINR: retain both raw pre/post-EQ fields. No published `ul_sinr` is generated until estimator, inclusion, and dB-versus-linear averaging are resolved.

## Unavailable fields

Buffers, packet counts, CQI, RSSI, PHR, requested PRBs, PMI, RI, `ul_n`, and turbo iterations are unavailable or not equivalent. They remain missing with explicit reason codes. They must not be zero-filled. `slice_prb` also remains missing unless enforced by native scheduling.

The field-by-field contract is in `coloran_ai_contract_mapping.csv`.

## Validation before use

The future aggregator must test exact window membership, duplicate keys, direction separation, zero scheduled-row behavior, failed TBs, HARQ `rv`, unit labels, and deterministic reproduction. A hand-traced one-cell window is required before any full run.
