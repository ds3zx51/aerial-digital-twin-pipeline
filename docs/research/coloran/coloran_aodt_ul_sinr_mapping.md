# ColO-RAN to AODT UL SINR mapping

Status: semantic audit only; no ECDF-based selection and no implementation change.

## ColO-RAN/SCOPE chain

For an actual scheduled UL grant, SCOPE decodes PUSCH, reads `enb_ul.chest_res.snr_db`, and passes it to `metrics_ul`. When PUSCH data exists, `metrics_ul` cumulatively averages SINR, MCS, RSSI argument, and turbo iterations using `ul.n_samples`. Metrics are copied and worker state is zeroed at each report retrieval. The configured reporting period is 0.25 s. CSV serialization maps NaN SINR to 0 and otherwise writes `max(0.1, value)`.

Thus the published `ul_sinr` column is a scheduled-PUSCH, channel-estimator `snr_db` window statistic plus export censoring. The internal name is SNR, so exact interference-estimator semantics are implementation-specific even though the exported column says SINR.

## AODT candidates

| AODT source | Classification | Decision |
|---|---|---|
| `preEqSinr` | `VERIFIED_CLOSEST_MEASUREMENT` | Closest native scheduled-resource measurement boundary. This does not mean semantic equivalence. |
| `postEqSinr` | `PLAUSIBLE_BUT_UNVERIFIED` | Native scheduled-resource SINR after equalization; measurement point differs from the closest interpretation of the SCOPE channel-estimator value. |
| legacy EM-derived all-UE interference proxy | `NOT_COMPARABLE` | No actual PRB/activity scheduling and not native RAN telemetry. Retain only as `aodt_em_proxy_ul_sinr_db`. |

The installed worker confirms `pusch channel estimation: MMSE`, but the client repository does not expose the estimator formula, resource-element weighting, layer combining, or exact derivation of pre/post-EQ fields. The following remain blocking: inclusion of failed decodes and retransmissions, PRB/layer weighting, arithmetic-dB versus linear averaging, time-window alignment, and LTE-versus-NR numerology.

No AODT field may be written directly into the ColO-RAN `ul_sinr` column yet. Raw pre/post-EQ telemetry must be preserved and any candidate aggregate must have a distinct name until validation closes these questions.
