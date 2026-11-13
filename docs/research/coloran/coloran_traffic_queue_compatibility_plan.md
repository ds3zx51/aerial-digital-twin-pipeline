# ColO-RAN traffic and queue compatibility plan

Status: design only. Nothing is implemented or executed.

## Native capability finding

The installed AODT 1.5.1 client schema and `worker/config_ran.json` expose PHY/MAC scheduling configuration and scheduled transport-block telemetry, but no per-UE packet-arrival API, offered-load profile, DL/UL queue, packet counter, requested-PRB counter, or traffic-class control. `ran_metric.py` derives BLER and throughput after simulation from `tbs` and `outcome`; it is not a traffic generator or scheduler queue.

Consequently, the currently feasible design is classified `POST_HOC_TRAFFIC_COMPATIBILITY_MODEL`. It is not full ColO-RAN equivalence because its queues cannot change which UE cuMAC schedules, how many PRBs it grants, or which MCS it selects.

`CLOSED_LOOP_TRAFFIC_SCHEDULER_INTEGRATION` remains a design option only. No public client API was found for injecting per-slot buffer status or arrival state into cuMAC.

## Unresolved source traffic

Evidence priority is fixed before results:

1. Runtime TGEN/job configuration tied to the exact 50 local CSVs.
2. Public dataset README.
3. Paper Table 2.

The exact runtime configuration was not recovered. eMBB is consistently 4 Mbit/s/UE CBR. MTC and URLLC are `SOURCE_CONFLICT_REQUIRES_RESOLUTION`: the README states 30 packets/s × 125 bytes and 10 packets/s × 125 bytes, while the paper states 44.6 and 89.3 kbit/s/UE Poisson. No implementation may silently choose between them.

## Deterministic external model design

Inputs per run: immutable UE/traffic assignment, selected resolved traffic profile, packet size distribution, arrival process, seed, slot duration, and raw AODT telemetry. Preserve the profile source and hash.

For each UE and direction maintain an ordered byte queue. At each slot:

1. Generate arrivals from the preregistered process. CBR uses deterministic inter-arrival times; Poisson uses an exponential inter-arrival process from a documented counter-based RNG stream per UE.
2. Append packet IDs, arrival timestamps, and byte sizes.
3. Read AODT scheduled `tbs`, `rv`, and `outcome` for that UE/link.
4. For a successful record, drain at most `tbs` bytes in FIFO order; for failure, drain zero unless a separately validated HARQ state model says otherwise.
5. Record arrivals, queued bytes, attempted TBS, successful bytes, failures, drops, and latency.

Required output lineage: run ID, UE ID, traffic type, slice metadata, profile/version/hash, RNG seed/stream, raw telemetry row key, queue state before/after, and whether the model was post hoc.

## Prohibited claims and values

- Never infer `traffic_type` from SINR, MCS, throughput, or errors.
- Never write intended slice allocations into `slice_prb` unless AODT actually enforced them.
- Never convert TBS records directly into packet counts without packet-boundary state.
- Never call post-hoc queue occupancy a native AODT buffer KPI.
- Never use ColO-RAN KPI values to initialize or correct AODT queues.

## Closed-loop feasibility gate

Reclassify to `CLOSED_LOOP_TRAFFIC_SCHEDULER_INTEGRATION` only if a versioned worker API is found that accepts per-UE arrivals/buffer state before scheduling and if readback proves the state affected grants. The gate requires source/API evidence, a one-cell test, and preserved input/output traces; ECDF similarity is not evidence.
