#!/usr/bin/env python3
"""Build a legacy AI contract from one successful AODT run.

The AODT serving/candidate traces are the channel source.  Missing MAC and
traffic state is produced by one deterministic, stateful queue and scheduler
model configured in the scenario. The contract preserves the historical
``ul_sinr`` column name, but its value is an EM-derived proxy, not native RAN
telemetry and not verified semantically equivalent to Colosseum UL SINR.
This program writes raw metrics CSVs and provenance only; it never writes
AdvTabDDPM PKL/NPY/model/plot artifacts.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


COLOSSEUM_COLUMNS = [
    "traffic_type", "Timestamp", "num_ues", "slice_prb", "power_multiplier",
    "dl_mcs", "dl_n_samples", "dl_buffer [bytes]", "tx_brate downlink [Mbps]",
    "tx_pkts downlink", "tx_errors downlink (%)", "dl_cqi", "ul_mcs",
    "ul_n_samples", "ul_buffer [bytes]", "rx_brate uplink [Mbps]",
    "rx_pkts uplink", "rx_errors uplink (%)", "ul_rssi", "ul_sinr", "phr",
    "sum_requested_prbs", "sum_granted_prbs", "dl_pmi", "dl_ri", "ul_n",
    "ul_turbo_iters", "IMSI", "RNTI", "slicing_enabled", "slice_id",
    "scheduling_policy",
]
SOURCE_TYPES = {
    "aodt_direct", "aodt_derived", "scenario_metadata",
    "deterministic_mapping", "simulated_network_state",
}
CHANNEL_TRACE_COLUMNS = [
    "slot_id", "ue_id", "bs_id", "distance_m", "path_gain_db",
    "path_loss_db", "rssi_dbm", "sinr_db", "delay_s", "path_type",
    "los_state", "valid",
]

# Project-fixed link-adaptation tables.  The CQI SINR thresholds are an
# implementation mapping, not normative thresholds defined by 3GPP.
CQI_SINR_THRESHOLDS_DB = [-6.7, -4.7, -2.3, 0.2, 2.4, 4.3, 5.9, 8.1, 10.3, 11.7, 14.1, 16.3, 18.7, 21.0, 22.7]
CQI_TO_MCS = [0, 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28]
MCS_EFFICIENCY = [
    0.2344, 0.3066, 0.3770, 0.4902, 0.6016, 0.7402, 0.8770, 1.0273,
    1.1758, 1.3262, 1.4766, 1.6953, 1.9141, 2.1602, 2.4063, 2.5703,
    2.7305, 3.0293, 3.3223, 3.6094, 3.9023, 4.2129, 4.5234, 4.8164,
    5.1152, 5.3320, 5.5547, 5.8906, 6.2266,
]


def read_csv(path: Path, expected: list[str]) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != expected:
            raise ValueError(f"{path.name}: columns {reader.fieldnames} != {expected}")
        return list(reader)


def finite(value: Any) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"non-finite numeric value: {value!r}")
    return number


def cqi_from_sinr(sinr_db: float) -> int:
    return sum(sinr_db >= threshold for threshold in CQI_SINR_THRESHOLDS_DB)


def mcs_from_cqi(cqi: int) -> int:
    return CQI_TO_MCS[max(0, min(15, int(cqi)))]


def bler_from_sinr(sinr_db: float, cqi: int) -> float:
    threshold = CQI_SINR_THRESHOLDS_DB[max(0, min(14, cqi - 1))] if cqi else CQI_SINR_THRESHOLDS_DB[0]
    return 1.0 / (1.0 + math.exp(max(-30.0, min(30.0, 1.25 * (sinr_db - threshold)))))


def dbm_to_mw(value: float) -> float:
    return 10.0 ** (value / 10.0)


def mw_to_dbm(value: float) -> float:
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("zero/non-finite channel power is not a valid formal link")
    return 10.0 * math.log10(value)


def profile_for_ue(traffic: dict[str, Any], ue_id: int, seed: int) -> dict[str, Any]:
    profiles = traffic["profiles"]
    if not profiles:
        raise ValueError("traffic_model.profiles must not be empty")
    index = ((ue_id * 2_654_435_761) ^ seed) % len(profiles)
    return profiles[index]


def arrival_packets(profile: dict[str, Any], direction: str, slot: int, ue_id: int, seed: int) -> int:
    period = int(profile["burst_period_slots"])
    phase = ((ue_id * 1_103_515_245) ^ seed) % period
    active = ((slot + phase) % period) < int(profile["burst_duration_slots"])
    return int(profile[f"{direction}_base_packets_per_slot"]) + (int(profile[f"{direction}_burst_packets_per_slot"]) if active else 0)


def allocate_prbs(items: list[dict[str, Any]], budget: int) -> None:
    """Deterministic weighted allocation with requested-PRB caps."""
    for item in items:
        item["granted_prbs"] = 0
    for _ in range(max(0, budget)):
        eligible = [item for item in items if item["granted_prbs"] < item["requested_prbs"]]
        if not eligible:
            break
        selected = max(
            eligible,
            key=lambda item: (item["scheduler_weight"] / (1 + item["granted_prbs"]), -item["ue_id"]),
        )
        selected["granted_prbs"] += 1


def provenance(seed: int) -> dict[str, Any]:
    def item(column: str, source_type: str, source_field: str, rule: str, units: str, seeded: bool = False) -> dict[str, Any]:
        return {
            "column_name": column, "source_type": source_type,
            "source_field": source_field, "formula_or_rule": rule,
            "units": units, "seed": seed if seeded else None,
        }

    rows = [
        item("traffic_type", "scenario_metadata", "ue.traffic_type + traffic_label_mapping", "scenario label mapping only; never read by queue generation", "class index"),
        item("Timestamp", "scenario_metadata", "config.meta.timestamp0_ms/slot_ms + serving_trace.slot_id", "timestamp0_ms + slot*slot_ms", "ms"),
        item("num_ues", "scenario_metadata", "meta.num_ue", "validated physical UE count", "UE"),
        item("slice_prb", "scenario_metadata", "scheduler_model.slice_prb_budget", "configured per-BS per-slice budget", "PRB"),
        item("power_multiplier", "scenario_metadata", "radio_resource_model.power_multiplier", "configured transmit-power multiplier", "ratio"),
        item("dl_mcs", "deterministic_mapping", "serving_trace.sinr_db -> dl_cqi", "CQI_TO_MCS fixed lookup", "MCS index"),
        item("dl_n_samples", "simulated_network_state", "granted DL PRB", "granted_dl_prb*subcarriers_per_prb*data_symbols_per_slot", "resource elements", True),
        item("dl_buffer [bytes]", "simulated_network_state", "DL queue previous state + arrivals", "Q[t]=min(Qmax,Q[t-1]-served[t-1]+arrivals[t])", "bytes", True),
        item("tx_brate downlink [Mbps]", "simulated_network_state", "delivered DL packets", "delivered_packets*packet_size*8/slot_duration", "Mbps", True),
        item("tx_pkts downlink", "simulated_network_state", "DL queue service", "whole packets served within PRB/MCS capacity", "packets", True),
        item("tx_errors downlink (%)", "deterministic_mapping", "serving_trace.sinr_db + dl_cqi", "bounded logistic BLER mapping", "%"),
        item("dl_cqi", "deterministic_mapping", "serving_trace.sinr_db", "project-fixed SINR threshold table; not a normative 3GPP threshold", "CQI index"),
        item("ul_mcs", "deterministic_mapping", "legacy EM proxy stored in ul_sinr", "UL CQI then CQI_TO_MCS fixed lookup", "MCS index"),
        item("ul_n_samples", "simulated_network_state", "granted UL PRB", "granted_ul_prb*subcarriers_per_prb*data_symbols_per_slot", "resource elements", True),
        item("ul_buffer [bytes]", "simulated_network_state", "UL queue previous state + arrivals", "Q[t]=min(Qmax,Q[t-1]-served[t-1]+arrivals[t])", "bytes", True),
        item("rx_brate uplink [Mbps]", "simulated_network_state", "delivered UL packets", "delivered_packets*packet_size*8/slot_duration", "Mbps", True),
        item("rx_pkts uplink", "simulated_network_state", "UL queue service", "whole packets served within PRB/MCS capacity", "packets", True),
        item("rx_errors uplink (%)", "deterministic_mapping", "legacy EM proxy stored in ul_sinr + ul_cqi", "bounded logistic BLER mapping", "%"),
        item("ul_rssi", "aodt_derived", "candidate_trace.path_gain_db", "ue_tx_power_dbm + serving path gain", "dBm"),
        item("ul_sinr", "aodt_derived", "candidate_trace.path_gain_db", "LEGACY CONTRACT FIELD: aodt_em_proxy_ul_sinr_db = serving UL power/(all-other-UE interference at serving BS + full-band noise); not native RAN SINR", "dB"),
        item("phr", "aodt_derived", "serving_trace.path_loss_db", "clip(ue_tx_power-(target_rx_power+path_loss),0,31)", "dB index"),
        item("sum_requested_prbs", "simulated_network_state", "DL+UL queue occupancy and MCS", "ceil(queued bits / effective bits per PRB)", "PRB", True),
        item("sum_granted_prbs", "simulated_network_state", "scheduler state", "per-BS/per-slice weighted allocation capped by request", "PRB", True),
        item("dl_pmi", "scenario_metadata", "radio_resource_model.tx_antenna_ports", "PMI=0 because configured AODT link is single-port", "PMI index"),
        item("dl_ri", "scenario_metadata", "radio_resource_model.tx_antenna_ports", "RI=1 because configured AODT link is single-port", "rank"),
        item("ul_n", "scenario_metadata", "radio_resource_model.numerology", "configured numerology compatibility field", "numerology index"),
        item("ul_turbo_iters", "deterministic_mapping", "UL BLER", "legacy-name decoder-effort proxy=ceil(1+11*BLER); not NR Turbo coding", "iterations"),
        item("IMSI", "scenario_metadata", "ue.imsi", "copied unchanged", "identifier"),
        item("RNTI", "scenario_metadata", "ue.rnti", "copied unchanged", "identifier"),
        item("slicing_enabled", "scenario_metadata", "scheduler_model.slicing_enabled", "configured flag", "boolean integer"),
        item("slice_id", "scenario_metadata", "ue.slice_id", "copied unchanged", "slice index"),
        item("scheduling_policy", "scenario_metadata", "ue.scheduling_policy", "copied unchanged; selects configured scheduler weight rule", "policy index"),
    ]
    assert [row["column_name"] for row in rows] == COLOSSEUM_COLUMNS
    assert {row["source_type"] for row in rows} <= SOURCE_TYPES
    return {
        "schema_version": "aodt-colosseum-provenance-v1",
        "seed": seed,
        "semantic_status": "AI_CONTRACT_COMPATIBLE_ONLY",
        "legacy_proxy_fields": {
            "ul_sinr": "aodt_em_proxy_ul_sinr_db",
        },
        "physical_distribution_comparison_allowed": False,
        "semantic_qualifications": {
            "ul_sinr": {
                "semantic_role": "aodt_em_proxy_ul_sinr_db",
                "semantic_equivalence_to_colosseum": "NOT_YET_VERIFIED_SEMANTIC_EQUIVALENT",
                "physical_distribution_comparison_allowed": False,
                "assumptions": [
                    "all other UEs cochannel-active",
                    "no PRB assignment or activity mask",
                    "full configured bandwidth used for noise",
                    "absolute CIR normalization not independently verified",
                ],
            }
        },
        "aodt_direct_inputs": [
            {"field": "candidate_trace.path_gain_db", "meaning": "06 AODT RT aggregate path power per UE/RU/slot"},
            {"field": "serving_trace", "meaning": "06 maximum-power serving link selected from valid AODT candidates"},
        ],
        "columns": rows,
    }


def trace_valid(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def normalize_06_traces(
    serving_rows: list[dict[str, str]],
    candidate_rows: list[dict[str, str]],
    scenario: dict[str, Any],
    allow_partial_smoke: bool = False,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Validate 06 channel lineage and map it to the internal adapter shape."""
    meta = scenario["meta"]
    ue_count = int(meta["num_ue"])
    bs_count = int(meta["num_bs"])
    slot_count = int(meta["time_steps"])
    expected_candidate_rows = slot_count * ue_count * bs_count
    expected_serving_rows = slot_count * ue_count
    tx_power_dbm = finite(meta["tx_power_dbm"])
    noise_dbm = -174.0 + 10.0 * math.log10(finite(meta["bandwidth_hz"])) + finite(meta["noise_figure_db"])
    noise_mw = dbm_to_mw(noise_dbm)
    timestamp0_ms = int(meta["timestamp0_ms"])
    slot_ms = int(meta["slot_ms"])
    candidates_by_pair: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    normalized_candidates: list[dict[str, str]] = []

    for row in candidate_rows:
        slot = int(row["slot_id"])
        ue_id = int(row["ue_id"])
        bs_id = int(row["bs_id"])
        valid = trace_valid(row["valid"])
        if valid:
            gain_db = finite(row["path_gain_db"])
            gain = 10.0 ** (gain_db / 10.0)
            if gain <= 0.0:
                raise ValueError(f"non-positive valid AODT gain for slot={slot}, ue={ue_id}, bs={bs_id}")
            if not math.isclose(finite(row["path_loss_db"]), -gain_db, abs_tol=1e-4):
                raise ValueError("candidate path loss is inconsistent with 06 path gain")
            expected_rssi = tx_power_dbm + gain_db
            if not math.isclose(finite(row["rssi_dbm"]), expected_rssi, abs_tol=1e-4):
                raise ValueError("candidate RSSI is inconsistent with 06 path gain")
            finite(row["sinr_db"])
            finite(row["delay_s"])
        else:
            if any(str(row[name]).strip() for name in ("path_gain_db", "path_loss_db", "rssi_dbm", "sinr_db", "delay_s")):
                raise ValueError("invalid candidate must not contain fabricated channel values")
            gain = 0.0
            expected_rssi = 0.0
        internal = {
            "slot": str(slot), "ue_id": str(ue_id), "bs_id": str(bs_id),
            "path_gain_linear": repr(gain), "rx_power_dbm": repr(expected_rssi),
            "valid": "1" if valid else "0",
        }
        normalized_candidates.append(internal)
        candidates_by_pair[(slot, ue_id)].append({**row, "_gain": gain, "_valid": valid})

    if len(candidate_rows) != expected_candidate_rows:
        raise ValueError(
            f"06 candidate trace must contain {expected_candidate_rows} rows, "
            f"got {len(candidate_rows)}"
        )
    for slot in range(slot_count):
        for ue_id in range(ue_count):
            group = candidates_by_pair.get((slot, ue_id), [])
            if len(group) != bs_count or sorted(int(row["bs_id"]) for row in group) != list(range(bs_count)):
                raise ValueError(f"06 candidate coverage is incomplete for slot={slot}, ue={ue_id}")
            total_rx_mw = dbm_to_mw(tx_power_dbm) * sum(float(row["_gain"]) for row in group)
            for row in group:
                if row["_valid"]:
                    signal_mw = dbm_to_mw(tx_power_dbm) * float(row["_gain"])
                    expected_sinr = 10.0 * math.log10(signal_mw / (total_rx_mw - signal_mw + noise_mw))
                    if not math.isclose(finite(row["sinr_db"]), expected_sinr, abs_tol=1e-3):
                        raise ValueError(f"candidate SINR is inconsistent with 06 candidate powers for slot={slot}, ue={ue_id}")

    normalized_serving: list[dict[str, str]] = []
    if len(serving_rows) != expected_serving_rows and not allow_partial_smoke:
        raise ValueError(
            f"06 serving trace must contain {expected_serving_rows} rows, "
            f"got {len(serving_rows)}"
        )
    if allow_partial_smoke and not serving_rows:
        raise ValueError("partial smoke input has no valid serving rows")
    seen_serving: set[tuple[int, int]] = set()
    for row in serving_rows:
        slot = int(row["slot_id"])
        ue_id = int(row["ue_id"])
        bs_id = int(row["bs_id"])
        key = (slot, ue_id)
        if key in seen_serving:
            raise ValueError(f"duplicate 06 serving row for slot={slot}, ue={ue_id}")
        seen_serving.add(key)
        if not trace_valid(row["valid"]):
            raise ValueError(f"formal serving row is invalid for slot={slot}, ue={ue_id}")
        group = candidates_by_pair[(slot, ue_id)]
        valid_group = [candidate for candidate in group if candidate["_valid"]]
        if not valid_group:
            raise ValueError(f"no valid 06 candidates for slot={slot}, ue={ue_id}")
        strongest = max(valid_group, key=lambda candidate: float(candidate["_gain"]))
        if int(strongest["bs_id"]) != bs_id:
            raise ValueError(f"serving BS is not the strongest valid 06 candidate for slot={slot}, ue={ue_id}")
        for field in ("path_gain_db", "path_loss_db", "rssi_dbm", "sinr_db", "distance_m", "delay_s"):
            if not math.isclose(finite(row[field]), finite(strongest[field]), abs_tol=1e-4):
                raise ValueError(f"serving field {field} differs from its 06 candidate row")
        normalized_serving.append({
            "slot": str(slot), "timestamp_ms": str(timestamp0_ms + slot * slot_ms),
            "ue_id": str(ue_id), "bs_id": str(bs_id),
            "sinr_db": row["sinr_db"], "rssi_dbm": row["rssi_dbm"],
            "distance_m": row["distance_m"], "path_loss_db": row["path_loss_db"],
        })
    if allow_partial_smoke and {int(row["ue_id"]) for row in serving_rows} != set(range(ue_count)):
        raise ValueError("partial smoke input must retain at least one valid serving row for every UE")
    return normalized_serving, normalized_candidates


def build_metrics(
    serving_rows: list[dict[str, str]],
    candidate_rows: list[dict[str, str]],
    scenario: dict[str, Any],
    seed: int,
    allow_partial_smoke: bool = False,
) -> tuple[dict[int, list[dict[str, Any]]], dict[str, Any]]:
    meta = scenario["meta"]
    definition = scenario["experiment_definition"]
    traffic = definition["traffic_model"]
    scheduler = definition["scheduler_model"]
    radio = definition["radio_resource_model"]
    ues = {int(row["id"]): row for row in scenario["ue"]}
    ue_count = int(meta["num_ue"])
    bs_count = int(meta["num_bs"])
    slot_count = int(meta["time_steps"])
    expected_candidate_rows = slot_count * ue_count * bs_count
    expected_serving_rows = slot_count * ue_count
    if sorted(ues) != list(range(ue_count)):
        raise ValueError(f"scenario must define exactly UE IDs 0..{ue_count - 1}")
    if traffic.get("profile_assignment_rule") != "seeded_ue_hash_independent_of_traffic_type":
        raise ValueError("traffic profile assignment must be independent of traffic_type")

    serving: dict[tuple[int, int], dict[str, Any]] = {}
    for row in serving_rows:
        key = (int(row["slot"]), int(row["ue_id"]))
        if key in serving:
            raise ValueError(f"duplicate serving row {key}")
        serving[key] = {**row, **{name: finite(row[name]) for name in ("sinr_db", "rssi_dbm", "distance_m", "path_loss_db")}}
    candidate: dict[tuple[int, int, int], float] = {}
    for row in candidate_rows:
        key = (int(row["slot"]), int(row["ue_id"]), int(row["bs_id"]))
        gain = finite(row["path_gain_linear"])
        if gain < 0.0 or key in candidate:
            raise ValueError(f"invalid/duplicate candidate row {key}")
        if gain > 0.0:
            expected_rx = finite(meta["tx_power_dbm"]) + 10.0 * math.log10(gain)
            if not math.isclose(finite(row["rx_power_dbm"]), expected_rx, abs_tol=1e-4):
                raise ValueError(f"candidate RX power is inconsistent with AODT path gain for {key}")
        candidate[key] = gain
    if len(candidate) != expected_candidate_rows:
        raise ValueError(
            f"candidate row count must be {expected_candidate_rows}, got {len(candidate)}"
        )
    if len(serving) != expected_serving_rows and not allow_partial_smoke:
        raise ValueError(
            f"formal row counts must be {expected_serving_rows}/"
            f"{expected_candidate_rows}, got {len(serving)}/{len(candidate)}"
        )

    slots = sorted({key[0] for key in candidate})
    if slots != list(range(slot_count)):
        raise ValueError(f"candidate slots must be exactly 0..{slot_count - 1}")
    if not allow_partial_smoke and any((slot, ue_id) not in serving for slot in slots for ue_id in ues):
        raise ValueError("serving trace does not cover every UE in every slot")
    if any(
        sum((slot, ue_id, bs_id) in candidate for bs_id in range(bs_count)) != bs_count
        for slot in slots
        for ue_id in ues
    ):
        raise ValueError(
            f"candidate trace must contain exactly {bs_count} RU links per UE/slot"
        )

    slot_seconds = finite(meta["slot_ms"]) / 1000.0
    bandwidth = finite(meta["bandwidth_hz"])
    noise_dbm = -174.0 + 10.0 * math.log10(bandwidth) + finite(meta["noise_figure_db"])
    noise_mw = dbm_to_mw(noise_dbm)
    ue_tx_dbm = finite(radio["ue_tx_power_dbm"])
    ue_tx_mw = dbm_to_mw(ue_tx_dbm)
    subcarriers = int(radio["subcarriers_per_prb"])
    data_symbols = int(radio["data_symbols_per_slot"])
    layers = int(radio["layers"])
    label_mapping = {str(key): int(value) for key, value in definition["traffic_label_mapping"].items()}
    slice_budgets = {int(key): int(value) for key, value in scheduler["slice_prb_budget"].items()}
    dl_fraction = finite(scheduler["dl_prb_fraction"])
    max_buffer = int(traffic["max_buffer_bytes"])

    state = {
        ue_id: {"dl_packets": 0, "ul_packets": 0, "avg_mbps": 0.01, "hol_slots": 0}
        for ue_id in ues
    }
    outputs: dict[int, list[dict[str, Any]]] = {ue_id: [] for ue_id in ues}
    dropped_packets = 0
    total_arrivals = 0
    total_delivered = 0

    for slot in slots:
        items: list[dict[str, Any]] = []
        for ue_id, ue in sorted(ues.items()):
            if (slot, ue_id) not in serving:
                continue
            link = serving[(slot, ue_id)]
            bs_id = int(link["bs_id"])
            gain = candidate[(slot, ue_id, bs_id)]
            if gain <= 0.0 or float(link["path_loss_db"]) >= 300.0:
                raise ValueError(f"formal serving path is invalid for slot={slot}, ue={ue_id}")
            if not math.isclose(float(link["path_loss_db"]), -10.0 * math.log10(gain), abs_tol=1e-4):
                raise ValueError(f"serving path loss is inconsistent with AODT path gain for slot={slot}, ue={ue_id}")
            if not math.isclose(float(link["rssi_dbm"]), finite(meta["tx_power_dbm"]) + 10.0 * math.log10(gain), abs_tol=1e-4):
                raise ValueError(f"serving RSSI is inconsistent with AODT path gain for slot={slot}, ue={ue_id}")
            strongest_bs = max(
                range(bs_count), key=lambda value: candidate[(slot, ue_id, value)]
            )
            if strongest_bs != bs_id:
                raise ValueError(f"serving BS does not match strongest AODT candidate for slot={slot}, ue={ue_id}")

            dl_sinr = float(link["sinr_db"])
            dl_signal_mw = dbm_to_mw(finite(meta["tx_power_dbm"]) + 10.0 * math.log10(gain))
            dl_interference_mw = dbm_to_mw(finite(meta["tx_power_dbm"])) * sum(
                candidate[(slot, ue_id, other_bs)]
                for other_bs in range(bs_count)
                if other_bs != bs_id
            )
            expected_dl_sinr = 10.0 * math.log10(dl_signal_mw / (dl_interference_mw + noise_mw))
            if not math.isclose(dl_sinr, expected_dl_sinr, abs_tol=1e-3):
                raise ValueError(f"serving SINR is inconsistent with AODT candidates for slot={slot}, ue={ue_id}")
            dl_cqi = cqi_from_sinr(dl_sinr)
            dl_mcs = mcs_from_cqi(dl_cqi)
            dl_bler = bler_from_sinr(dl_sinr, dl_cqi)

            ul_signal = ue_tx_mw * gain
            ul_interference = sum(
                ue_tx_mw * candidate[(slot, other_id, bs_id)]
                for other_id in ues if other_id != ue_id
            )
            ul_rssi = mw_to_dbm(ul_signal)
            ul_sinr = 10.0 * math.log10(ul_signal / (ul_interference + noise_mw))
            ul_cqi = cqi_from_sinr(ul_sinr)
            ul_mcs = mcs_from_cqi(ul_cqi)
            ul_bler = bler_from_sinr(ul_sinr, ul_cqi)

            profile = profile_for_ue(traffic, ue_id, seed)
            packet_bytes = int(profile["packet_size_bytes"])
            current = state[ue_id]
            dl_arrivals = arrival_packets(profile, "dl", slot, ue_id, seed)
            ul_arrivals = arrival_packets(profile, "ul", slot, ue_id, seed)
            total_arrivals += dl_arrivals + ul_arrivals
            max_packets = max_buffer // packet_bytes
            new_dl = min(max_packets, current["dl_packets"] + dl_arrivals)
            new_ul = min(max_packets, current["ul_packets"] + ul_arrivals)
            dropped_packets += current["dl_packets"] + dl_arrivals - new_dl
            dropped_packets += current["ul_packets"] + ul_arrivals - new_ul
            current["dl_packets"], current["ul_packets"] = new_dl, new_ul
            current["hol_slots"] = current["hol_slots"] + 1 if new_dl + new_ul else 0

            dl_bits_per_prb = subcarriers * data_symbols * layers * MCS_EFFICIENCY[dl_mcs] * (1.0 - dl_bler)
            ul_bits_per_prb = subcarriers * data_symbols * layers * MCS_EFFICIENCY[ul_mcs] * (1.0 - ul_bler)
            queued_dl_bits = new_dl * packet_bytes * 8
            queued_ul_bits = new_ul * packet_bytes * 8
            requested = math.ceil(queued_dl_bits / max(dl_bits_per_prb, 1.0)) + math.ceil(queued_ul_bits / max(ul_bits_per_prb, 1.0))
            policy = int(ue["scheduling_policy"])
            if policy == 0:
                weight = (1.0 + queued_dl_bits + queued_ul_bits) * max(MCS_EFFICIENCY[dl_mcs], 0.1) / (1.0 + current["avg_mbps"])
            elif policy == 1:
                weight = 1.0 if requested else 0.0
            elif policy == 2:
                weight = (1.0 + current["hol_slots"]) ** 2 * max(MCS_EFFICIENCY[dl_mcs], 0.1)
            else:
                raise ValueError(f"unsupported scheduling_policy={policy}")
            items.append({
                "ue_id": ue_id, "ue": ue, "link": link, "bs_id": bs_id,
                "profile": profile, "packet_bytes": packet_bytes,
                "dl_cqi": dl_cqi, "dl_mcs": dl_mcs, "dl_bler": dl_bler,
                "ul_mcs": ul_mcs, "ul_bler": ul_bler, "ul_rssi": ul_rssi, "ul_sinr": ul_sinr,
                "dl_bits_per_prb": dl_bits_per_prb, "ul_bits_per_prb": ul_bits_per_prb,
                "requested_prbs": requested, "scheduler_weight": weight,
                "slice_id": int(ue["slice_id"]),
            })

        groups: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
        for item in items:
            groups[(item["bs_id"], item["slice_id"])].append(item)
        for (_bs_id, slice_id), group in groups.items():
            if slice_id not in slice_budgets:
                raise ValueError(f"missing PRB budget for slice {slice_id}")
            allocate_prbs(group, slice_budgets[slice_id])

        for item in items:
            ue_id = item["ue_id"]
            ue = item["ue"]
            current = state[ue_id]
            packet_bytes = item["packet_bytes"]
            granted = int(item["granted_prbs"])
            dl_granted = int(round(granted * dl_fraction))
            ul_granted = granted - dl_granted
            dl_capacity_packets = int(item["dl_bits_per_prb"] * dl_granted // (packet_bytes * 8))
            ul_capacity_packets = int(item["ul_bits_per_prb"] * ul_granted // (packet_bytes * 8))
            delivered_dl = min(current["dl_packets"], dl_capacity_packets)
            delivered_ul = min(current["ul_packets"], ul_capacity_packets)
            dl_buffer_bytes = current["dl_packets"] * packet_bytes
            ul_buffer_bytes = current["ul_packets"] * packet_bytes
            current["dl_packets"] -= delivered_dl
            current["ul_packets"] -= delivered_ul
            total_delivered += delivered_dl + delivered_ul
            dl_mbps = delivered_dl * packet_bytes * 8 / slot_seconds / 1e6
            ul_mbps = delivered_ul * packet_bytes * 8 / slot_seconds / 1e6
            current["avg_mbps"] = 0.9 * current["avg_mbps"] + 0.1 * (dl_mbps + ul_mbps)
            if current["dl_packets"] + current["ul_packets"] == 0:
                current["hol_slots"] = 0
            traffic_name = str(ue["traffic_type"])
            if traffic_name not in label_mapping:
                raise ValueError(f"missing traffic label mapping for {traffic_name}")
            link = item["link"]
            phr = int(max(0, min(31, round(ue_tx_dbm - (finite(radio["target_gnb_rx_power_dbm"]) + float(link["path_loss_db"]))))))
            row = {
                "traffic_type": label_mapping[traffic_name],
                "Timestamp": int(link["timestamp_ms"]),
                "num_ues": len(ues),
                "slice_prb": slice_budgets[item["slice_id"]],
                "power_multiplier": finite(radio["power_multiplier"]),
                "dl_mcs": item["dl_mcs"],
                "dl_n_samples": dl_granted * subcarriers * data_symbols,
                "dl_buffer [bytes]": dl_buffer_bytes,
                "tx_brate downlink [Mbps]": round(dl_mbps, 6),
                "tx_pkts downlink": delivered_dl,
                "tx_errors downlink (%)": round(100.0 * item["dl_bler"], 6),
                "dl_cqi": item["dl_cqi"],
                "ul_mcs": item["ul_mcs"],
                "ul_n_samples": ul_granted * subcarriers * data_symbols,
                "ul_buffer [bytes]": ul_buffer_bytes,
                "rx_brate uplink [Mbps]": round(ul_mbps, 6),
                "rx_pkts uplink": delivered_ul,
                "rx_errors uplink (%)": round(100.0 * item["ul_bler"], 6),
                "ul_rssi": round(item["ul_rssi"], 6),
                "ul_sinr": round(item["ul_sinr"], 6),
                "phr": phr,
                "sum_requested_prbs": item["requested_prbs"],
                "sum_granted_prbs": granted,
                "dl_pmi": 0,
                "dl_ri": 1,
                "ul_n": int(radio["numerology"]),
                "ul_turbo_iters": int(math.ceil(1.0 + 11.0 * item["ul_bler"])),
                "IMSI": str(ue["imsi"]),
                "RNTI": int(ue["rnti"]),
                "slicing_enabled": int(bool(scheduler["slicing_enabled"])),
                "slice_id": item["slice_id"],
                "scheduling_policy": int(ue["scheduling_policy"]),
            }
            if list(row) != COLOSSEUM_COLUMNS:
                raise AssertionError("internal AI-contract field order changed")
            for name, value in row.items():
                if name not in {"IMSI"} and isinstance(value, (int, float)) and not math.isfinite(float(value)):
                    raise ValueError(f"non-finite output {name}")
            outputs[ue_id].append(row)

    if not allow_partial_smoke and any(
        len(rows) != slot_count for rows in outputs.values()
    ):
        raise ValueError(f"every UE must have exactly {slot_count} metrics rows")
    if allow_partial_smoke and any(not rows for rows in outputs.values()):
        raise ValueError(
            f"partial smoke output must retain all {ue_count} UE files with at least one row"
        )
    report = {
        "traffic_model": traffic,
        "scheduler_model": scheduler,
        "radio_resource_model": radio,
        "seed": seed,
        "queue_conservation": {
            "arrival_packets": total_arrivals,
            "delivered_packets": total_delivered,
            "dropped_packets": dropped_packets,
            "remaining_packets": sum(value["dl_packets"] + value["ul_packets"] for value in state.values()),
        },
        "noise_power_dbm": noise_dbm,
    }
    conservation = report["queue_conservation"]
    if conservation["arrival_packets"] != conservation["delivered_packets"] + conservation["dropped_packets"] + conservation["remaining_packets"]:
        raise AssertionError("packet conservation failed")
    return outputs, report


def write_metrics(output_dir: Path, outputs: dict[int, list[dict[str, Any]]], scenario: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=False)
    ues = {int(row["id"]): row for row in scenario["ue"]}
    for ue_id, rows in sorted(outputs.items()):
        ue = ues[ue_id]
        serving_bs = int(ue.get("colosseum_source_bs_id", 0))  # filename compatibility metadata only
        filename = f"sched{int(ue['scheduling_policy'])}_tr0_exp0_bs{serving_bs}_slices_bs{serving_bs}_{ue['traffic_type']}_{ue['imsi']}_metrics.csv"
        with (output_dir / filename).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=COLOSSEUM_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
