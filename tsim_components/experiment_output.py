"""Versioned, machine-readable output for architecture experiments."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from tsim_components.noc import NOC_DYNAMIC_ENERGY_PJ_PER_BYTE_HOP, Topo


EXPERIMENT_SCHEMA_VERSION = 3


def _json_value(value: Any) -> Any:
    """Convert numpy/enums/tuples and other scalar wrappers to JSON values."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_value(item) for item in value]
    if hasattr(value, "item"):
        return _json_value(value.item())
    if hasattr(value, "value"):
        return _json_value(value.value)
    return str(value)


def _operator_record(log: object, tiling: Mapping[str, Any] | None) -> dict:
    meta = getattr(log, "spatial_meta", {}) or {}
    accesses = meta.get("dram_access_records", []) or []
    banks = sorted({
        int(bank)
        for access in accesses
        for bank in (access.get("bank_ids", []) or [])
    })
    channels = sorted({
        int(channel)
        for access in accesses
        for channel in (access.get("channel_ids", []) or [])
    })
    tensor_accesses = [
        {
            "tensor_id": _json_value(access.get("tensor_id")),
            "stage": str(access.get("stage", "")),
            "address": (
                int(access["address"])
                if access.get("address") is not None else None
            ),
            "allocation_bytes": int(access.get("allocation_bytes", 0) or 0),
            "transfer_bytes": int(access.get("total_bytes", 0) or 0),
            "bank_ids": [
                int(bank) for bank in (access.get("bank_ids", []) or [])
            ],
            "channel_ids": [
                int(channel)
                for channel in (access.get("channel_ids", []) or [])
            ],
            "requester_core_weights": [
                [int(core), float(weight)]
                for core, weight in (
                    access.get("requester_core_weights", []) or []
                )
            ],
            "channel_noc_nodes": [
                int(node)
                for node in (access.get("channel_noc_nodes", []) or [])
            ],
            "dram_noc_channel_ids": [
                int(channel)
                for channel in (
                    access.get("dram_noc_channel_ids", []) or []
                )
            ],
            "dram_noc_byte_hops": float(
                access.get("dram_noc_byte_hops", 0.0) or 0.0
            ),
            "dram_noc_cycles": int(
                access.get("dram_noc_cycles", 0) or 0
            ),
            "dram_noc_max_hops": int(
                access.get("dram_noc_max_hops", 0) or 0
            ),
            "dram_noc_max_link_bytes": float(
                access.get("dram_noc_max_link_bytes", 0.0) or 0.0
            ),
        }
        for access in accesses
    ]
    return {
        "operator_index": int(getattr(log, "op_id", 0)),
        "tiling": _json_value(tiling or {}),
        "timing_cycles": {
            "start": int(getattr(log, "t_dram_ld_start", 0)),
            "finish": int(getattr(log, "t_finish", 0)),
            "dram_read": int(getattr(log, "dram_ld_dur", 0)),
            "dram_write": int(getattr(log, "dram_st_dur", 0)),
            "noc_broadcast": int(getattr(log, "bcast_dur", 0)),
            "noc_shift": int(getattr(log, "shift_dur", 0)),
            "noc_reduce": int(getattr(log, "reduce_dur", 0)),
            "compute": int(getattr(log, "comp_dur", 0)),
        },
        "dram_bytes": {
            "read": int(getattr(log, "dram_r_bytes", 0)),
            "write": int(getattr(log, "dram_w_bytes", 0)),
        },
        "row_buffer": {
            "read": {
                "hits": int(getattr(log, "dram_r_row_hits", 0)),
                "misses": int(getattr(log, "dram_r_row_misses", 0)),
                "conflicts": int(getattr(log, "dram_r_row_conflicts", 0)),
            },
            "write": {
                "hits": int(getattr(log, "dram_w_row_hits", 0)),
                "misses": int(getattr(log, "dram_w_row_misses", 0)),
                "conflicts": int(getattr(log, "dram_w_row_conflicts", 0)),
            },
        },
        "dram_dynamic_energy_pj": {
            "base_transfer": float(getattr(log, "energy_dram_base", 0.0)),
            "row_conflict_act_pre": float(
                getattr(log, "energy_dram_row_conflict", 0.0)
            ),
            "total": float(getattr(log, "energy_dram", 0.0)),
        },
        "noc_byte_hops": {
            "broadcast": float(getattr(log, "noc_bcast_byte_hops", 0.0)),
            "shift": float(getattr(log, "noc_shift_byte_hops", 0.0)),
            "reduce": float(getattr(log, "noc_reduce_byte_hops", 0.0)),
            "operator_total": float(
                getattr(log, "operator_noc_byte_hops", 0.0)
            ),
            "dram_read": float(
                getattr(log, "dram_noc_read_byte_hops", 0.0)
            ),
            "dram_write": float(
                getattr(log, "dram_noc_write_byte_hops", 0.0)
            ),
            "dram_total": float(
                getattr(log, "dram_noc_byte_hops", 0.0)
            ),
            "total": float(getattr(log, "noc_byte_hops", 0.0)),
        },
        "noc_dynamic_energy_pj": {
            "energy_per_byte_hop": float(getattr(
                log,
                "noc_energy_pj_per_byte_hop",
                NOC_DYNAMIC_ENERGY_PJ_PER_BYTE_HOP,
            )),
            "control": float(
                getattr(log, "energy_noc_control", 0.0)
            ),
            "operator_transport": float(
                getattr(log, "energy_noc_operator", 0.0)
            ),
            "dram_transport": float(
                getattr(log, "energy_noc_dram_transport", 0.0)
            ),
            "transport_total": float(
                getattr(log, "energy_noc_operator", 0.0)
                + getattr(log, "energy_noc_dram_transport", 0.0)
            ),
            "total": float(getattr(log, "energy_noc", 0.0)),
            "average_power_w": float(
                getattr(log, "noc_dynamic_power_W", 0.0)
            ),
        },
        "placement": {
            "policy": str(getattr(log, "dram_placement_policy", "unknown")),
            "bank_ids": banks,
            "channel_ids": channels,
            "access_record_count": len(accesses),
            "tensor_accesses": tensor_accesses,
        },
    }


def build_experiment_record(
    *,
    workload_name: str,
    hardware: object,
    stats: Mapping[str, Any],
    operator_logs: Sequence[object],
    partitions: Sequence[tuple],
    operator_names: Sequence[str],
    placement_policy: str,
    total_layers: int,
    simulated_layers: int,
    tiling_cache: Mapping[str, Any] | None = None,
    motif_start_layer: int = 0,
    motif_layers: int = 0,
    source_total_layers: int = 0,
) -> dict:
    """Build one complete L1-ready result record for a design point."""
    dram = hardware.mem
    comp = hardware.comp
    noc = hardware.noc
    topology = getattr(getattr(noc, "topology", None), "value", None)
    if topology is None:
        topology = str(getattr(noc, "topology", "unknown"))

    tilings = []
    for index, ((temporal, spatial), name) in enumerate(
        zip(partitions, operator_names)
    ):
        tilings.append({
            "operator_index": int(index),
            "operator_name": str(name),
            "spatial": _json_value(spatial),
            "temporal": _json_value(temporal),
        })

    frequency_mhz = int(hardware.npu_freq_mHz)
    exec_cycles = int(stats.get("exec_time", -1))
    exec_ms = (
        exec_cycles / (frequency_mhz * 1_000.0)
        if exec_cycles >= 0 and frequency_mhz > 0 else None
    )
    scale = (
        float(total_layers) / float(simulated_layers)
        if simulated_layers else 1.0
    )
    geometry = dram.geometry.as_dict()
    row_read = {
        "hits": int(stats.get("dram_r_row_hits", 0)),
        "misses": int(stats.get("dram_r_row_misses", 0)),
        "conflicts": int(stats.get("dram_r_row_conflicts", 0)),
    }
    row_write = {
        "hits": int(stats.get("dram_w_row_hits", 0)),
        "misses": int(stats.get("dram_w_row_misses", 0)),
        "conflicts": int(stats.get("dram_w_row_conflicts", 0)),
    }
    workload = {
        "name": str(workload_name),
        "total_layers": int(total_layers),
        "simulated_layers": int(simulated_layers or total_layers),
        "extrapolation_factor": scale,
    }
    if motif_layers:
        workload["motif"] = {
            "start_layer": int(motif_start_layer),
            "layers": int(motif_layers),
            "source_total_layers": int(source_total_layers or total_layers),
        }

    record = {
        "schema_version": EXPERIMENT_SCHEMA_VERSION,
        "workload": workload,
        "architecture": {
            "num_cores": int(hardware.num_cores),
            "core_group_size": int(hardware.core_grp_size),
            "core_sram_bytes": int(hardware.sram_size),
            "execution_sram_bytes": int(hardware.exe_sram_size),
            "frequency_mhz": frequency_mhz,
            "compute": {
                "systolic_array_shape": _json_value(
                    list(getattr(comp, "mm_pad_shape", []))[-2:]
                ),
                "vector_width": int(getattr(comp, "ew_pad_len", 0)),
            },
            "noc": {
                "topology": topology,
                "link_bandwidth_bytes_per_cycle": float(
                    getattr(noc, "bandwidth_bytepc", 0.0)
                ),
                "routing": (
                    "dimension_ordered_xy"
                    if getattr(noc, "topology", None) == Topo.MESH
                    else "shortest_path"
                ),
                "dynamic_energy_pj_per_byte_hop": (
                    float(getattr(
                        noc,
                        "energy_pj_per_byte_hop",
                        NOC_DYNAMIC_ENERGY_PJ_PER_BYTE_HOP,
                    ))
                ),
                "router_pipeline_cycles_per_hop": int(getattr(
                    noc, "router_pipeline_cycles_per_hop", 1
                )),
                "dram_noc_startup_cycles": getattr(
                    noc, "dram_noc_startup_cycles", None
                ),
            },
            "dram": {
                **geometry,
                "capacity_bytes": getattr(dram, "capacity_bytes", None),
                "timing_cycles": {
                    "tCL": int(dram.CL),
                    "tRCD": int(dram.tRCD),
                    "tRP": int(dram.tRP),
                    "tRAS_recorded_not_modeled": int(
                        getattr(dram, "tRAS_recorded", 0)
                    ),
                },
                "aggregate_bandwidth_gb_per_s": float(hardware.dram_bw_GBps),
                "channel_bandwidth_gb_per_s": float(
                    hardware.dram_bw_GBps / max(1, dram.num_channels)
                ),
                "bandwidth_unit": "GB/s_decimal",
                "tsv_buses_per_channel": int(
                    getattr(dram, "tsv_buses_per_channel", 1)
                ),
                "tsv_bus_bandwidth_gb_per_s": float(
                    getattr(
                        dram,
                        "tsv_bus_bandwidth_GBps",
                        hardware.dram_bw_GBps / max(1, dram.num_channels),
                    )
                ),
                "aggregate_bytes_per_cycle_per_core": float(
                    dram.total_bytes_per_cycle
                ),
                "channel_bytes_per_cycle_per_core": float(
                    dram.channel_bytes_per_cycle
                ),
                "modeled_aggregate_bytes_per_cycle": float(
                    dram.total_bytes_per_cycle * hardware.num_cores
                ),
                "modeled_channel_bytes_per_cycle": float(
                    dram.channel_bytes_per_cycle * hardware.num_cores
                ),
                "scheduler": {
                    "name": "bounded_fr_fcfs",
                    "request_window": int(getattr(dram, "frfcfs_window", 32)),
                    "models_tRAS": False,
                },
                "placement_policy": str(placement_policy),
            },
        },
        "tiling": tilings,
        "metrics": {
            "valid": exec_cycles >= 0,
            "end_to_end_cycles": exec_cycles,
            "end_to_end_time_ms": exec_ms,
            "dram_bytes": {
                "read": int(stats.get("dram_r_bytes", 0)),
                "write": int(stats.get("dram_w_bytes", 0)),
            },
            "row_buffer": {
                "read": row_read,
                "write": row_write,
                "total": {
                    key: row_read[key] + row_write[key]
                    for key in ("hits", "misses", "conflicts")
                },
            },
            "dram_dynamic_energy_pj": {
                "scope": "dynamic_no_refresh_no_static",
                "act_pre_energy_per_conflict_pj": 7270.0,
                "base_transfer": float(stats.get("dram_base_energy", 0.0)),
                "row_conflict_act_pre": float(
                    stats.get("dram_row_conflict_energy", 0.0)
                ),
                "total": float(stats.get("dram_energy", 0.0)),
            },
            "noc_byte_hops": {
                "broadcast": float(stats.get("noc_bcast_byte_hops", 0.0)),
                "shift": float(stats.get("noc_shift_byte_hops", 0.0)),
                "reduce": float(stats.get("noc_reduce_byte_hops", 0.0)),
                "operator_total": float(
                    stats.get("operator_noc_byte_hops", 0.0)
                ),
                "dram_read": float(
                    stats.get("dram_noc_read_byte_hops", 0.0)
                ),
                "dram_write": float(
                    stats.get("dram_noc_write_byte_hops", 0.0)
                ),
                "dram_total": float(stats.get("dram_noc_byte_hops", 0.0)),
                "total": float(stats.get("noc_byte_hops", 0.0)),
            },
            "noc_dynamic_energy_pj": {
                "energy_per_byte_hop": float(getattr(
                    noc,
                    "energy_pj_per_byte_hop",
                    NOC_DYNAMIC_ENERGY_PJ_PER_BYTE_HOP,
                )),
                "control": float(
                    stats.get("noc_control_energy", 0.0)
                ),
                "operator_transport": float(
                    stats.get("noc_operator_energy", 0.0)
                ),
                "dram_transport": float(
                    stats.get("noc_dram_transport_energy", 0.0)
                ),
                "transport_total": float(
                    stats.get("noc_operator_energy", 0.0)
                    + stats.get("noc_dram_transport_energy", 0.0)
                ),
                "total": float(stats.get("noc_energy", 0.0)),
                "average_power_w": float(
                    stats.get("noc_dynamic_power_w", 0.0)
                ),
            },
            "dram_noc_cycles": {
                "read": int(stats.get("dram_noc_read_cycles", 0)),
                "write": int(stats.get("dram_noc_write_cycles", 0)),
            },
            "noc_max_link_bytes": float(
                stats.get("noc_max_link_bytes", 0.0)
            ),
            "noc_max_route_hops": int(
                stats.get("noc_max_route_hops", 0)
            ),
            "total_dynamic_energy_pj": float(stats.get("exec_energy", 0.0)),
        },
        "cache": {"tiling": _json_value(tiling_cache or {})},
        "operators": [
            _operator_record(log, tilings[index] if index < len(tilings) else None)
            for index, log in enumerate(operator_logs)
        ],
    }
    identity_payload = {
        "workload": record["workload"],
        "architecture": record["architecture"],
        "tiling": record["tiling"],
    }
    record["configuration_id"] = hashlib.blake2b(
        json.dumps(identity_payload, sort_keys=True, separators=(",", ":")).encode(),
        digest_size=12,
    ).hexdigest()
    return record


def write_experiment_records_jsonl(
    path: str | os.PathLike[str],
    records: Iterable[Mapping[str, Any]],
) -> Path:
    """Atomically write one JSON object per design point."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f"{destination.name}.tmp-{os.getpid()}"
    )
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(
                _json_value(record), sort_keys=True, separators=(",", ":")
            ))
            handle.write("\n")
    os.replace(temporary, destination)
    return destination


def write_experiment_record_files(
    directory: str | os.PathLike[str],
    records: Iterable[Mapping[str, Any]],
) -> list[Path]:
    """Write collision-free per-configuration JSON files for sweep reuse."""
    output_dir = Path(directory)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for record in records:
        configuration_id = str(record.get("configuration_id", ""))
        if not configuration_id:
            raise ValueError("experiment record is missing configuration_id")
        destination = output_dir / f"{configuration_id}.json"
        temporary = destination.with_name(
            f"{destination.name}.tmp-{os.getpid()}"
        )
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(_json_value(record), handle, sort_keys=True, indent=2)
            handle.write("\n")
        os.replace(temporary, destination)
        paths.append(destination)
    return paths
