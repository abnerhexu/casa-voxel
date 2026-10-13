"""Replay externally selected block placement in Voxel's timing model."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from typing import Iterable, Mapping, Sequence

from .dram_placement import build_placement_plan, record_signature
from .system_model import Hardware, execute, route


REQUIRED_FIELDS = frozenset({
    "demand_id", "tensor_id", "kind", "replica_id", "portal",
    "portal_node", "bank", "num_bytes", "allocation_bytes", "live_bytes",
    "live_start", "live_end", "consumers", "assigned_consumers", "producers",
})


def _canonical_hash(value) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.blake2b(payload, digest_size=12).hexdigest()


def _physical_name(record: Mapping) -> str:
    return (
        f"{record['tensor_id']}#block={record['demand_id']}"
        f"#replica={int(record['replica_id'])}"
    )


def build_exact_replay_graph(
    placement_records: Iterable[Mapping], *, cores: int,
    compute_cores: Sequence[int], compute_cycles: int,
    scratch_bytes_per_core: int, source_policy: str,
    total_banks: int, num_channels: int,
) -> dict:
    """Lower one CASA operator placement into a bounded aggregate graph."""
    records = [dict(record) for record in placement_records]
    if not records:
        raise ValueError("exact replay needs at least one placement record")
    if cores <= 0 or compute_cycles < 0 or scratch_bytes_per_core < 0:
        raise ValueError("invalid exact replay dimensions or costs")
    compute_cores = tuple(sorted(set(map(int, compute_cores))))
    if not compute_cores or any(core < 0 or core >= cores for core in compute_cores):
        raise ValueError("exact replay compute cores are outside the machine")

    tensors, mapping, placement_inputs = {}, {}, []
    accesses, broadcasts, reductions = [], [], []
    extra_sram = Counter()
    vertical = Counter()
    seen_physical = set()
    reduced_demands = set()
    participants = set(compute_cores)
    portal_nodes = {}

    for record in sorted(
        records, key=lambda row: (str(row.get("demand_id")), int(row.get("replica_id", -1)))
    ):
        missing = REQUIRED_FIELDS - set(record)
        if missing:
            raise ValueError(f"exact placement record is missing {sorted(missing)}")
        kind = str(record["kind"])
        if kind not in {"read", "write"}:
            raise ValueError(f"invalid exact replay demand kind {kind!r}")
        portal = int(record["portal"])
        portal_node = int(record["portal_node"])
        bank = int(record["bank"])
        num_bytes = int(record["num_bytes"])
        allocation = int(record["allocation_bytes"])
        live_bytes = int(record["live_bytes"])
        if (
            portal < 0 or portal_node < 0 or portal_node >= cores or bank < 0
            or num_bytes <= 0 or allocation <= 0 or live_bytes <= 0
            or num_bytes % allocation or live_bytes > allocation
        ):
            raise ValueError("invalid exact replay placement geometry or sizes")
        previous_node = portal_nodes.setdefault(portal, portal_node)
        if previous_node != portal_node:
            raise ValueError("one portal maps to multiple injection nodes")
        consumers = tuple(sorted(set(map(int, record["consumers"]))))
        assigned = tuple(sorted(set(map(int, record["assigned_consumers"]))))
        producers = tuple(sorted(set(map(int, record["producers"]))))
        if (
            any(core < 0 or core >= cores for core in (*consumers, *assigned, *producers))
            or not set(assigned) <= set(consumers)
        ):
            raise ValueError("exact replay names invalid consumers or producers")
        name = _physical_name(record)
        if name in seen_physical:
            raise ValueError(f"duplicate physical replica {name}")
        seen_physical.add(name)
        tensors[name] = {
            "bytes": allocation, "readonly": kind == "read",
            "placement_policy": source_policy,
        }
        descriptor = {
            "tensor_id": name, "total_bytes": allocation,
            "allocation_bytes": allocation, "exact_bank_ids": [bank],
            "exact_channel_ids": [portal],
        }
        placement_inputs.append(descriptor)
        vertical[portal] += num_bytes
        participants.update((portal_node, *consumers, *producers))

        if kind == "read":
            accesses.append({
                "tensor": name, "kind": "read", "core": portal_node,
                "offset": 0, "bytes": allocation,
                "passes": num_bytes // allocation,
            })
            if int(record["replica_id"]) > 0:
                extra_sram[portal_node] += live_bytes
            for consumer in assigned:
                if consumer == portal_node:
                    continue
                broadcasts.append({
                    "src": portal_node, "dst": consumer, "bytes": num_bytes,
                })
        else:
            if len(consumers) != 1:
                raise ValueError("write demand needs one owner consumer")
            owner = consumers[0]
            accesses.append({
                "tensor": name, "kind": "write", "core": owner,
                "offset": 0, "bytes": allocation, "passes": 1,
            })
            demand_id = str(record["demand_id"])
            if demand_id not in reduced_demands:
                reduced_demands.add(demand_id)
                for producer in producers:
                    if producer == owner:
                        continue
                    reductions.append({
                        "src": producer, "dst": owner, "bytes": num_bytes,
                    })

    # Build expected link traffic after callers provide the actual mesh width.
    graph = {
        "schema_version": 1,
        "source": "voxel_exact_external_block_replay_v1",
        "execution_model": "aggregate",
        "tensors": tensors,
        "jobs": [{
            "id": "operator", "deps": [], "cores": sorted(participants),
            "compute_cycles": int(compute_cycles),
            "scratch_bytes_per_core": int(scratch_bytes_per_core) + max(extra_sram.values(), default=0),
            "work": 0,
            "accesses": accesses,
            "broadcast": broadcasts,
            "shift": [], "reduce": reductions,
            "exact_extra_sram_bytes_by_core": {
                str(core): value for core, value in sorted(extra_sram.items())
            },
            "steps": [{
                "cores": list(compute_cores),
                "compute_cycles": int(compute_cycles),
                "reduce_compute_cycles": 0,
                "reduce_cores": list(compute_cores),
                "accesses": accesses, "broadcast": broadcasts,
                "shift": [], "reduce": reductions, "carry_bytes": {},
            }],
        }],
        "exact_replay": {
            "schema_version": 1, "source_policy": str(source_policy),
            "placement_hash": _canonical_hash(records),
            "expected_channel_bytes": dict(sorted(vertical.items())),
            "record_count": len(records),
        },
    }
    plan = build_placement_plan(
        placement_inputs,
        total_banks=total_banks,
        num_channels=num_channels,
        policy="exact_external",
    )
    for descriptor in placement_inputs:
        signature = record_signature(descriptor)
        mapping[str(descriptor["tensor_id"])] = plan[signature].bank_ids
    return {"graph": graph, "mapping": mapping, "placement_records": records}


def _link_vector(metrics: Mapping) -> Counter[tuple[int, int]]:
    output = Counter()
    for key, value in metrics.get("link_bytes", {}).items():
        source, destination = map(int, str(key).split(":"))
        output[source, destination] += int(value)
    return output


def replay_exact_placement(
    placement_records: Iterable[Mapping], hardware: Hardware, *,
    compute_cores: Sequence[int], compute_cycles: int,
    scratch_bytes_per_core: int, source_policy: str,
    trace: bool = False, max_tasks: int = 2_000_000,
) -> dict:
    """Execute and audit an externally chosen block placement."""
    contract = build_exact_replay_graph(
        placement_records, cores=hardware.cores, compute_cores=compute_cores,
        compute_cycles=compute_cycles,
        scratch_bytes_per_core=scratch_bytes_per_core,
        source_policy=source_policy, total_banks=hardware.banks,
        num_channels=hardware.channels,
    )
    records = contract["placement_records"]
    for record in records:
        portal = int(record["portal"])
        bank = int(record["bank"])
        if portal >= hardware.channels or bank >= hardware.banks:
            raise ValueError("exact placement exceeds configured DRAM geometry")
        if hardware.bank_channels[bank] != portal:
            raise ValueError("exact placement bank is not attached to its declared portal")
        if hardware.channel_nodes[portal] != int(record["portal_node"]):
            raise ValueError("exact placement portal node differs from hardware")

    result = execute(
        contract["graph"], hardware, contract["mapping"],
        paradigm="compute_shift", trace=trace, max_tasks=max_tasks,
    )
    metrics = result["metrics"]
    expected_channels = [0] * hardware.channels
    expected_links: Counter[tuple[int, int]] = Counter()
    reduced_demands = set()
    for record in records:
        portal = int(record["portal"])
        node = int(record["portal_node"])
        size = int(record["num_bytes"])
        expected_channels[portal] += size
        if record["kind"] == "read":
            for consumer in record["assigned_consumers"]:
                for link in route(node, int(consumer), hardware):
                    expected_links[link] += size
        else:
            owner = int(record["consumers"][0])
            for link in route(owner, node, hardware):
                expected_links[link] += size
            demand_id = str(record["demand_id"])
            if demand_id not in reduced_demands:
                reduced_demands.add(demand_id)
                for producer in record["producers"]:
                    for link in route(int(producer), owner, hardware):
                        expected_links[link] += size
    actual_channels = [
        int(metrics.get("channel_bytes", {}).get(channel, 0))
        for channel in range(hardware.channels)
    ]
    actual_links = _link_vector(metrics)
    if actual_channels != expected_channels:
        raise AssertionError(
            f"exact replay changed TSV vector: {actual_channels} != {expected_channels}"
        )
    if actual_links != expected_links:
        raise AssertionError(
            f"exact replay changed NoC vector: {actual_links} != {expected_links}"
        )
    result["exact_replay"] = {
        **contract["graph"]["exact_replay"],
        "channel_load_bytes": actual_channels,
        "link_load_bytes": [
            [source, destination, value]
            for (source, destination), value in sorted(actual_links.items())
        ],
        "critical_tsv_bytes": max(actual_channels, default=0),
        "critical_noc_link_bytes": max(actual_links.values(), default=0),
        "traffic_invariants_passed": True,
    }
    return result
