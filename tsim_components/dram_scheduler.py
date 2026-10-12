"""Placement-aware DRAM event scheduling.

The scheduler consumes the same tensor access records exported by TSim and
turns them into compact per-bank row runs.  Timing, row-buffer statistics,
and conflict energy can therefore use one source of truth without expanding
large tensors into one Python object per DRAM transaction.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
from math import ceil, lcm
from typing import Dict, List, Mapping, MutableMapping, Sequence, Tuple

import numpy as np

from tsim_components.dram_placement import (
    allocation_bytes,
    build_placement_plan,
    channel_injection_nodes,
    record_bytes,
    requester_core_weights,
    record_signature,
    TensorPlacement,
)
from tsim_components.noc import (
    ALL_INIT_CYCLES,
    CUSTOM_INIT_CYCLES,
    MESH_INIT_CYCLES,
    NOC_ROUTER_PIPELINE_CYCLES_PER_HOP,
    TORUS_INIT_CYCLES,
    Topo,
)


@dataclass(frozen=True)
class DRAMRowRun:
    """Consecutive physical rows of one tensor assigned to one bank."""

    record_index: int
    bank_id: int
    channel_id: int
    row_start: int
    row_count: int
    num_bytes: int
    physical_final_row: int | None = None


@dataclass
class DRAMStageStats:
    cycles: int = 0
    dram_cycles: int = 0
    noc_cycles: int = 0
    num_bytes: int = 0
    row_hits: int = 0
    row_misses: int = 0
    row_conflicts: int = 0
    stall_causing_row_conflicts: int = 0
    noc_byte_hops: float = 0.0
    noc_max_link_bytes: float = 0.0
    noc_max_hops: int = 0
    noc_flow_count: int = 0
    resource: dict = field(default_factory=dict)
    trace: list = field(default_factory=list)

    def as_dict(self) -> Dict[str, object]:
        return {
            "cycles": int(self.cycles),
            "dram_cycles": int(self.dram_cycles),
            "noc_cycles": int(self.noc_cycles),
            "bytes": int(self.num_bytes),
            "row_hits": int(self.row_hits),
            "row_misses": int(self.row_misses),
            "row_conflicts": int(self.row_conflicts),
            "stall_causing_row_conflicts": int(
                self.stall_causing_row_conflicts
            ),
            "stall_causing_row_conflict_ratio": (
                float(self.stall_causing_row_conflicts) / self.row_conflicts
                if self.row_conflicts else 0.0
            ),
            "noc_byte_hops": float(self.noc_byte_hops),
            "noc_max_link_bytes": float(self.noc_max_link_bytes),
            "noc_max_hops": int(self.noc_max_hops),
            "noc_flow_count": int(self.noc_flow_count),
            **({"resource": self.resource} if self.resource else {}),
            **({"trace": self.trace} if self.trace else {}),
        }


@dataclass
class DRAMScheduleResult:
    read: DRAMStageStats
    write: DRAMStageStats
    records: List[dict]


@dataclass
class DRAMConcurrentResult:
    """Joint read/write result for one compiler-concurrent operator stage."""

    stats: DRAMStageStats
    records: List[dict]


def completion_breakdown(stats):
    """A DRAM-reference completion decomposition, NOT idle-cause inference.

    Each bank is attached to one channel. Its previous completion cannot be
    later than that channel's previous completion. Thus each gap before a
    transfer is exposed aggregate row readiness in this row-run model.
    NoC extension is an algebraic max-model residual, not measured backpressure.
    """
    r = stats.resource
    if not r or "channel_finish" not in r:
        return
    critical = max(range(len(r["channel_finish"])), key=lambda c: (r["channel_finish"][c], -c))
    transfer = r["channel_busy_cycles"][critical]
    gap = r["channel_row_ready_gap"][critical]
    if transfer + gap != stats.dram_cycles:
        raise AssertionError("critical-channel time does not conserve DRAM completion")
    r["completion"] = {
        "critical_channel": critical, "critical_transfer": transfer,
        "critical_row_ready_gap": gap,
        "noc_extension": max(0, stats.noc_cycles - stats.dram_cycles),
        "memory_cycles": max(stats.dram_cycles, stats.noc_cycles),
        "active_channels": sum(v > 0 for v in r["channel_bytes"]),
        "channel_count": len(r["channel_bytes"]),
        "max_mean_load": (max(r["channel_bytes"]) * len(r["channel_bytes"]) / stats.num_bytes
                          if stats.num_bytes else 0),
    }


def _noc_stage_startup_cycles(noc) -> int:
    configured = getattr(noc, "dram_noc_startup_cycles", None)
    if configured is not None:
        return int(configured)
    if noc.topology in (Topo.MESH, Topo.MESH3D):
        return int(MESH_INIT_CYCLES)
    if noc.topology in (Topo.TORUS, Topo.TORUS3D):
        return int(TORUS_INIT_CYCLES)
    if noc.topology == Topo.ALL:
        return int(ALL_INIT_CYCLES)
    return int(CUSTOM_INIT_CYCLES)


def _directed_noc_links(noc) -> Tuple[Tuple[int, int], ...]:
    cached = getattr(noc, "_dram_noc_directed_links", None)
    if cached is not None:
        return cached
    links = tuple(
        (int(source), int(destination))
        for source, row in enumerate(noc.interconnect_graph)
        for destination, connected in enumerate(row)
        if source != destination and connected
    )
    noc._dram_noc_directed_links = links
    return links


def _locality_matched_dram_flows(
    noc,
    channel_nodes,
    channel_weights: Mapping[int, float],
    demand: Sequence[Tuple[int, float]],
    explicit_flows=None,
) -> Tuple[Tuple[int, int, float, Tuple[int, ...]], ...]:
    """Match placed channel shards to requester shards by physical locality.

    ``requester_core_weights`` describes a partitioned tensor: its weights
    are mutually exclusive byte shards, not a broadcast fanout.  Applying
    the complete requester distribution independently to every channel would
    destroy dimension-ordered tile mapping and route most tensor bytes across
    the middle of the mesh.  Instead, greedily match remaining channel supply
    and requester demand by shortest dimension-ordered path.  Stable
    channel/core tie breaks make this a deterministic placement replay.
    """
    supplies = {
        int(channel): float(weight)
        for channel, weight in channel_weights.items()
        if float(weight) > 0
    }
    demands = {
        int(core): float(weight)
        for core, weight in demand
        if float(weight) > 0
    }
    supply_total = sum(supplies.values())
    demand_total = sum(demands.values())
    if supply_total <= 0 or demand_total <= 0:
        return tuple()
    if explicit_flows is not None:
        flows = []
        supplied: Dict[int, float] = {}
        requested: Dict[int, float] = {}
        for item in explicit_flows:
            if len(item) != 3:
                raise ValueError(
                    "channel_requester_flows entries must be "
                    "[channel, core, bytes]"
                )
            channel, core, amount = int(item[0]), int(item[1]), float(item[2])
            if channel not in supplies or core not in demands or amount <= 0:
                raise ValueError("explicit channel/requester flow is outside demand")
            endpoint = int(channel_nodes[channel])
            path = tuple(int(node) for node in noc.get_dimension_ordered_path(
                endpoint, core
            ))
            flows.append((channel, core, amount, path))
            supplied[channel] = supplied.get(channel, 0.0) + amount
            requested[core] = requested.get(core, 0.0) + amount
        tolerance = max(1e-6, supply_total * 1e-12)
        if any(abs(supplied.get(channel, 0.0) - amount) > tolerance
               for channel, amount in supplies.items()):
            raise ValueError("explicit flows do not conserve per-channel supply")
        normalized_demand = {
            core: amount * supply_total / demand_total
            for core, amount in demands.items()
        }
        if any(abs(requested.get(core, 0.0) - amount) > tolerance
               for core, amount in normalized_demand.items()):
            raise ValueError("explicit flows do not conserve requester demand")
        return tuple(flows)
    # Imported traces may encode relative requester weights. Normalize them
    # to the exact scheduled byte supply before constructing physical flows.
    scale = supply_total / demand_total
    demands = {core: weight * scale for core, weight in demands.items()}
    paths = {}
    for channel in supplies:
        endpoint = int(channel_nodes[channel])
        for core in demands:
            path = tuple(int(node) for node in noc.get_dimension_ordered_path(
                endpoint, core
            ))
            paths[(channel, core)] = path

    flows = []
    epsilon = max(1e-9, supply_total * 1e-12)
    while supplies and demands:
        channel, core = min(
            (
                (channel, core)
                for channel in supplies
                for core in demands
            ),
            key=lambda pair: (
                len(paths[pair]) - 1,
                pair[0],
                pair[1],
            ),
        )
        amount = min(supplies[channel], demands[core])
        flows.append((channel, core, amount, paths[(channel, core)]))
        supplies[channel] -= amount
        demands[core] -= amount
        if supplies[channel] <= epsilon:
            del supplies[channel]
        if demands[core] <= epsilon:
            del demands[core]
    if supplies or demands:
        raise AssertionError("DRAM channel/requester matching did not conserve bytes")
    return tuple(flows)


def _attach_dram_noc_stats(
    *,
    stats: DRAMStageStats,
    stage: str,
    runs: Sequence[DRAMRowRun],
    records: List[MutableMapping[str, object]],
    noc,
    num_channels: int,
) -> None:
    """Account for channel-to-requester traffic without packet expansion.

    Each record's bytes are first aggregated by the channels selected by its
    physical banks. Requester demand then partitions each channel's bytes.
    Dimension-ordered paths produce both byte-hops and directed-link loads;
    the most heavily loaded link determines payload latency.
    """
    stats.dram_cycles = int(stats.cycles)
    if noc is None or not runs or stats.num_bytes <= 0:
        completion_breakdown(stats)
        return
    if not getattr(noc, "exact_topo", False):
        raise ValueError("DRAM-to-core NoC accounting requires an exact topology")

    channel_nodes = channel_injection_nodes(num_channels, int(noc.num_cores))
    per_record_channels: Dict[int, Dict[int, float]] = {}
    for run in runs:
        weights = per_record_channels.setdefault(run.record_index, {})
        weights[run.channel_id] = (
            weights.get(run.channel_id, 0.0) + float(run.num_bytes)
        )

    directed_links = _directed_noc_links(noc)
    stage_link_loads = np.zeros(len(directed_links), dtype=np.float64)
    stage_byte_hops = 0.0
    stage_max_hops = 0
    stage_flow_count = 0
    for record_index, channel_weights in per_record_channels.items():
        record = records[record_index]
        total_channel_bytes = sum(channel_weights.values())
        expected_record_bytes = float(record_bytes(record))
        if abs(total_channel_bytes - expected_record_bytes) > max(
            1e-6, expected_record_bytes * 1e-12
        ):
            raise AssertionError(
                "DRAM channel byte accounting is not conservative: "
                f"scheduled={total_channel_bytes}, "
                f"record={expected_record_bytes}"
            )
        demand = requester_core_weights(record)
        if not demand:
            # Backward-compatible fallback for imported/legacy records. New
            # TSim records always carry explicit requester weights.
            demand = ((0, total_channel_bytes),)
        total_demand = sum(weight for _core, weight in demand)
        if total_demand <= 0:
            continue

        flows = _locality_matched_dram_flows(
            noc, channel_nodes, channel_weights, demand,
            explicit_flows=record.get("channel_requester_flows"),
        )
        link_index = {link: index for index, link in enumerate(directed_links)}
        record_link_loads = np.zeros(len(directed_links), dtype=np.float64)
        record_byte_hops = 0.0
        record_max_hops = 0
        for _channel, _core, flow_bytes, forward_path in flows:
            path = (
                forward_path
                if stage == "read" else tuple(reversed(forward_path))
            )
            hops = max(0, len(path) - 1)
            record_byte_hops += float(flow_bytes) * hops
            record_max_hops = max(record_max_hops, hops)
            for source, destination in zip(path, path[1:]):
                record_link_loads[
                    link_index[(int(source), int(destination))]
                ] += float(flow_bytes)
        stage_link_loads += record_link_loads
        record_flow_count = len(flows)
        record_max_link = float(np.max(record_link_loads, initial=0.0))
        record_noc_cycles = 0
        if record_max_link > 0:
            record_noc_cycles = (
                _noc_stage_startup_cycles(noc)
                + int(ceil(record_max_link / float(noc.bandwidth_bytepc)))
                + record_max_hops * int(getattr(
                    noc,
                    "router_pipeline_cycles_per_hop",
                    NOC_ROUTER_PIPELINE_CYCLES_PER_HOP,
                ))
            )
        record["dram_noc_channel_ids"] = [
            int(channel) for channel in sorted(channel_weights)
        ]
        record["channel_noc_nodes"] = [
            int(channel_nodes[channel]) for channel in sorted(channel_weights)
        ]
        record["dram_noc_byte_hops"] = float(record_byte_hops)
        record["dram_noc_max_hops"] = int(record_max_hops)
        record["dram_noc_max_link_bytes"] = float(record_max_link)
        record["dram_noc_cycles"] = int(record_noc_cycles)
        record["scheduled_cycles"] = max(
            int(record.get("scheduled_cycles", 0)), int(record_noc_cycles)
        )
        stage_byte_hops += record_byte_hops
        stage_max_hops = max(stage_max_hops, record_max_hops)
        stage_flow_count += record_flow_count

    stats.noc_byte_hops = float(stage_byte_hops)
    stats.noc_max_link_bytes = float(np.max(stage_link_loads, initial=0.0))
    stats.noc_max_hops = int(stage_max_hops)
    stats.noc_flow_count = int(stage_flow_count)
    stats.resource.setdefault("noc_directed_links", [
        [int(source), int(destination)] for source, destination in directed_links
    ])
    stats.resource["noc_link_load_bytes"] = [
        float(value) for value in stage_link_loads
    ]
    if stats.noc_max_link_bytes > 0:
        stats.noc_cycles = (
            _noc_stage_startup_cycles(noc)
            + int(ceil(
                stats.noc_max_link_bytes / float(noc.bandwidth_bytepc)
            ))
            + stats.noc_max_hops * int(getattr(
                noc,
                "router_pipeline_cycles_per_hop",
                NOC_ROUTER_PIPELINE_CYCLES_PER_HOP,
            ))
        )
    if abs(float(np.sum(stage_link_loads)) - stats.noc_byte_hops) > max(
        1e-6, stats.noc_byte_hops * 1e-12
    ):
        raise AssertionError(
            "DRAM NoC link loads do not conserve byte-hops: "
            f"links={float(np.sum(stage_link_loads))}, "
            f"byte_hops={stats.noc_byte_hops}"
        )
    # DRAM and the mesh stream independent 128-B transactions. Their payload
    # phases therefore overlap and the slower resource determines completion.
    stats.cycles = max(int(stats.dram_cycles), int(stats.noc_cycles))
    completion_breakdown(stats)


class DRAMExecutionSession:
    """Stateful open-row model shared by timing and energy accounting.

    Open-row state persists between fused operators.  Each read or write
    stage starts a fresh relative timing window because the outer TSim
    scheduler assigns its absolute start time. The default bounded policy
    selects earliest-ready, using a row hit only to break ties. When DRAM.tRAS
    is nonzero, every within-stage PRE obeys PRE >= ACT+tRAS. An inherited
    open row is treated as tRAS-mature because absolute inter-stage time is
    unavailable. Optional controls retain the same stage boundary.
    """

    def __init__(
        self,
        dram,
        placement_policy: str = "software_aware",
        replication_factor: int = 1,
        frfcfs_window: int = 32,
        noc=None,
        seed: int = 0,
        instrument: bool = False,
        trace: bool = False,
        frozen: dict | None = None,
        observer=None,
        policy: str = "current",
        row_budget: int | None = None,
        max_bypass: int = 32,
        fixed_bank_order: bool = False,
        concurrent_window_model: bool = False,
        counterfactual: str = "none",
        counterfactual_op_indices=None,
    ) -> None:
        self.dram = dram
        self.placement_policy = str(placement_policy).lower().replace("-", "_")
        self.replication_factor = max(1, int(replication_factor))
        self.frfcfs_window = max(1, int(frfcfs_window))
        self.noc = noc
        self.seed = int(seed)
        self.instrument = instrument or trace
        self.trace_enabled = trace
        self.frozen = frozen
        self.observer = observer
        if policy not in ("current", "fcfs", "age_protected"):
            raise ValueError(f"unknown scheduling policy: {policy}")
        if row_budget is not None and row_budget < 1:
            raise ValueError("row_budget must be positive or None")
        if max_bypass < 1:
            raise ValueError("max_bypass must be positive")
        self.policy = policy
        self.row_budget = row_budget
        self.max_bypass = max_bypass
        self.fixed_bank_order = fixed_bank_order
        self.concurrent_window_model = bool(concurrent_window_model)
        if counterfactual not in (
            "none", "no_row_conflict_penalty", "ideal_row_switch",
            "ideal_dram_noc", "ideal_dram", "ideal_memory",
        ):
            raise ValueError("unknown counterfactual")
        self.counterfactual = counterfactual
        self.requested_counterfactual = counterfactual
        self.counterfactual_op_indices = counterfactual_op_indices
        self.snapshot = {"version": 1, "operations": {}}
        self._open_rows = [-1] * int(dram.geometry.total_banks)
        self._placement_plan = None
        self._tensor_addresses: Dict[object, int] = {}
        self._prepared_records: List[dict] = []

    @property
    def _chip_channel_bytes_per_cycle(self) -> float:
        buses = (
            max(1, int(getattr(self.dram, "tsv_buses_per_channel", 1)))
            if bool(getattr(
                self.dram, "tsv_bus_bandwidth_scales_service", False
            )) else 1
        )
        return max(
            1e-12,
            float(self.dram.channel_bytes_per_cycle)
            * self.replication_factor
            * buses,
        )

    def _row_runs(
        self,
        records: Sequence[Mapping[str, object]],
        op_index: int,
    ) -> Tuple[List[DRAMRowRun], Dict[object, object]]:
        geometry = self.dram.geometry
        if self._placement_plan is None:
            self.prepare_records(records)
        plan = self._placement_plan
        if self.frozen is not None:
            saved = self.frozen["operations"][str(op_index)]
            if saved["logical_hash"] != self.logical_hash(records):
                raise ValueError("frozen replay logical access mismatch")
            frozen_runs = [DRAMRowRun(**{
                **run, "channel_id": geometry.decode_bank(run["bank_id"])[0]
            }) for run in saved["runs"]]
            per_record_bytes = [0] * len(records)
            for run in frozen_runs:
                if (not 0 <= run.record_index < len(records) or run.row_count < 1 or
                        run.num_bytes < 1 or run.row_start < 0 or not 0 <= run.bank_id < geometry.total_banks):
                    raise ValueError("invalid frozen row run")
                placement = plan.get(record_signature(records[run.record_index]))
                if placement is None or run.bank_id not in placement.bank_ids:
                    raise ValueError("frozen row run is outside tensor bank placement")
                per_record_bytes[run.record_index] += run.num_bytes
            if per_record_bytes != [record_bytes(r) for r in records]:
                raise ValueError("frozen row runs do not conserve logical bytes")
            return frozen_runs, plan
        missing = {
            record_signature(record) for record in records
            if record_bytes(record) > 0
            and record_signature(record) not in plan
        }
        if missing:
            raise ValueError(
                "DRAM execution session saw tensors absent from its global "
                f"placement plan: {sorted(missing)}"
            )
        runs: List[DRAMRowRun] = []
        for record_index, record in enumerate(records):
            size = record_bytes(record)
            placement = plan.get(record_signature(record))
            if size <= 0 or placement is None or not placement.bank_ids:
                continue
            granularity = max(
                geometry.transaction_bytes,
                min(
                    int(record.get("access_granularity_bytes") or geometry.bytes_per_row),
                    geometry.bytes_per_row,
                ),
            )
            # A tiling with short contiguous accesses can touch more logical
            # rows than a streaming tensor of the same size.  Keep both the
            # physical row-capacity lower bound and the tiling-derived access
            # count, matching TSim's established mapping assumption.
            total_row_touches = max(
                int(ceil(size / geometry.bytes_per_row)),
                int(ceil(size / granularity)),
            )
            active_banks = placement.bank_ids[:min(
                len(placement.bank_ids), total_row_touches
            )]
            byte_quotient, byte_remainder = divmod(size, len(active_banks))
            row_quotient, row_remainder = divmod(total_row_touches, len(active_banks))
            signature = record_signature(record)
            base_address = self._tensor_addresses[signature]
            base_row = base_address // geometry.bytes_per_row
            max_rows_per_bank = int(ceil(total_row_touches / len(active_banks)))
            for bank_ordinal, bank_id in enumerate(active_banks):
                bank_bytes = byte_quotient + (1 if bank_ordinal < byte_remainder else 0)
                if bank_bytes <= 0:
                    continue
                row_count = row_quotient + (1 if bank_ordinal < row_remainder else 0)
                # Tensor regions are stable within an op and deliberately
                # distinct between tensors.  Consecutive rows are represented
                # by one run, keeping scheduling cost O(records * banks).
                row_start = base_row + bank_ordinal * max_rows_per_bank
                channel_id, _bank, _layer, _bank_in_layer = \
                    geometry.decode_bank(bank_id)
                runs.append(DRAMRowRun(
                    record_index=record_index,
                    bank_id=int(bank_id),
                    channel_id=int(channel_id),
                    row_start=int(row_start),
                    row_count=row_count,
                    num_bytes=int(bank_bytes),
                    physical_final_row=int(row_start + row_count - 1),
                ))
        return runs, plan

    def prepare_records(
        self,
        access_records: Sequence[Mapping[str, object]],
    ) -> None:
        """Allocate motif-global tensor addresses and build one placement.

        The allocation is deterministic and transaction-aligned. Explicit
        addresses supplied by an imported trace are preserved; tensors
        without one receive a non-overlapping logical base address. The same
        plan is then shared by every fused operator in this execution session.
        """
        geometry = self.dram.geometry
        geometry_key = {
            "banks": int(geometry.total_banks),
            "row_bytes": int(geometry.bytes_per_row),
            "transaction_bytes": int(geometry.transaction_bytes),
        }
        self.snapshot.update(geometry=geometry_key,
                             logical_hash=self.logical_hash(access_records))
        if self.frozen is not None:
            if self.frozen.get("version") != 1:
                raise ValueError("unsupported frozen replay version")
            if (self.frozen["geometry"] != geometry_key or
                    self.frozen["logical_hash"] != self.snapshot["logical_hash"]):
                raise ValueError("frozen replay geometry/logical input mismatch")
            self._placement_plan = {}
            for entry in self.frozen["allocations"]:
                signature = record_signature(entry["record"])
                banks = tuple(entry["bank_ids"])
                if not banks or any(b < 0 or b >= geometry.total_banks for b in banks):
                    raise ValueError("invalid frozen bank IDs")
                self._placement_plan[signature] = TensorPlacement(
                    self.placement_policy, banks,
                    tuple(sorted({geometry.decode_bank(b)[0] for b in banks})))
                self._tensor_addresses[signature] = int(entry["address"])
                if entry["address"] < 0:
                    raise ValueError("negative frozen allocation address")
                capacity = getattr(self.dram, "capacity_bytes", None)
                if capacity is not None and entry["address"] + allocation_bytes(entry["record"]) > capacity:
                    raise ValueError("frozen allocation exceeds capacity")
            self.snapshot["allocations"] = self.frozen["allocations"]
            return
        by_signature: Dict[object, dict] = {}
        requester_demands: Dict[object, Dict[int, float]] = {}
        for source in access_records:
            if record_bytes(source) <= 0:
                continue
            record = dict(source)
            signature = record_signature(record)
            demand = requester_demands.setdefault(signature, {})
            for core, weight in requester_core_weights(record):
                demand[core] = demand.get(core, 0.0) + weight
            size = allocation_bytes(record)
            previous = by_signature.get(signature)
            if previous is None or size > allocation_bytes(previous):
                if previous is not None and previous.get("address") is not None:
                    record.setdefault("address", previous["address"])
                record["allocation_bytes"] = size
                by_signature[signature] = record
            elif previous.get("address") is None and record.get("address") is not None:
                previous["address"] = int(record["address"])

        for signature, record in by_signature.items():
            record["requester_core_weights"] = [
                [core, weight]
                for core, weight in sorted(requester_demands[signature].items())
            ]

        alignment = max(1, int(geometry.transaction_bytes))
        explicit_ranges = sorted(
            (
                int(record["address"]),
                int(record["address"]) + allocation_bytes(record),
                signature,
            )
            for signature, record in by_signature.items()
            if record.get("address") is not None
        )
        for previous, current in zip(explicit_ranges, explicit_ranges[1:]):
            if current[0] < previous[1]:
                raise ValueError(
                    "explicit tensor address ranges overlap: "
                    f"{previous[2]} and {current[2]}"
                )
        cursor = max((end for _start, end, _signature in explicit_ranges), default=0)
        for signature in sorted(by_signature):
            record = by_signature[signature]
            size = allocation_bytes(record)
            address = record.get("address")
            if address is None:
                cursor = int(ceil(cursor / alignment) * alignment)
                address = cursor
            address = int(address)
            record["address"] = address
            self._tensor_addresses[signature] = address
            cursor = max(cursor, address + size)

        capacity = getattr(self.dram, "capacity_bytes", None)
        if capacity is not None and cursor > int(capacity):
            raise ValueError(
                f"tensor allocations require {cursor} bytes, exceeding "
                f"DRAM capacity {capacity} bytes"
            )
        self._prepared_records = list(by_signature.values())
        placement_records = (
            access_records
            if self.placement_policy == "software_aware"
            else self._prepared_records
        )
        self._placement_plan = build_placement_plan(
            placement_records,
            geometry.total_banks,
            geometry.num_channels,
            "software_aware" if self.placement_policy.startswith("stage_aware") else self.placement_policy,
            seed=self.seed,
            stripe_bytes=geometry.transaction_bytes,
            noc=self.noc,
        )
        if self.placement_policy.startswith("stage_aware"):
            from tsim_components.dram_stage_placement import stage_aware_plan
            self._placement_plan = stage_aware_plan(
                access_records, self._placement_plan, geometry, self.noc,
                use_load=self.placement_policy != "stage_aware_no_load",
                use_distance=self.placement_policy != "stage_aware_no_distance")
        self.snapshot["allocations"] = [{
            "record": self.logical_record(record), "address": self._tensor_addresses[record_signature(record)],
            "bank_ids": list(self._placement_plan[record_signature(record)].bank_ids),
        } for record in self._prepared_records]

    @staticmethod
    def logical_record(record):
        # get_dram_time also emits preliminary analytical timing/conflicts.
        # Those are outputs of hardware, not part of the fixed access demand.
        outputs = {"cycles_per_core", "scheduled_cycles", "row_conflicts_per_core",
                   "total_row_conflicts", "stall_causing_row_conflicts",
                   "row_hits", "row_misses", "source"}
        return {k: v for k, v in record.items() if k not in outputs}

    @classmethod
    def logical_hash(cls, records):
        return hashlib.sha256(json.dumps([cls.logical_record(r) for r in records], sort_keys=True,
                                        separators=(",", ":")).encode()).hexdigest()

    def _schedule_stage(
        self,
        runs: Sequence[DRAMRowRun],
        records: List[MutableMapping[str, object]],
    ) -> DRAMStageStats:
        stats = DRAMStageStats(num_bytes=sum(run.num_bytes for run in runs))
        if not runs:
            return stats
        if self.dram.use_sram:
            buses = (
                max(1, int(getattr(self.dram, "tsv_buses_per_channel", 1)))
                if bool(getattr(
                    self.dram, "tsv_bus_bandwidth_scales_service", False
                )) else 1
            )
            stats.cycles = int(ceil(stats.num_bytes / (
                float(self.dram.total_bytes_per_cycle)
                * self.replication_factor
                * buses
            )))
            return stats

        # Lazy row-budget slicing keeps memory O(input runs), not O(rows).
        pending = list(runs)
        input_indices = list(range(len(runs)))
        bypasses = [0] * len(runs)
        decisions = 0
        first_visible = {}
        bank_free = [0] * int(self.dram.geometry.total_banks)
        channel_free = [0] * int(self.dram.geometry.num_channels)
        record_start: Dict[int, int] = {}
        record_finish: Dict[int, int] = {}
        record_counts: Dict[int, List[int]] = {}
        events = []
        channel_busy = [0] * len(channel_free)
        channel_row_ready_gap = [0] * len(channel_free)
        channel_bytes = [0] * len(channel_free)
        bank_bytes = [0] * len(bank_free)
        bank_busy_integral = 0
        record_stall_conflicts: Dict[int, int] = {}

        # tRAS timestamps are local to this stage.  Rows left open by an
        # earlier stage/operator are conservatively treated as already old
        # enough to precharge because the outer TSim timeline is not passed
        # into this relative stage scheduler.
        activated_at: List[int | None] = [None] * len(bank_free)

        def row_timing(candidate_run: DRAMRowRun):
            """Return compact sequential row timing for one bank run.

            Each ACT records its issue time.  A later PRE is delayed until
            ACT+tRAS, then pays tRP before the next ACT.  With tRAS=0 this is
            algebraically identical to the legacy CL/tRCD/tRP row-run cost.
            """
            bank = candidate_run.bank_id
            cursor = int(bank_free[bank])
            open_row = self._open_rows[bank]
            first_hit = open_row == candidate_run.row_start
            first_miss = open_row < 0
            hits = 1 if first_hit else 0
            misses = 1 if first_miss else 0
            conflicts = (
                (0 if first_hit or first_miss else 1)
                + max(0, candidate_run.row_count - 1)
            )
            conflict_delay = 0
            active = activated_at[bank]

            if first_hit:
                cursor += int(self.dram.CL)
            elif first_miss:
                active = cursor
                cursor += int(self.dram.tRCD) + int(self.dram.CL)
            else:
                precharge = cursor
                if active is not None:
                    precharge = max(
                        precharge, active + int(getattr(self.dram, "tRAS", 0))
                    )
                conflict_delay += (
                    precharge - cursor + int(self.dram.tRP) + int(self.dram.tRCD)
                )
                active = precharge + int(self.dram.tRP)
                cursor = active + int(self.dram.tRCD) + int(self.dram.CL)

            remaining = max(0, candidate_run.row_count - 1)
            # Rows in a run are consecutive and share identical timing after
            # the first ACT, so apply the recurrence in closed form.  This is
            # what keeps multi-GiB tensors O(records*banks), not O(rows).
            if remaining and active is None:
                conflict_delay += int(self.dram.tRP) + int(self.dram.tRCD)
                active = cursor + int(self.dram.tRP)
                cursor = active + int(self.dram.tRCD) + int(self.dram.CL)
                remaining -= 1
            if remaining:
                active_to_data = int(self.dram.tRCD) + int(self.dram.CL)
                tras_wait = max(
                    0,
                    int(getattr(self.dram, "tRAS", 0)) - active_to_data,
                )
                act_step = max(
                    active_to_data,
                    int(getattr(self.dram, "tRAS", 0)),
                ) + int(self.dram.tRP)
                conflict_delay += remaining * (
                    tras_wait + int(self.dram.tRP) + int(self.dram.tRCD)
                )
                active += remaining * act_step
                cursor = active + active_to_data
            return {
                "row_ready": cursor,
                "no_conflict_row_ready": cursor - conflict_delay,
                "final_activation": active,
                "hits": hits,
                "misses": misses,
                "conflicts": conflicts,
                "conflict_delay": conflict_delay,
            }

        while pending:
            # Bounded FR-FCFS approximation.  First choose the request that
            # can reach its channel bus earliest, then prefer a row hit among
            # requests tied at that ready time.  This avoids idling a channel
            # behind an older request whose bank is still busy while keeping
            # the scheduler O(window) instead of expanding transaction-level
            # traces.
            def candidate_key(candidate: int) -> Tuple[int, int, int]:
                candidate_run = self._service_chunk(pending[candidate])[0]
                timing = row_timing(candidate_run)
                candidate_hit = bool(timing["hits"])
                candidate_row_ready = int(timing["row_ready"])
                if self.counterfactual == "no_row_conflict_penalty":
                    candidate_row_ready = int(timing["no_conflict_row_ready"])
                elif self.counterfactual == "ideal_row_switch":
                    candidate_row_ready = (
                        bank_free[candidate_run.bank_id]
                        + candidate_run.row_count * int(self.dram.CL)
                    )
                elif self.counterfactual in ("ideal_dram", "ideal_memory"):
                    candidate_row_ready = bank_free[candidate_run.bank_id]
                ready = max(
                    candidate_row_ready,
                    channel_free[candidate_run.channel_id],
                )
                return ready, 0 if candidate_hit else 1, candidate

            eligible = list(range(min(self.frfcfs_window, len(pending))))
            if self.trace_enabled:
                for index in eligible:
                    first_visible.setdefault(input_indices[index], decisions)
            if self.fixed_bank_order:
                seen = set()
                heads = []
                for index in eligible:
                    if pending[index].bank_id not in seen:
                        heads.append(index)
                    seen.add(pending[index].bank_id)
                eligible = heads
            protected = [i for i in eligible if bypasses[i] >= self.max_bypass]
            reason = "earliest_ready_hit_tiebreak"
            if self.policy == "fcfs":
                choice, reason = eligible[0], "fcfs"
            elif self.policy == "age_protected" and protected:
                choice, reason = protected[0], "age_protection"
            else:
                choice = min(eligible, key=candidate_key)
            input_index = input_indices[choice]
            run, remainder = self._service_chunk(pending[choice])
            visible_alternatives = [
                self._service_chunk(pending[index])[0]
                for index in eligible
                if index != choice
            ]
            bypass_count = bypasses[choice]
            for index in eligible:
                bypasses[index] += 1
            if remainder is None:
                pending.pop(choice)
                input_indices.pop(choice)
                bypasses.pop(choice)
            else:
                pending[choice] = remainder
                bypasses[choice] = 0
            timing = row_timing(run)
            hits = int(timing["hits"])
            misses = int(timing["misses"])
            conflicts = int(timing["conflicts"])
            row_ready = int(timing["row_ready"])
            transfer_cycles = max(
                1, int(ceil(run.num_bytes / self._chip_channel_bytes_per_cycle))
            )
            if self.counterfactual == "no_row_conflict_penalty":
                # Retain the initial ACT/tRCD miss and all column latency, but
                # remove only PRE/ACT delay caused by replacing an open row.
                # This is the counterfactual needed for the paper's shaded
                # row-buffer-conflict execution-time overhead.
                row_ready = int(timing["no_conflict_row_ready"])
            elif self.counterfactual == "ideal_row_switch":
                row_ready = bank_free[run.bank_id] + run.row_count * int(self.dram.CL)
            elif self.counterfactual in ("ideal_dram", "ideal_memory"):
                row_ready = bank_free[run.bank_id]
                transfer_cycles = 0

            # Attribute conflicts to a TSV-bus stall only when their timing
            # penalty is exposed after the channel becomes free and no other
            # visible request on another bank of this channel can start data
            # transfer sooner.  The compact row-run model counts every
            # sequential conflict in the selected run under that condition.
            stall_conflicts = 0
            if (
                conflicts > 0
                and self.counterfactual == "none"
                and row_ready > channel_free[run.channel_id]
                and row_ready > max(
                    channel_free[run.channel_id],
                    int(timing["no_conflict_row_ready"]),
                )
            ):
                alternative_ready = []
                for alt_run in visible_alternatives:
                    if (
                        alt_run.channel_id == run.channel_id
                        and alt_run.bank_id != run.bank_id
                    ):
                        alternative_ready.append(
                            max(
                                int(row_timing(alt_run)["row_ready"]),
                                channel_free[alt_run.channel_id],
                            )
                        )
                if not alternative_ready or min(alternative_ready) >= row_ready:
                    stall_conflicts = conflicts
            transfer_start = max(row_ready, channel_free[run.channel_id])
            finish = transfer_start + transfer_cycles
            if self.instrument:
                if bank_free[run.bank_id] > channel_free[run.channel_id]:
                    raise AssertionError("bank/channel predecessor order is inconsistent")
                channel_row_ready_gap[run.channel_id] += transfer_start - channel_free[run.channel_id]
                channel_busy[run.channel_id] += transfer_cycles
                channel_bytes[run.channel_id] += run.num_bytes
                bank_bytes[run.bank_id] += run.num_bytes
                bank_busy_integral += finish - bank_free[run.bank_id]
            if self.trace_enabled:
                events.append({**asdict(run), "input_index": input_index,
                               "selection_reason": reason, "bypass_count": bypass_count,
                               "decision_index": decisions,
                               "visible_decision": first_visible[input_index],
                               "release": 0, "bank_start": bank_free[run.bank_id],
                               "row_ready": row_ready, "transfer_start": transfer_start,
                               "finish": finish, "hits": hits, "misses": misses,
                               "conflicts": conflicts,
                               "stall_causing_conflicts": stall_conflicts,
                               "conflict_delay_cycles": int(timing["conflict_delay"])})
            decisions += 1
            bank_free[run.bank_id] = finish
            channel_free[run.channel_id] = finish
            self._open_rows[run.bank_id] = (
                run.physical_final_row
                if run.physical_final_row is not None
                else run.row_start + run.row_count - 1
            )
            activated_at[run.bank_id] = timing["final_activation"]

            stats.row_hits += hits
            stats.row_misses += misses
            stats.row_conflicts += conflicts
            stats.stall_causing_row_conflicts += stall_conflicts
            record_start[run.record_index] = min(
                record_start.get(run.record_index, transfer_start), transfer_start
            )
            record_finish[run.record_index] = max(
                record_finish.get(run.record_index, 0), finish
            )
            counts = record_counts.setdefault(run.record_index, [0, 0, 0])
            counts[0] += hits
            counts[1] += misses
            counts[2] += conflicts
            record_stall_conflicts[run.record_index] = (
                record_stall_conflicts.get(run.record_index, 0)
                + stall_conflicts
            )

        stats.cycles = max(channel_free, default=0)
        if self.instrument:
            channels = len(channel_free)
            # Idle causes cannot be inferred from bus gaps alone. Keep them
            # explicitly unclassified, rather than inventing timing stalls.
            tail = sum(stats.cycles - end for end in channel_free)
            stats.resource = {
                "channel_bytes": channel_bytes, "bank_bytes": bank_bytes,
                "channel_busy_cycles": channel_busy, "channel_finish": channel_free,
                "channel_row_ready_gap": channel_row_ready_gap,
                "channel_time": channels * stats.cycles,
                "transfer": sum(channel_busy), "tail_idle": tail,
                "unclassified_idle": channels * stats.cycles - sum(channel_busy) - tail,
                "bank_busy_integral": bank_busy_integral,
                "service_events": decisions,
                "input_row_touches": sum(run.row_count for run in runs),
            }
            if self.trace_enabled:
                stats.trace = events
                stats.resource["stage_release_to_bus_start_p95"] = float(np.percentile(
                    [event["transfer_start"] for event in events], 95))
                stats.resource["wait_sample_unit"] = "serviced_row_chunk"
        for record_index, counts in record_counts.items():
            record = records[record_index]
            record["row_hits"] = int(counts[0])
            record["row_misses"] = int(counts[1])
            record["total_row_conflicts"] = int(counts[2])
            record["stall_causing_row_conflicts"] = int(
                record_stall_conflicts.get(record_index, 0)
            )
            record["scheduled_cycles"] = int(
                record_finish[record_index] - record_start[record_index]
            )
        return stats

    def _service_chunk(self, run):
        if self.row_budget is None or run.row_count <= self.row_budget:
            return run, None
        budget = self.row_budget
        quotient, remainder = divmod(run.num_bytes, run.row_count)
        size = quotient * budget + min(budget, remainder)
        if size <= 0:
            raise ValueError("row budget would produce zero-byte service")
        prefix = DRAMRowRun(
            run.record_index, run.bank_id, run.channel_id,
            run.row_start, budget, size, run.row_start + budget - 1,
        )
        suffix = DRAMRowRun(
            run.record_index, run.bank_id, run.channel_id,
            run.row_start + budget, run.row_count - budget,
            run.num_bytes - size, run.physical_final_row,
        )
        return prefix, suffix

    def _concurrent_window_runs(
        self,
        runs: Sequence[DRAMRowRun],
    ) -> List[DRAMRowRun]:
        """Compactly model interleaved tensor streams in one bank.

        A normal row run is atomic and therefore cannot express the paper's
        same-bank concurrent-access penalty.  For a bank shared by ``k``
        tensor streams, a round-robin producer exposes at most
        ``floor(FR-FCFS window / k)`` transactions from one stream before the
        controller must expose another stream.  Row boundaries and these
        visibility-quantum boundaries both split a run into row-buffer
        service segments.  Their union is counted in closed form, retaining
        O(records*banks) cost instead of expanding 128-byte transactions.

        When only one tensor uses a bank, the original physical row count is
        preserved exactly.  This makes the extra conflicts depend solely on
        concurrent tensors sharing a bank, which is the condition targeted
        by software-aware placement.
        """
        streams_per_bank: Dict[int, set[int]] = {}
        for run in runs:
            streams_per_bank.setdefault(run.bank_id, set()).add(
                run.record_index
            )
        stream_counts = {
            bank: len(record_indices)
            for bank, record_indices in streams_per_bank.items()
        }
        transaction_bytes = max(1, int(self.dram.geometry.transaction_bytes))
        adjusted = []
        for run in runs:
            stream_count = stream_counts[run.bank_id]
            if stream_count <= 1:
                adjusted.append(run)
                continue
            transactions = max(1, int(ceil(run.num_bytes / transaction_bytes)))
            quantum = max(1, self.frfcfs_window // stream_count)
            transactions_per_row = max(
                1, int(ceil(transactions / run.row_count))
            )
            row_boundaries = min(
                run.row_count - 1,
                (transactions - 1) // transactions_per_row,
            )
            quantum_boundaries = (transactions - 1) // quantum
            shared_boundaries = (transactions - 1) // lcm(
                transactions_per_row, quantum
            )
            service_segments = min(
                transactions,
                1 + row_boundaries + quantum_boundaries - shared_boundaries,
            )
            adjusted.append(DRAMRowRun(
                record_index=run.record_index,
                bank_id=run.bank_id,
                channel_id=run.channel_id,
                row_start=run.row_start,
                row_count=service_segments,
                num_bytes=run.num_bytes,
                physical_final_row=(
                    run.physical_final_row
                    if run.physical_final_row is not None
                    else run.row_start + run.row_count - 1
                ),
            ))
        return adjusted

    def schedule_records(
        self,
        access_records: Sequence[Mapping[str, object]],
        op_index: int,
    ) -> DRAMScheduleResult:
        """Place and schedule one fused operator's selected DRAM accesses."""
        records = [dict(record) for record in access_records]
        self.counterfactual = (self.requested_counterfactual
                              if self.counterfactual_op_indices is None or op_index in self.counterfactual_op_indices
                              else "none")
        runs, plan = self._row_runs(records, op_index)
        if self.concurrent_window_model:
            if self.frozen is None:
                runs = self._concurrent_window_runs(runs)
            elif not self.frozen["operations"][str(op_index)].get(
                "concurrent_window_model"
            ):
                raise ValueError(
                    "frozen replay operation lacks concurrent-window metadata"
                )
        self.snapshot["operations"][str(op_index)] = {
            "logical_hash": self.logical_hash(access_records),
            "records": [self.logical_record(r) for r in access_records],
            "runs": [asdict(run) for run in runs],
            "concurrent_window_model": (
                "round_robin_visible_quantum_v1"
                if self.concurrent_window_model else None
            ),
        }
        for record in records:
            signature = record_signature(record)
            placement = plan.get(signature)
            if placement is None:
                continue
            record.update({
                "source": "tsim_dram_execution_session",
                "placement_policy": self.placement_policy,
                "bank_ids": list(placement.bank_ids),
                "channel_ids": list(placement.channel_ids),
                "address": int(self._tensor_addresses[signature]),
                "allocation_bytes": int(allocation_bytes(record)),
            })

        read_runs = [
            run for run in runs
            if str(records[run.record_index].get("stage", "")).lower() == "read"
        ]
        write_runs = [
            run for run in runs
            if str(records[run.record_index].get("stage", "")).lower() == "write"
        ]
        read = self._schedule_stage(read_runs, records)
        write = self._schedule_stage(write_runs, records)
        _attach_dram_noc_stats(
            stats=read,
            stage="read",
            runs=read_runs,
            records=records,
            noc=self.noc,
            num_channels=int(self.dram.geometry.num_channels),
        )
        _attach_dram_noc_stats(
            stats=write,
            stage="write",
            runs=write_runs,
            records=records,
            noc=self.noc,
            num_channels=int(self.dram.geometry.num_channels),
        )
        if self.counterfactual in ("ideal_dram_noc", "ideal_memory"):
            # Remove only time. Physical traffic/energy remain accounted for.
            for stats in (read, write):
                stats.noc_cycles = 0
                stats.cycles = stats.dram_cycles
                completion_breakdown(stats)
            for record in records:
                record["dram_noc_cycles"] = 0
        result = DRAMScheduleResult(read=read, write=write, records=records)
        if self.observer is not None:
            self.observer(self, op_index, result)
        return result

    def schedule_concurrent_records(
        self,
        access_records: Sequence[Mapping[str, object]],
        op_index: int,
    ) -> DRAMConcurrentResult:
        """Schedule one operator's reads and writes as a concurrent stream.

        The software-aware mapping described by the Voxel paper derives
        concurrency from singular/fused operators, including an output write
        overlapping input prefetch.  The legacy :meth:`schedule_records` API
        deliberately retains separate read and write timing stages.  This
        explicit API instead places both directions in one bounded FR-FCFS
        window and is intended for conflict/interference studies.

        Runs are ordered bank-major so records targeting the same bank become
        jointly visible to the controller.  This preserves O(records*banks)
        scheduling cost while avoiding record-major serialization.  Mixed
        direction NoC timing is intentionally unsupported because reads and
        writes route in opposite directions; callers needing NoC latency must
        continue to use :meth:`schedule_records`.
        """
        if self.noc is not None:
            raise ValueError(
                "concurrent read/write DRAM scheduling does not support NoC timing"
            )
        records = [dict(record) for record in access_records]
        self.counterfactual = (
            self.requested_counterfactual
            if self.counterfactual_op_indices is None
            or op_index in self.counterfactual_op_indices
            else "none"
        )
        runs, plan = self._row_runs(records, op_index)
        if self.frozen is None:
            runs = self._concurrent_window_runs(runs)
        elif not self.frozen["operations"][str(op_index)].get(
            "concurrent_window_model"
        ):
            raise ValueError(
                "frozen replay operation lacks concurrent-window metadata"
            )
        runs = sorted(
            runs,
            key=lambda run: (
                run.channel_id,
                run.bank_id,
                run.record_index,
                run.row_start,
            ),
        )
        self.snapshot["operations"][str(op_index)] = {
            "logical_hash": self.logical_hash(access_records),
            "records": [self.logical_record(record) for record in access_records],
            "runs": [asdict(run) for run in runs],
            "concurrent_read_write": True,
            "concurrent_window_model": "round_robin_visible_quantum_v1",
        }
        for record in records:
            signature = record_signature(record)
            placement = plan.get(signature)
            if placement is None:
                continue
            record.update({
                "source": "tsim_dram_execution_session_concurrent",
                "placement_policy": self.placement_policy,
                "bank_ids": list(placement.bank_ids),
                "channel_ids": list(placement.channel_ids),
                "address": int(self._tensor_addresses[signature]),
                "allocation_bytes": int(allocation_bytes(record)),
            })
        stats = self._schedule_stage(runs, records)
        stats.dram_cycles = int(stats.cycles)
        completion_breakdown(stats)
        result = DRAMConcurrentResult(stats=stats, records=records)
        if self.observer is not None:
            self.observer(self, op_index, result)
        return result
