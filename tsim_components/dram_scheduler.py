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
from math import ceil
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


@dataclass
class DRAMStageStats:
    cycles: int = 0
    dram_cycles: int = 0
    noc_cycles: int = 0
    num_bytes: int = 0
    row_hits: int = 0
    row_misses: int = 0
    row_conflicts: int = 0
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


def _dram_noc_route_profiles(noc, channel_nodes, demand, stage):
    """Return cached channel-by-link traffic fractions for one demand shape."""
    total_demand = sum(weight for _core, weight in demand)
    demand_key = tuple(
        (int(core), round(float(weight) / total_demand, 15))
        for core, weight in demand
    )
    cache = getattr(noc, "_dram_noc_route_profile_cache", None)
    if cache is None:
        cache = {}
        noc._dram_noc_route_profile_cache = cache
    key = (tuple(int(node) for node in channel_nodes), demand_key, str(stage))
    cached = cache.get(key)
    if cached is not None:
        return cached

    links = _directed_noc_links(noc)
    link_index = {link: index for index, link in enumerate(links)}
    profiles = np.zeros((len(channel_nodes), len(links)), dtype=np.float64)
    average_hops = np.zeros(len(channel_nodes), dtype=np.float64)
    max_hops = np.zeros(len(channel_nodes), dtype=np.int64)
    for channel, endpoint in enumerate(channel_nodes):
        for (core, _weight), (_key_core, fraction) in zip(demand, demand_key):
            source, destination = (
                (int(endpoint), int(core))
                if stage == "read" else (int(core), int(endpoint))
            )
            path = noc.get_dimension_ordered_path(source, destination)
            hops = max(0, len(path) - 1)
            average_hops[channel] += fraction * hops
            max_hops[channel] = max(max_hops[channel], hops)
            for u, v in zip(path, path[1:]):
                profiles[channel, link_index[(int(u), int(v))]] += fraction
    cached = (profiles, average_hops, max_hops, links)
    # A sweep normally reuses only a small set of requester shapes, but cap
    # this potentially large channel-by-link matrix cache for adversarial or
    # imported traces with one unique demand vector per tensor.
    if len(cache) >= 128:
        cache.pop(next(iter(cache)))
    cache[key] = cached
    return cached


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

        profiles, average_hops, channel_max_hops, profile_links = (
            _dram_noc_route_profiles(
                noc, channel_nodes, demand, stage
            )
        )
        if profile_links != directed_links:
            raise AssertionError("cached DRAM NoC link ordering changed")
        channel_vector = np.zeros(num_channels, dtype=np.float64)
        for channel, channel_bytes in channel_weights.items():
            channel_vector[channel] = float(channel_bytes)
        record_link_loads = channel_vector @ profiles
        stage_link_loads += record_link_loads
        record_byte_hops = float(channel_vector @ average_hops)
        active_channels = np.flatnonzero(channel_vector)
        record_max_hops = int(
            max((channel_max_hops[channel] for channel in active_channels), default=0)
        )
        record_flow_count = int(len(active_channels) * len(demand))
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


class DRAMExecutionSession:
    """Stateful open-row model shared by timing and energy accounting.

    Open-row state persists between fused operators.  Each read or write
    stage starts a fresh relative timing window because the outer TSim
    scheduler assigns its absolute start time. The default bounded policy
    selects earliest-ready, using a row hit only to break ties. No tRAS
    constraint is modelled. Optional controls retain the same stage boundary.
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
        if counterfactual not in ("none", "ideal_row_switch", "ideal_dram_noc", "ideal_dram"):
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
        return max(
            1e-12,
            float(self.dram.channel_bytes_per_cycle) * self.replication_factor,
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
        self._placement_plan = build_placement_plan(
            self._prepared_records,
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
                   "total_row_conflicts", "row_hits", "row_misses", "source"}
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
            stats.cycles = int(ceil(stats.num_bytes / (
                float(self.dram.total_bytes_per_cycle) * self.replication_factor
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
        channel_bytes = [0] * len(channel_free)
        bank_bytes = [0] * len(bank_free)
        bank_busy_integral = 0

        while pending:
            # Bounded FR-FCFS approximation.  First choose the request that
            # can reach its channel bus earliest, then prefer a row hit among
            # requests tied at that ready time.  This avoids idling a channel
            # behind an older request whose bank is still busy while keeping
            # the scheduler O(window) instead of expanding transaction-level
            # traces.
            def candidate_key(candidate: int) -> Tuple[int, int, int]:
                candidate_run = self._service_chunk(pending[candidate])[0]
                candidate_open = self._open_rows[candidate_run.bank_id]
                candidate_hit = candidate_open == candidate_run.row_start
                candidate_miss = candidate_open < 0
                candidate_conflicts = (
                    (0 if candidate_hit or candidate_miss else 1)
                    + max(0, candidate_run.row_count - 1)
                )
                candidate_row_cycles = (
                    candidate_run.row_count * int(self.dram.CL)
                    + (1 if candidate_miss else 0) * int(self.dram.tRCD)
                    + candidate_conflicts
                    * (int(self.dram.tRP) + int(self.dram.tRCD))
                )
                if self.counterfactual == "ideal_row_switch":
                    candidate_row_cycles = candidate_run.row_count * int(self.dram.CL)
                elif self.counterfactual == "ideal_dram":
                    candidate_row_cycles = 0
                ready = max(
                    bank_free[candidate_run.bank_id] + candidate_row_cycles,
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
            open_row = self._open_rows[run.bank_id]
            first_hit = open_row == run.row_start
            first_miss = open_row < 0
            hits = 1 if first_hit else 0
            misses = 1 if first_miss else 0
            conflicts = (0 if first_hit or first_miss else 1) + max(0, run.row_count - 1)

            # One CL per touched row.  Closed-row accesses pay ACT/tRCD;
            # replacements pay PRE+tRCD.  tRAS is intentionally absent.
            row_cycles = (
                run.row_count * int(self.dram.CL)
                + misses * int(self.dram.tRCD)
                + conflicts * (int(self.dram.tRP) + int(self.dram.tRCD))
            )
            row_ready = bank_free[run.bank_id] + row_cycles
            transfer_cycles = max(
                1, int(ceil(run.num_bytes / self._chip_channel_bytes_per_cycle))
            )
            if self.counterfactual == "ideal_row_switch":
                row_ready = bank_free[run.bank_id] + run.row_count * int(self.dram.CL)
            elif self.counterfactual == "ideal_dram":
                row_ready = bank_free[run.bank_id]
                transfer_cycles = 0
            transfer_start = max(row_ready, channel_free[run.channel_id])
            finish = transfer_start + transfer_cycles
            if self.instrument:
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
                               "conflicts": conflicts})
            decisions += 1
            bank_free[run.bank_id] = finish
            channel_free[run.channel_id] = finish
            self._open_rows[run.bank_id] = run.row_start + run.row_count - 1

            stats.row_hits += hits
            stats.row_misses += misses
            stats.row_conflicts += conflicts
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

        stats.cycles = max(channel_free, default=0)
        if self.instrument:
            channels = len(channel_free)
            # Idle causes cannot be inferred from bus gaps alone. Keep them
            # explicitly unclassified, rather than inventing timing stalls.
            tail = sum(stats.cycles - end for end in channel_free)
            stats.resource = {
                "channel_bytes": channel_bytes, "bank_bytes": bank_bytes,
                "channel_busy_cycles": channel_busy, "channel_finish": channel_free,
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
        prefix = DRAMRowRun(run.record_index, run.bank_id, run.channel_id,
                            run.row_start, budget, size)
        suffix = DRAMRowRun(run.record_index, run.bank_id, run.channel_id,
                            run.row_start + budget, run.row_count - budget,
                            run.num_bytes - size)
        return prefix, suffix

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
        self.snapshot["operations"][str(op_index)] = {
            "logical_hash": self.logical_hash(access_records),
            "records": [self.logical_record(r) for r in access_records],
            "runs": [asdict(run) for run in runs],
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
        if self.counterfactual == "ideal_dram_noc":
            # Remove only time. Physical traffic/energy remain accounted for.
            for stats in (read, write):
                stats.noc_cycles = 0
                stats.cycles = stats.dram_cycles
            for record in records:
                record["dram_noc_cycles"] = 0
        result = DRAMScheduleResult(read=read, write=write, records=records)
        if self.observer is not None:
            self.observer(self, op_index, result)
        return result
