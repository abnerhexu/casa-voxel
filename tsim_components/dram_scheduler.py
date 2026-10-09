"""Placement-aware DRAM event scheduling.

The scheduler consumes the same tensor access records exported by TSim and
turns them into compact per-bank row runs.  Timing, row-buffer statistics,
and conflict energy can therefore use one source of truth without expanding
large tensors into one Python object per DRAM transaction.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import Dict, List, Mapping, MutableMapping, Sequence, Tuple

from tsim_components.dram_placement import (
    allocation_bytes,
    build_placement_plan,
    channel_injection_nodes,
    record_bytes,
    requester_core_weights,
    record_signature,
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
    num_bytes: int = 0
    row_hits: int = 0
    row_misses: int = 0
    row_conflicts: int = 0

    def as_dict(self) -> Dict[str, int]:
        return {
            "cycles": int(self.cycles),
            "bytes": int(self.num_bytes),
            "row_hits": int(self.row_hits),
            "row_misses": int(self.row_misses),
            "row_conflicts": int(self.row_conflicts),
        }


@dataclass
class DRAMScheduleResult:
    read: DRAMStageStats
    write: DRAMStageStats
    records: List[dict]


class DRAMExecutionSession:
    """Stateful open-row model shared by timing and energy accounting.

    Open-row state persists between fused operators.  Each read or write
    stage starts a fresh relative timing window because the outer TSim
    scheduler assigns its absolute start time.  A bounded hit-first window is
    used as a lightweight FR-FCFS approximation; no tRAS constraint is
    modelled.
    """

    def __init__(
        self,
        dram,
        placement_policy: str = "software_aware",
        replication_factor: int = 1,
        frfcfs_window: int = 32,
        noc=None,
    ) -> None:
        self.dram = dram
        self.placement_policy = str(placement_policy).lower().replace("-", "_")
        self.replication_factor = max(1, int(replication_factor))
        self.frfcfs_window = max(1, int(frfcfs_window))
        self.noc = noc
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
            self.placement_policy,
            seed=0,
            stripe_bytes=geometry.transaction_bytes,
            noc=self.noc,
        )

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

        pending = list(runs)
        bank_free = [0] * int(self.dram.geometry.total_banks)
        channel_free = [0] * int(self.dram.geometry.num_channels)
        record_start: Dict[int, int] = {}
        record_finish: Dict[int, int] = {}
        record_counts: Dict[int, List[int]] = {}

        while pending:
            # Bounded FR-FCFS approximation.  First choose the request that
            # can reach its channel bus earliest, then prefer a row hit among
            # requests tied at that ready time.  This avoids idling a channel
            # behind an older request whose bank is still busy while keeping
            # the scheduler O(window) instead of expanding transaction-level
            # traces.
            def candidate_key(candidate: int) -> Tuple[int, int, int]:
                candidate_run = pending[candidate]
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
                ready = max(
                    bank_free[candidate_run.bank_id] + candidate_row_cycles,
                    channel_free[candidate_run.channel_id],
                )
                return ready, 0 if candidate_hit else 1, candidate

            choice = min(
                range(min(self.frfcfs_window, len(pending))),
                key=candidate_key,
            )
            run = pending.pop(choice)
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
            transfer_start = max(row_ready, channel_free[run.channel_id])
            finish = transfer_start + transfer_cycles
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
        for record_index, counts in record_counts.items():
            record = records[record_index]
            record["row_hits"] = int(counts[0])
            record["row_misses"] = int(counts[1])
            record["total_row_conflicts"] = int(counts[2])
            record["scheduled_cycles"] = int(
                record_finish[record_index] - record_start[record_index]
            )
        return stats

    def schedule_records(
        self,
        access_records: Sequence[Mapping[str, object]],
        op_index: int,
    ) -> DRAMScheduleResult:
        """Place and schedule one fused operator's selected DRAM accesses."""
        records = [dict(record) for record in access_records]
        runs, plan = self._row_runs(records, op_index)
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
            if self.placement_policy == "noc_aware":
                nodes = channel_injection_nodes(
                    int(self.dram.geometry.num_channels),
                    int(self.noc.num_cores),
                )
                record["channel_noc_nodes"] = [
                    nodes[channel] for channel in placement.channel_ids
                ]

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
        return DRAMScheduleResult(read=read, write=write, records=records)
