"""Counterfactual NoC-to-DRAM reuse replay for native TSim logs.

The ordinary TSim pipeline chooses one communication source for a compiled
tiling.  Intersect-slope needs mixed points: some operator deliveries remain
on the NoC while others are fetched/materialized through stacked DRAM.  This
module replays the *native* per-operator TSim records with that source choice.

It deliberately consumes :class:`FusedOperatorExecLog` objects rather than
reconstructing tensor traffic from a byte total.  Added input copies are DRAM
reads at the destination.  Added reduction copies are ordered producer writes
followed by owner reads.  Every DRAM phase is scheduled by Voxel's existing
placement-aware bounded FR-FCFS scheduler, including tRAS and bank state.

The replay is an operator-serial dependency schedule.  DRAM records inside a
phase retain scheduler parallelism, and NoC phase costs come from the native
tiling.  This is intentionally conservative for a dependent motif and avoids
claiming packet-level overlap that the compressed TSim trace does not expose.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence


@dataclass(frozen=True)
class ReuseChoice:
    """Chip-wide endpoint payload selected for one operator."""

    input_total_bytes: int
    input_reroute_bytes: int
    reduction_total_bytes: int
    reduction_reroute_bytes: int

    def __post_init__(self):
        values = (
            self.input_total_bytes,
            self.input_reroute_bytes,
            self.reduction_total_bytes,
            self.reduction_reroute_bytes,
        )
        if any(type(value) is not int or value < 0 for value in values):
            raise ValueError("reuse bytes must be nonnegative integers")
        if self.input_reroute_bytes > self.input_total_bytes:
            raise ValueError("input reroute exceeds input NoC payload")
        if self.reduction_reroute_bytes > self.reduction_total_bytes:
            raise ValueError("reduction reroute exceeds reduction NoC payload")

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "ReuseChoice":
        return cls(**{
            field: int(value.get(field, 0))
            for field in (
                "input_total_bytes", "input_reroute_bytes",
                "reduction_total_bytes", "reduction_reroute_bytes",
            )
        })


def _scaled_cycles(cycles: int, remaining: int, total: int) -> int:
    if total <= 0 or remaining <= 0 or cycles <= 0:
        return 0
    return int(math.ceil(int(cycles) * int(remaining) / int(total)))


def _scaled_float(value: float, remaining: int, total: int) -> float:
    if total <= 0 or remaining <= 0 or value <= 0:
        return 0.0
    return float(value) * int(remaining) / int(total)


def _weights(total_bytes: int, cores: int) -> list[list[int]]:
    quotient, remainder = divmod(int(total_bytes), int(cores))
    return [
        [core, quotient + (core < remainder)]
        for core in range(cores)
        if quotient + (core < remainder)
    ]


def _extra_record(
    *, op_index: int, role: str, stage: str, total_bytes: int,
    cores: int, transaction_bytes: int,
) -> dict:
    if total_bytes <= 0:
        raise ValueError("extra reuse record must carry positive traffic")
    tensor_id = f"reuse/{role}/op{op_index:04d}"
    return {
        "version": 1,
        "source": "tsim_reuse_frontier",
        "subop_index": int(op_index),
        "tensor_index": -1,
        "tensor_role": role,
        "tensor_id": tensor_id,
        "stage": stage,
        "allocation_bytes": int(total_bytes),
        "bytes_per_core": int(math.ceil(total_bytes / cores)),
        "total_bytes": int(total_bytes),
        "requester_core_weights": _weights(total_bytes, cores),
        # Rerouted copies are coalesced dense tensor segments. The physical
        # interface still emits ``transaction_bytes`` bursts, while the DRAM
        # scheduler must not reinterpret every 128-B burst as a new row open.
        "access_granularity_bytes": int(max(transaction_bytes, 8192)),
        "row_conflicts_per_core": 0,
        "total_row_conflicts": 0,
        "cycles_per_core": 0,
        "scheduled_cycles": 0,
        "ignored": False,
        "reuse_materialization": True,
    }


def _logical_records(log) -> tuple[list[dict], list[dict]]:
    records = [
        dict(record)
        for record in getattr(log, "spatial_meta", {}).get(
            "dram_access_records", []
        )
    ]
    return (
        [record for record in records if record.get("stage") == "read"],
        [record for record in records if record.get("stage") == "write"],
    )


def replay_reuse_plan(
    logs: Sequence[object],
    dram,
    noc,
    choices: Sequence[ReuseChoice | Mapping[str, object]],
    *,
    placement_policy: str = "software_aware",
    num_cores: int = 32,
) -> dict:
    """Replay one mixed-source policy and return timing and traffic metrics.

    ``choices`` and ``logs`` contain one entry per unfused TSim operator.
    Payload counts are chip-wide endpoint bytes and must use the same logical
    unit as the ski-slope movement profile.
    """
    if len(logs) != len(choices):
        raise ValueError("one reuse choice is required per TSim operator")
    normalized = [
        value if isinstance(value, ReuseChoice)
        else ReuseChoice.from_mapping(value)
        for value in choices
    ]
    transaction = int(getattr(dram, "transaction_bytes", 128))
    for choice in normalized:
        if any(value % transaction for value in (
            choice.input_total_bytes, choice.input_reroute_bytes,
            choice.reduction_total_bytes, choice.reduction_reroute_bytes,
        )):
            raise ValueError("reuse payload must be DRAM-transaction aligned")

    base_records = [_logical_records(log) for log in logs]
    phase_records = []
    all_records = []
    for op_index, ((reads, writes), choice) in enumerate(
        zip(base_records, normalized)
    ):
        reads = list(reads)
        writes = list(writes)
        materialize_write = []
        materialize_read = []
        if choice.input_reroute_bytes:
            reads.append(_extra_record(
                op_index=op_index, role="input_replica", stage="read",
                total_bytes=choice.input_reroute_bytes, cores=num_cores,
                transaction_bytes=transaction,
            ))
        if choice.reduction_reroute_bytes:
            materialize_write.append(_extra_record(
                op_index=op_index, role="partial_sum", stage="write",
                total_bytes=choice.reduction_reroute_bytes, cores=num_cores,
                transaction_bytes=transaction,
            ))
            materialize_read.append(_extra_record(
                op_index=op_index, role="partial_sum", stage="read",
                total_bytes=choice.reduction_reroute_bytes, cores=num_cores,
                transaction_bytes=transaction,
            ))
        phases = (reads, materialize_write, materialize_read, writes)
        phase_records.append(phases)
        all_records.extend(record for phase in phases for record in phase)

    session = dram.new_execution_session(
        placement_policy=placement_policy,
        replication_factor=num_cores,
        noc=noc,
    )
    session.prepare_records(all_records)

    cycles = 0
    read_bytes = write_bytes = 0
    row_conflicts = row_hits = row_misses = 0
    dram_noc_byte_hops = 0.0
    operator_noc_byte_hops = 0.0
    per_operator = []

    def schedule(records: list[dict], op_key: int, direction: str):
        if not records:
            return None
        result = session.schedule_records(records, op_index=op_key)
        return result.read if direction == "read" else result.write

    for op_index, (log, choice, phases) in enumerate(
        zip(logs, normalized, phase_records)
    ):
        reads, materialize_write, materialize_read, writes = phases
        # Unique keys preserve one scheduler snapshot per ordered phase.
        read = schedule(reads, op_index * 4, "read")
        partial_write = schedule(
            materialize_write, op_index * 4 + 1, "write"
        )
        partial_read = schedule(
            materialize_read, op_index * 4 + 2, "read"
        )
        write = schedule(writes, op_index * 4 + 3, "write")

        input_remaining = (
            choice.input_total_bytes - choice.input_reroute_bytes
        )
        reduction_remaining = (
            choice.reduction_total_bytes - choice.reduction_reroute_bytes
        )
        bcast = _scaled_cycles(
            int(getattr(log, "bcast_dur", 0)), input_remaining,
            choice.input_total_bytes,
        )
        shift = _scaled_cycles(
            int(getattr(log, "shift_dur", 0)), input_remaining,
            choice.input_total_bytes,
        )
        reduce = _scaled_cycles(
            int(getattr(log, "reduce_dur", 0)), reduction_remaining,
            choice.reduction_total_bytes,
        )
        compute = int(getattr(log, "comp_dur", 0))
        phase_cycles = {
            "dram_read": int(read.cycles) if read else 0,
            "noc_broadcast": bcast,
            "compute_shift": max(compute, shift),
            "partial_sum_write": (
                int(partial_write.cycles) if partial_write else 0
            ),
            "partial_sum_read": (
                int(partial_read.cycles) if partial_read else 0
            ),
            "noc_reduce": reduce,
            "dram_write": int(write.cycles) if write else 0,
        }
        op_cycles = sum(phase_cycles.values())
        cycles += op_cycles

        scheduled = [
            value for value in (read, partial_write, partial_read, write)
            if value is not None
        ]
        for stats in scheduled:
            row_conflicts += int(stats.row_conflicts)
            row_hits += int(stats.row_hits)
            row_misses += int(stats.row_misses)
            dram_noc_byte_hops += float(stats.noc_byte_hops)
        read_bytes += sum(
            int(stats.num_bytes)
            for stats in (read, partial_read) if stats is not None
        )
        write_bytes += sum(
            int(stats.num_bytes)
            for stats in (partial_write, write) if stats is not None
        )
        input_hops = (
            float(getattr(log, "noc_bcast_byte_hops", 0.0))
            + float(getattr(log, "noc_shift_byte_hops", 0.0))
        )
        reduction_hops = float(
            getattr(log, "noc_reduce_byte_hops", 0.0)
        )
        op_hops = (
            _scaled_float(input_hops, input_remaining,
                          choice.input_total_bytes)
            + _scaled_float(reduction_hops, reduction_remaining,
                            choice.reduction_total_bytes)
        )
        operator_noc_byte_hops += op_hops
        per_operator.append({
            "operator_index": op_index,
            "cycles": op_cycles,
            "phase_cycles": phase_cycles,
            "operator_noc_byte_hops": op_hops,
            "input_reroute_bytes": choice.input_reroute_bytes,
            "reduction_reroute_bytes": choice.reduction_reroute_bytes,
        })

    input_reroute = sum(value.input_reroute_bytes for value in normalized)
    reduction_reroute = sum(
        value.reduction_reroute_bytes for value in normalized
    )
    return {
        "model": "native_tsim_mixed_reuse_replay_v1",
        "cycles": int(cycles),
        "dram_read_bytes": int(read_bytes),
        "dram_write_bytes": int(write_bytes),
        "additional_tsv_bytes": int(input_reroute + 2 * reduction_reroute),
        "removed_noc_bytes": int(input_reroute + reduction_reroute),
        "input_reroute_bytes": int(input_reroute),
        "reduction_reroute_bytes": int(reduction_reroute),
        "row_conflicts": int(row_conflicts),
        "row_hits": int(row_hits),
        "row_misses": int(row_misses),
        "operator_noc_byte_hops": float(operator_noc_byte_hops),
        "dram_noc_byte_hops": float(dram_noc_byte_hops),
        "noc_byte_hops": float(
            operator_noc_byte_hops + dram_noc_byte_hops
        ),
        "per_operator": per_operator,
        "ordering": (
            "read -> broadcast -> compute/shift -> partial write -> "
            "partial read -> reduce -> final write"
        ),
    }
