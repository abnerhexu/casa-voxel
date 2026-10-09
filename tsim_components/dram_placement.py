"""Deterministic tensor-to-DRAM placement policies.

The functions in this module are deliberately independent from the thermal
and timing backends.  A placement plan names physical global banks; consumers
may then translate those banks into thermal blocks or into channel/bank/row
requests without reimplementing policy logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, floor
from typing import Dict, Iterable, Mapping, Sequence, Tuple


RecordSignature = Tuple[int, int, str, str]

SUPPORTED_DRAM_PLACEMENTS = frozenset({
    "address_trace",
    "hbm_interleave",
    "uniform",
    "interleave_size",
    "software_aware",
    "channel_aware",
})


def record_signature(record: Mapping[str, object]) -> RecordSignature:
    return (
        int(record.get("subop_index", 0) or 0),
        int(record.get("tensor_index", 0) or 0),
        str(record.get("tensor_role", "tensor")),
        str(record.get("stage", "")),
    )


def record_bytes(record: Mapping[str, object]) -> int:
    return max(0, int(
        record.get("total_bytes") or record.get("bytes_per_core") or 0
    ))


@dataclass(frozen=True)
class TensorPlacement:
    """Physical banks selected for one tensor access."""

    policy: str
    bank_ids: Tuple[int, ...]
    channel_ids: Tuple[int, ...]

    def bank_weights(self, total_bytes: int) -> Dict[int, float]:
        total_bytes = max(0, int(total_bytes))
        if total_bytes <= 0 or not self.bank_ids:
            return {}
        weight = float(total_bytes) / len(self.bank_ids)
        return {int(bank): weight for bank in self.bank_ids}

    def channel_weights(self, total_bytes: int, num_channels: int) -> Dict[int, float]:
        weights: Dict[int, float] = {}
        for bank, weight in self.bank_weights(total_bytes).items():
            channel = int(bank) % max(1, int(num_channels))
            weights[channel] = weights.get(channel, 0.0) + float(weight)
        return weights


def _rotating_indices(total: int, start: int, span: int) -> Tuple[int, ...]:
    total = max(1, int(total))
    span = max(1, min(total, int(span)))
    start = int(start) % total
    return tuple((start + offset) % total for offset in range(span))


def _proportional_spans(weights: Sequence[int], capacity: int) -> Tuple[int, ...]:
    """Allocate at least one slot per item, then distribute remaining slots."""
    capacity = max(1, int(capacity))
    if not weights:
        return tuple()
    positive = [max(1, int(weight)) for weight in weights]
    if len(positive) >= capacity:
        return tuple(1 for _ in positive)
    total = sum(positive)
    remaining = capacity - len(positive)
    raw_extra = [remaining * weight / total for weight in positive]
    spans = [1 + int(floor(value)) for value in raw_extra]
    left = capacity - sum(spans)
    order = sorted(
        range(len(spans)),
        key=lambda idx: (raw_extra[idx] - floor(raw_extra[idx]), positive[idx], -idx),
        reverse=True,
    )
    for idx in order[:left]:
        spans[idx] += 1
    return tuple(spans)


def software_aware_placements(
    records: Iterable[Mapping[str, object]],
    total_banks: int,
    seed: int = 0,
) -> Dict[RecordSignature, TensorPlacement]:
    """Reproduce the existing size-proportional contiguous-bank strategy."""
    total_banks = max(1, int(total_banks))
    ordered = sorted(
        (record for record in records if record_bytes(record) > 0),
        key=record_signature,
    )
    spans = _proportional_spans([record_bytes(record) for record in ordered], total_banks)
    cursor = int(seed) % total_banks
    result: Dict[RecordSignature, TensorPlacement] = {}
    for record, span in zip(ordered, spans):
        banks = _rotating_indices(total_banks, cursor, span)
        result[record_signature(record)] = TensorPlacement(
            policy="software_aware",
            bank_ids=banks,
            channel_ids=tuple(),
        )
        cursor += span
    return result


def channel_aware_placements(
    records: Iterable[Mapping[str, object]],
    total_banks: int,
    num_channels: int,
    seed: int = 0,
) -> Dict[RecordSignature, TensorPlacement]:
    """Place tensors with software-aware bank stripes plus channel balancing.

    Large tensors may span several channels.  Records are considered largest
    first and assigned to the currently least-loaded channels.  Inside each
    selected channel, banks are allocated as a deterministic rotating
    contiguous stripe.  Read and write stages use independent bank cursors,
    preserving the software-aware strategy's separation of access roles while
    allowing channel topology to influence placement.
    """
    total_banks = max(1, int(total_banks))
    num_channels = max(1, int(num_channels))
    if total_banks % num_channels:
        raise ValueError(
            f"total_banks ({total_banks}) must be divisible by "
            f"num_channels ({num_channels})"
        )
    banks_per_channel = total_banks // num_channels
    valid_records = [record for record in records if record_bytes(record) > 0]
    if not valid_records:
        return {}

    total_bytes = sum(record_bytes(record) for record in valid_records)
    ordered = sorted(
        valid_records,
        key=lambda record: (-record_bytes(record), record_signature(record)),
    )
    channel_load = [0.0] * num_channels
    read_cursor = [int(seed + channel * 17) % banks_per_channel for channel in range(num_channels)]
    write_cursor = [
        int(seed + channel * 17 + max(1, banks_per_channel // 2)) % banks_per_channel
        for channel in range(num_channels)
    ]
    result: Dict[RecordSignature, TensorPlacement] = {}

    for record in ordered:
        size = record_bytes(record)
        # A record receives channels in proportion to its share of traffic.
        # At least one channel is always selected; very large tensors can use
        # every channel and exploit channel-level parallelism.
        channel_span = max(1, min(
            num_channels,
            int(ceil(num_channels * size / max(1, total_bytes))),
        ))
        rotation = int(seed) % num_channels
        selected = sorted(
            range(num_channels),
            key=lambda channel: (
                channel_load[channel],
                (channel - rotation) % num_channels,
            ),
        )[:channel_span]

        # Allocate a size-proportional stripe inside each selected channel.
        local_span = max(1, min(
            banks_per_channel,
            int(ceil(banks_per_channel * size / max(1, total_bytes))),
        ))
        stage = str(record.get("stage", "")).lower()
        cursors = write_cursor if stage == "write" else read_cursor
        global_banks = []
        for channel in sorted(selected):
            local_banks = _rotating_indices(
                banks_per_channel, cursors[channel], local_span
            )
            global_banks.extend(
                channel + local_bank * num_channels
                for local_bank in local_banks
            )
            cursors[channel] = (cursors[channel] + local_span) % banks_per_channel
            channel_load[channel] += float(size) / channel_span

        result[record_signature(record)] = TensorPlacement(
            policy="channel_aware",
            bank_ids=tuple(global_banks),
            channel_ids=tuple(sorted(selected)),
        )

    return result

