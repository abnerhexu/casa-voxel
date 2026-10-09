"""Deterministic tensor-to-DRAM placement policies.

The functions in this module are deliberately independent from the thermal
and timing backends.  A placement plan names physical global banks; consumers
may then translate those banks into thermal blocks or into channel/bank/row
requests without reimplementing policy logic.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from functools import lru_cache
from math import ceil, floor
from typing import Dict, Iterable, Mapping, Sequence, Tuple


RecordSignature = Tuple[str, int, int, str, str]

SUPPORTED_DRAM_PLACEMENTS = frozenset({
    "address_trace",
    "hbm_interleave",
    "uniform",
    "interleave_size",
    "software_aware",
    "channel_aware",
})


def record_signature(record: Mapping[str, object]) -> RecordSignature:
    tensor_id = record.get("tensor_id")
    if tensor_id is not None:
        # A physical tensor keeps one placement across reads, writes,
        # operators, and layers. String conversion also supports synthetic
        # IDs used by tests and imported traces.
        return (str(tensor_id), 0, 0, "", "")
    return (
        "",
        int(record.get("subop_index", 0) or 0),
        int(record.get("tensor_index", 0) or 0),
        str(record.get("tensor_role", "tensor")),
        str(record.get("stage", "")),
    )


def record_bytes(record: Mapping[str, object]) -> int:
    return max(0, int(
        record.get("total_bytes") or record.get("bytes_per_core") or 0
    ))


def allocation_bytes(record: Mapping[str, object]) -> int:
    """Return physical allocation size, distinct from transfer traffic."""
    return max(0, int(
        record.get("allocation_bytes") or record_bytes(record)
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


def stable_u64(*parts: object) -> int:
    """Return a process-independent 64-bit hash for placement decisions."""
    payload = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little")


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
    channel_banks = [
        tuple(range(channel, total_banks, num_channels))
        for channel in range(num_channels)
    ]
    active_channels = [
        channel for channel, banks in enumerate(channel_banks) if banks
    ]
    valid_records = [record for record in records if record_bytes(record) > 0]
    if not valid_records:
        return {}

    total_bytes = sum(record_bytes(record) for record in valid_records)
    ordered = sorted(
        valid_records,
        key=lambda record: (-record_bytes(record), record_signature(record)),
    )
    channel_load = [0.0] * num_channels
    read_cursor = [
        int(seed + channel * 17) % max(1, len(channel_banks[channel]))
        for channel in range(num_channels)
    ]
    write_cursor = [
        int(seed + channel * 17 + max(1, len(channel_banks[channel]) // 2))
        % max(1, len(channel_banks[channel]))
        for channel in range(num_channels)
    ]
    result: Dict[RecordSignature, TensorPlacement] = {}

    for record in ordered:
        size = record_bytes(record)
        # A record receives channels in proportion to its share of traffic.
        # At least one channel is always selected; very large tensors can use
        # every channel and exploit channel-level parallelism.
        channel_span = max(1, min(
            len(active_channels),
            int(ceil(len(active_channels) * size / max(1, total_bytes))),
        ))
        rotation = int(seed) % num_channels
        selected = sorted(
            active_channels,
            key=lambda channel: (
                channel_load[channel],
                (channel - rotation) % max(1, num_channels),
            ),
        )[:channel_span]

        # Allocate a size-proportional stripe inside each selected channel.
        stage = str(record.get("stage", "")).lower()
        cursors = write_cursor if stage == "write" else read_cursor
        global_banks = []
        for channel in sorted(selected):
            available_banks = channel_banks[channel]
            local_span = max(1, min(
                len(available_banks),
                int(ceil(
                    len(available_banks) * size / max(1, total_bytes)
                )),
            ))
            local_indices = _rotating_indices(
                len(available_banks), cursors[channel], local_span
            )
            global_banks.extend(
                available_banks[local_index]
                for local_index in local_indices
            )
            cursors[channel] = (
                cursors[channel] + local_span
            ) % len(available_banks)
            channel_load[channel] += float(size) / channel_span

        result[record_signature(record)] = TensorPlacement(
            policy="channel_aware",
            bank_ids=tuple(global_banks),
            channel_ids=tuple(sorted(selected)),
        )

    return result


def _build_placement_plan_uncached(
    records: Iterable[Mapping[str, object]],
    total_banks: int,
    num_channels: int,
    policy: str,
    seed: int = 0,
    stripe_bytes: int = 128,
) -> Dict[RecordSignature, TensorPlacement]:
    """Build one deterministic physical-bank plan for any public policy.

    This is the shared placement entry point for timing, conflict/energy,
    and thermal consumers.  It deliberately returns bank IDs rather than
    backend-specific events so every consumer observes the same placement.
    """
    total_banks = max(1, int(total_banks))
    num_channels = max(1, int(num_channels))
    stripe_bytes = max(1, int(stripe_bytes))
    policy = str(policy).lower().replace("-", "_")
    if policy not in SUPPORTED_DRAM_PLACEMENTS:
        raise ValueError(
            f"unsupported DRAM placement {policy!r}; expected one of "
            f"{sorted(SUPPORTED_DRAM_PLACEMENTS)}"
        )

    ordered = sorted(
        (record for record in records if record_bytes(record) > 0),
        key=record_signature,
    )
    if policy == "software_aware":
        return software_aware_placements(ordered, total_banks, seed)
    if policy == "channel_aware":
        return channel_aware_placements(
            ordered, total_banks, num_channels, seed
        )

    max_bytes = max((record_bytes(record) for record in ordered), default=1)
    result: Dict[RecordSignature, TensorPlacement] = {}
    for ordinal, record in enumerate(ordered):
        signature = record_signature(record)
        size = record_bytes(record)
        if policy == "uniform":
            banks = tuple(range(total_banks))
        elif policy == "interleave_size":
            span = max(1, min(
                total_banks,
                int(ceil(total_banks * size / max_bytes)),
            ))
            start = stable_u64(policy, seed, *signature) % total_banks
            banks = _rotating_indices(total_banks, start, span)
        elif policy == "hbm_interleave":
            # Idealized fine-grain HBM interleaving rotates each tensor stripe
            # independently of its allocated physical base. This intentionally
            # differs from address_trace, whose bank order is decoded from the
            # motif-global tensor address.
            start = stable_u64(policy, seed, ordinal, *signature) % total_banks
            span = max(1, min(total_banks, int(ceil(size / stripe_bytes))))
            banks = _rotating_indices(total_banks, start, span)
        else:
            # address_trace stripes the tensor's deterministic synthetic
            # physical address range. If a trace supplies an address, it
            # becomes the starting stripe; otherwise the tensor signature is
            # a stable substitute.
            address = record.get("address")
            if address is None:
                address = stable_u64(policy, seed, ordinal, *signature)
            start = (int(address) // stripe_bytes) % total_banks
            span = max(1, min(total_banks, int(ceil(size / stripe_bytes))))
            banks = _rotating_indices(total_banks, start, span)

        result[signature] = TensorPlacement(
            policy=policy,
            bank_ids=banks,
            channel_ids=tuple(sorted({bank % num_channels for bank in banks})),
        )
    return result


@lru_cache(maxsize=8192)
def _cached_placement_plan(
    record_specs: Tuple[Tuple[RecordSignature, int, object], ...],
    total_banks: int,
    num_channels: int,
    policy: str,
    seed: int,
    stripe_bytes: int,
) -> Tuple[Tuple[RecordSignature, TensorPlacement], ...]:
    records = []
    for signature, size, address in record_specs:
        tensor_id, subop_index, tensor_index, tensor_role, stage = signature
        record = {
            "tensor_id": tensor_id or None,
            "subop_index": subop_index,
            "tensor_index": tensor_index,
            "tensor_role": tensor_role,
            "stage": stage,
            "total_bytes": size,
        }
        if address is not None:
            record["address"] = address
        records.append(record)
    plan = _build_placement_plan_uncached(
        records,
        total_banks=total_banks,
        num_channels=num_channels,
        policy=policy,
        seed=seed,
        stripe_bytes=stripe_bytes,
    )
    return tuple(sorted(plan.items()))


def build_placement_plan(
    records: Iterable[Mapping[str, object]],
    total_banks: int,
    num_channels: int,
    policy: str,
    seed: int = 0,
    stripe_bytes: int = 128,
) -> Dict[RecordSignature, TensorPlacement]:
    """Return a placement plan from the placement-only LRU cache.

    The cache key contains tensor access descriptors, policy, and complete
    channel/bank geometry. It contains no tiling-selection state; changing
    channel count or policy therefore recomputes placement without evicting
    or invalidating the independent tiling cache.
    """
    normalized_policy = str(policy).lower().replace("-", "_")
    # Placement operates on physical allocations, not individual accesses.
    # Coalesce repeated reads/writes of one tensor and use its largest known
    # allocation while retaining one stable base address.
    allocations: Dict[RecordSignature, Tuple[int, object]] = {}
    for record in records:
        size = allocation_bytes(record)
        if size <= 0:
            continue
        signature = record_signature(record)
        address = (
            int(record["address"]) if record.get("address") is not None else None
        )
        previous = allocations.get(signature)
        if previous is None:
            allocations[signature] = (size, address)
        else:
            previous_size, previous_address = previous
            if (
                address is not None and previous_address is not None
                and address != previous_address
            ):
                raise ValueError(
                    f"tensor {signature[0]!r} has inconsistent base addresses"
                )
            allocations[signature] = (
                max(previous_size, size),
                previous_address if previous_address is not None else address,
            )
    record_specs = tuple(sorted(
        (signature, size, address)
        for signature, (size, address) in allocations.items()
    ))
    cached = _cached_placement_plan(
        record_specs,
        max(1, int(total_banks)),
        max(1, int(num_channels)),
        normalized_policy,
        int(seed),
        max(1, int(stripe_bytes)),
    )
    return dict(cached)


def placement_cache_info():
    """Expose placement-cache statistics for experiment diagnostics."""
    return _cached_placement_plan.cache_info()


def clear_placement_cache() -> None:
    """Clear only placement plans; tiling selections are unaffected."""
    _cached_placement_plan.cache_clear()
