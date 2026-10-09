"""Placement-independent k-best tiling search for multi-operator motifs."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import heapq
import json
from math import ceil
from typing import Any, List, Sequence, Tuple


def _freeze(value: Any) -> Any:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if hasattr(value, "item"):
        return value.item()
    return value


def _jsonable(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if hasattr(value, "item"):
        return value.item()
    return value


@dataclass(frozen=True)
class RankedOperatorTiling:
    """One operator tiling scored without a DRAM placement policy."""

    config: Tuple[Any, Any]
    score_cycles: int
    compute_noc_cycles: int
    transfer_lower_bound_cycles: int
    hot_memory_bytes: int


@dataclass(frozen=True)
class MotifTilingCandidate:
    """One k-best selection containing exactly one tiling per motif op."""

    rank: int
    candidate_id: str
    config_indices: Tuple[int, ...]
    configs: Tuple[Tuple[Any, Any], ...]
    score_cycles: int
    compute_noc_cycles: int
    transfer_lower_bound_cycles: int
    execution_sram_bytes: int

    def as_dict(self) -> dict:
        return {
            "rank": self.rank,
            "candidate_id": self.candidate_id,
            "config_indices": list(self.config_indices),
            "configs": [_jsonable(config) for config in self.configs],
            "placement_independent_lower_bound_cycles": self.score_cycles,
            "compute_noc_cycles": self.compute_noc_cycles,
            "transfer_lower_bound_cycles": self.transfer_lower_bound_cycles,
            "execution_sram_bytes": self.execution_sram_bytes,
        }


def rank_operator_tilings(
    op,
    dram,
    noc,
    core_group_size: int,
    max_memory_bytes: int,
    limit: int | None = None,
) -> List[RankedOperatorTiling]:
    """Rank feasible tilings by a placement-independent latency lower bound.

    Compute and NoC use the compiled mapping and the dimension-ordered NoC
    estimator. DRAM contributes only aggregate transfer time: no bank, row,
    channel, or placement term is included. The two paths may overlap, so
    their maximum is the lower bound used for ranking.
    """
    ranked: List[RankedOperatorTiling] = []
    bandwidth = max(1e-12, float(dram.total_bytes_per_cycle))
    for raw_config, values in op.expr.config_dict.items():
        hot_memory, _exe_time, comp_cycles, _shift_cycles = values
        if int(hot_memory) > int(max_memory_bytes):
            continue
        spatial, temporal = _freeze(raw_config)
        tensor_shapes = op.expr.get_sub_op_var_sizes(temporal, spatial, False)
        temporal_replicas = op.expr.get_temporal_var_replicas(temporal, spatial)
        spatial_replicas = op.expr.get_spatial_var_replicas(temporal, spatial)
        shift_info = op.expr.get_shift_info(
            temporal, spatial, output_special_format_for_tsim=True
        )
        tensor_sizes = op.expr.get_sub_op_var_sizes(temporal, spatial)
        (broadcast, shift, reduce), _traffic = (
            noc.get_total_cycles_and_traffic_hops_from_expression(
                tensor_sizes,
                temporal_replicas,
                spatial_replicas,
                shift_info,
                num_bytes_per_elem=op.num_byte_per_elem,
            )
        )
        compute_noc = int(ceil(
            float(broadcast)
            + max(float(comp_cycles), float(shift))
            + float(reduce)
        ))
        _cycles, byte_counts = dram.get_dram_access_list(
            tensor_shapes,
            temporal_replicas,
            core_group_size,
            op.num_byte_per_elem,
            return_cycles_and_bytes=True,
            for_tiling=True,
        )
        transfer_bytes = int(byte_counts[0])
        transfer_bytes += sum(
            int(num_bytes)
            for num_bytes, ignored in zip(
                byte_counts[1:], op.expr.ignore_variables[1:]
            )
            if not ignored
        )
        transfer_cycles = int(ceil(transfer_bytes / bandwidth))
        ranked.append(RankedOperatorTiling(
            config=(spatial, temporal),
            score_cycles=max(compute_noc, transfer_cycles),
            compute_noc_cycles=compute_noc,
            transfer_lower_bound_cycles=transfer_cycles,
            hot_memory_bytes=int(hot_memory),
        ))

    ranked.sort(key=lambda item: (
        item.score_cycles,
        item.compute_noc_cycles,
        item.transfer_lower_bound_cycles,
        item.hot_memory_bytes,
        repr(item.config),
    ))
    if not ranked:
        raise ValueError(
            f"operator {getattr(op, 'name', '<unnamed>')} has no tiling "
            f"within {max_memory_bytes} bytes"
        )
    return ranked[:limit] if limit is not None else ranked


def enumerate_motif_tiling_candidates(
    prog,
    dram,
    noc,
    core_group_size: int,
    max_memory_bytes: int,
    limit: int = 512,
    per_operator_limit: int = 8,
) -> List[MotifTilingCandidate]:
    """Enumerate up to ``limit`` globally best motif tiling combinations.

    A heap-based k-best Cartesian-product traversal visits only combinations
    reachable from the all-best point. For a 64-op motif and 512 requested
    candidates this avoids materializing ``per_operator_limit ** 64`` points.
    """
    if int(limit) <= 0 or int(per_operator_limit) <= 0:
        raise ValueError("limit and per_operator_limit must be positive")
    pools = [
        rank_operator_tilings(
            op,
            dram,
            noc,
            core_group_size,
            max_memory_bytes,
            limit=per_operator_limit,
        )
        for op in prog.ops
    ]
    if not pools:
        return []

    initial = tuple(0 for _ in pools)

    def totals(indices: Sequence[int]) -> Tuple[int, int, int]:
        chosen = [pool[index] for pool, index in zip(pools, indices)]
        return (
            sum(item.score_cycles for item in chosen),
            sum(item.compute_noc_cycles for item in chosen),
            sum(item.transfer_lower_bound_cycles for item in chosen),
        )

    initial_totals = totals(initial)
    heap = [(initial_totals[0], initial)]
    seen = {initial}
    results: List[MotifTilingCandidate] = []
    while heap and len(results) < int(limit):
        _score, indices = heapq.heappop(heap)
        score, compute_noc, transfer = totals(indices)
        configs = tuple(
            pool[index].config for pool, index in zip(pools, indices)
        )
        identity = hashlib.blake2b(
            json.dumps(
                _jsonable(configs), sort_keys=True, separators=(",", ":")
            ).encode("utf-8"),
            digest_size=12,
        ).hexdigest()
        results.append(MotifTilingCandidate(
            rank=len(results) + 1,
            candidate_id=identity,
            config_indices=indices,
            configs=configs,
            score_cycles=score,
            compute_noc_cycles=compute_noc,
            transfer_lower_bound_cycles=transfer,
            execution_sram_bytes=max(
                pool[index].hot_memory_bytes
                for pool, index in zip(pools, indices)
            ),
        ))
        for operator_index, pool in enumerate(pools):
            next_index = indices[operator_index] + 1
            if next_index >= len(pool):
                continue
            neighbor = list(indices)
            neighbor[operator_index] = next_index
            neighbor_tuple = tuple(neighbor)
            if neighbor_tuple in seen:
                continue
            seen.add(neighbor_tuple)
            neighbor_score = (
                score
                - pool[indices[operator_index]].score_cycles
                + pool[next_index].score_cycles
            )
            heapq.heappush(heap, (neighbor_score, neighbor_tuple))
    return results
