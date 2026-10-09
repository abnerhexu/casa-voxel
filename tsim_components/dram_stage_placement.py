"""Greedy stage-aware bank rotation with explicit load/distance ablations.

Preserves each tensor's software-aware bank span. Scores mean stage peak
channel load and bank sharing, normalized by stage bytes, plus normalized
requester distance. This is a heuristic candidate, not an optimal allocator.
"""
from collections import defaultdict
from tsim_components.dram_placement import (
    TensorPlacement, channel_injection_nodes, record_bytes, record_signature,
    requester_core_weights,
)


def stage_aware_plan(records, initial, geometry, noc, *, use_load=True, use_distance=True):
    demands = defaultdict(lambda: defaultdict(float))
    stage_bytes = defaultdict(float)
    requesters = defaultdict(lambda: defaultdict(float))
    for record in records:
        size = record_bytes(record)
        if size <= 0:
            continue
        if "execution_stage_id" not in record:
            raise ValueError("stage-aware placement requires execution_stage_id metadata")
        sig, stage = record_signature(record), str(record["execution_stage_id"])
        demands[sig][stage] += size
        stage_bytes[stage] += size
        for core, weight in requester_core_weights(record):
            requesters[sig][core] += weight
    channels, banks = geometry.num_channels, geometry.total_banks
    loads = {s: [0.] * channels for s in stage_bytes}
    bank_loads = {s: [0.] * banks for s in stage_bytes}
    nodes = channel_injection_nodes(channels, noc.num_cores) if noc else None
    if use_distance and (noc is None or not noc.exact_topo):
        raise ValueError("stage-aware distance objective requires exact NoC")
    result = {}
    for sig in sorted(demands, key=lambda k: (-sum(demands[k].values()), k)):
        span = len(initial[sig].bank_ids)
        distances = [0.] * channels
        if use_distance:
            total = sum(requesters[sig].values())
            if total <= 0:
                raise ValueError("stage-aware placement requires requester metadata")
            distances = [sum(w * noc.get_hops(core, node)[0] for core, w in requesters[sig].items()) / total
                         for node in nodes]
            norm = max(1., max(distances))
            distances = [d / norm for d in distances]

        def score(start):
            selected = tuple((start + i) % banks for i in range(span))
            weights = [0] * channels
            for bank in selected:
                weights[geometry.decode_bank(bank)[0]] += 1
            value = 0.
            if use_load:
                for stage, size in demands[sig].items():
                    value += (max(loads[stage][c] + size*weights[c]/span for c in range(channels))
                              + sum(bank_loads[stage][b] for b in selected)/span) / stage_bytes[stage]
                value /= len(demands[sig])
            if use_distance:
                value += sum(weights[c]*distances[c] for c in range(channels))/span
            return value, start

        start = min(range(banks), key=score)
        chosen = tuple((start+i) % banks for i in range(span))
        for stage, size in demands[sig].items():
            for bank in chosen:
                loads[stage][geometry.decode_bank(bank)[0]] += size/span
                bank_loads[stage][bank] += size/span
        result[sig] = TensorPlacement("stage_aware", chosen,
                                      tuple(sorted({geometry.decode_bank(b)[0] for b in chosen})))
    return result
