"""Explicit core-group lowering for bounded reuse experiments.

This is a canonical dense-stream schedule, not reconstruction of affine tile
order. Costs use every clipped cell; only spatially active cores divide work.
Whole intermediate residents are conservatively reserved for the entire run.
Use tile_trace for independent, bounded affine validation.
"""
from collections import Counter
from dataclasses import replace
import copy
import math

from .aggregate_trace import export_program
from .tile_trace import audit_specs

VERSION = "core_reuse_stream_v1"


def fit_spatial(spec, core_count):
    result = copy.deepcopy(spec)
    while math.prod(result["spatial"]) > core_count:
        axis = max(range(len(result["spatial"])), key=lambda d: result["spatial"][d])
        result["spatial"][axis] = max(1, result["spatial"][axis] // 2)
    return result


def lower_program(specs, comp, hw, *, paradigm, stages=2, instances=1):
    if paradigm not in ("spmd", "compute_shift", "dataflow"):
        raise ValueError("unsupported paradigm")
    if not specs or instances < 1 or stages < 1 or hw.cores % stages:
        raise ValueError("nonempty program and divisible positive stages required")
    audit = audit_specs(specs)
    if not audit["valid"]:
        raise ValueError("invalid source tensor graph")
    graph = dict(schema_version=1, execution_model="aggregate", lowering=VERSION,
                 source=VERSION, tensors={}, jobs=[], source_audit=audit,
                 independent_instances=instances)
    width = hw.cores // stages if paradigm == "dataflow" else hw.cores
    for instance in range(instances):
        prefix = f"instance{instance}"
        for tensor, size in audit["tensor_bytes"].items():
            graph["tensors"][f"{prefix}/tensor{tensor}"] = dict(
                bytes=size, readonly=tensor not in audit["producers"],
                persist=tensor == specs[-1]["tensors"][0], source_tensor=tensor)
        for index, source in enumerate(specs):
            spec = fit_spatial(source, width)
            active = math.prod(min(s, math.ceil(d / math.ceil(d / s)))
                               for s, d in zip(spec["spatial"], spec["dims"]))
            stage = min(stages-1, index * stages // len(specs)) if paradigm == "dataflow" else 0
            base = stage * width
            local = replace(hw, cores=active, mesh_width=math.gcd(active, hw.mesh_width),
                            channel_nodes=(), bank_channels=())
            lowered = export_program([spec], comp, local, paradigm=paradigm, stages=1)
            job = lowered["jobs"][0]
            job.update(id=f"{prefix}/op{index:04d}", stage=stage, microbatch=instance,
                       schedule_priority=index + instance,
                       source_spec=spec, cores=[base+c for c in job["cores"]])
            job["deps"] = [f"{prefix}/op{index-1:04d}"] if index else []
            if index == 0 and instance and paradigm != "dataflow":
                job["deps"].append(f"instance{instance-1}/op{len(specs)-1:04d}")
            for access in job["accesses"]:
                access["core"] += base
                access["tensor"] = access["tensor"].replace("instance0/", prefix+"/", 1)
            for kind in ("shift", "broadcast", "reduce"):
                for flow in job.get(kind, []):
                    flow["src"] += base
                    flow["dst"] += base
                    flow["tensor"] = flow["tensor"].replace("instance0/", prefix+"/", 1)
            graph["jobs"].append(job)
    graph["semantics"] = (
        "balanced dense streams on active spatial cores; ring sharing; "
        "bulk row-run reservations; dataflow uses disjoint contiguous stage "
        "groups and independent full invocations, never implicit sequence splits")
    return graph


def intermediate_residents(graph, hw):
    """Greedy, capacity-checked whole-tensor forwarding; no free tile cache.

    One owner receives all producer shards. Each chosen tensor remains live
    until run end, so this deliberately overestimates inter-op live storage.
    Final externally visible outputs always drain to backing store.
    """
    scratch = Counter()
    writers, readers = {}, Counter()
    for job in graph["jobs"]:
        for core in job["cores"]:
            scratch[core] = max(scratch[core], job["scratch_bytes_per_core"])
        for access in job["accesses"]:
            if access["kind"] == "write":
                writers.setdefault(access["tensor"], access["core"])
            else:
                readers[access["tensor"]] += access["bytes"] * access.get("passes", 1)
    candidates = [t for t in writers if readers[t] and not graph["tensors"][t].get("persist")]
    candidates.sort(key=lambda t: (-readers[t] / graph["tensors"][t]["bytes"], t))
    used, resident = Counter(), {}
    for tensor in candidates:
        owner, size = writers[tensor], graph["tensors"][tensor]["bytes"]
        if used[owner] + scratch[owner] + size <= hw.sram_bytes:
            resident[tensor] = [owner]
            used[owner] += size
    return resident


def traffic_summary(metrics):
    traffic = metrics["traffic"]
    inputs = sum(traffic.get(k+"_bytes", 0) for k in ("broadcast", "shift", "reuse", "forward"))
    reduction = traffic.get("reduce_bytes", 0)
    return dict(vertical_bytes=traffic.get("read_bytes", 0)+traffic.get("write_bytes", 0),
                input_remote_bytes=inputs, reduction_remote_bytes=reduction,
                remote_bytes=inputs+reduction,
                noc_byte_hops=traffic.get("noc_byte_hops", 0),
                critical_link_bytes=max(metrics["link_bytes"].values(), default=0),
                critical_channel_bytes=max(metrics["channel_bytes"].values(), default=0))
