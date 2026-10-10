"""Analytic tile cost + canonical contiguous streams (not affine tile replay).

Tensor traffic is conserved for a no-cross-cell-reuse tiling baseline. Actual
strided tile order is replaced by dense, repeated per-requester sweeps. This is
an explicit modeling approximation, not lossless compression of tile_trace.
"""
from collections import Counter
from itertools import product
import math
import numpy as np
from .tile_trace import audit_specs


def summary(spec, comp):
    histograms = []
    for d,s,ts in zip(spec["dims"], spec["spatial"], spec["temporal"]):
        spatial = math.ceil(d/s)
        tile = math.ceil(spatial/max(ts))
        lengths = Counter()
        for start in range(0,d,spatial):
            length = min(spatial,d-start)
            full, tail = divmod(length,tile)
            lengths[tile] += full
            if tail:
                lengths[tail] += 1
        histograms.append(lengths)
    ndim, nvars = len(histograms), len(spec["variables"])
    total, work, peak = 0, 0, 0
    for items in product(*(h.items() for h in histograms)):
        shape, counts = zip(*items)
        count = math.prod(counts)
        op = comp.convert_op_simple(dim_len=np.array(shape),
            variables=[np.array([[d] for d in axes]) for axes in spec["variables"]],
            is_ew=spec["op_type"] != 5, ignore_variables=[False]*nvars,
            tensor_id_list=list(range(nvars)), op_type=spec["op_type"])
        total += count*math.ceil(comp.get_total_cycle_for_fused_op([op], [[[1]*nvars for _ in range(ndim)]])[0])
        work += count*math.prod(shape)*(2 if spec["op_type"] == 5 else 1)
        peak = max(peak,4*spec["element_bytes"]*sum(math.prod(shape[d] for d in axes) for axes in spec["variables"]))
    if work != math.prod(spec["dims"])*(2 if spec["op_type"] == 5 else 1):
        raise ValueError("analytic cell histogram does not conserve work")
    cell_counts = [sum(h.values()) for h in histograms]
    passes = [math.prod(cell_counts[d] for d in range(ndim) if d not in axes)
              for axes in spec["variables"]]
    output_bytes = spec["element_bytes"]*math.prod(spec["dims"][d] for d in spec["variables"][0])
    reduction_cycles = math.ceil((passes[0]-1)*(output_bytes//spec["element_bytes"])/comp.get_peak_flopc()[1])
    return dict(total_core_cycles=total, reduction_core_cycles=reduction_cycles,
                work=work, scratch_bytes_per_core=peak, passes=passes,
                cells=math.prod(cell_counts))


def export_program(specs, comp, hw, *, paradigm, instances=1, stages=2, **unused):
    if instances < 1 or stages < 1 or hw.cores % stages:
        raise ValueError("invalid instance/stage count")
    audit = audit_specs(specs)
    if not audit["valid"]:
        raise ValueError("aggregate source audit failed")
    tensors, jobs = {}, []
    summaries = [summary(s,comp) for s in specs]
    for instance in range(instances):
        ids = {t:f"instance{instance}/tensor{t}" for t in audit["tensor_bytes"]}
        for t,n in audit["tensor_bytes"].items():
            tensors[ids[t]] = dict(bytes=n, readonly=t not in audit["producers"],
                persist=t == specs[-1]["tensors"][0], source_tensor=t)
        for i,(spec,cost) in enumerate(zip(specs,summaries)):
            stage = min(stages-1,i*stages//len(specs))
            n = hw.cores//stages if paradigm == "dataflow" else hw.cores
            cores = list(range(stage*n,(stage+1)*n)) if paradigm == "dataflow" else list(range(n))
            accesses, shifts, reductions = [], [], []
            # Canonical flat streams, same logical bytes/work in every mode.
            # Partitioning changes with physical core group, not input volume.
            for v,t in enumerate(spec["tensors"]):
                size = audit["tensor_bytes"][t]
                elements = size//spec["element_bytes"]
                spatial_copies = math.prod(min(s, math.ceil(d/math.ceil(d/s)))
                    for axis,(d,s) in enumerate(zip(spec["dims"],spec["spatial"])) if axis not in spec["variables"][v])
                # A declared ring-sharing policy, not inferred legacy routing.
                # Missing-axis spatial replicas circulate streamed operands;
                # remaining temporal rereads still go to DRAM.
                sharing = math.gcd(cost["passes"][v],math.gcd(spatial_copies,n)) if v else 1
                for k,core in enumerate(cores):
                    left = elements*k//n*spec["element_bytes"]
                    right = elements*(k+1)//n*spec["element_bytes"]
                    if right > left:
                        accesses.append(dict(tensor=ids[t],kind="write" if v == 0 else "read",
                            core=core,offset=left,bytes=right-left,passes=1 if v == 0 else cost["passes"][v]//sharing))
                        if v and sharing > 1:
                            group_start = k//sharing*sharing
                            dst = cores[group_start+(k+1-group_start)%sharing]
                            shifts.append(dict(src=core,dst=dst,tensor=ids[t],
                                bytes=(right-left)*(sharing-1)*(cost["passes"][v]//sharing)))
                        if v == 0 and spatial_copies > 1 and n > 1:
                            reductions.append(dict(src=core,dst=cores[(k+1)%n],tensor=ids[t],
                                bytes=(right-left)*(min(spatial_copies,n)-1)))
            deps = [f"instance{instance}/op{i-1:04d}"] if i else []
            if i == 0 and instance and paradigm != "dataflow":
                deps.append(f"instance{instance-1}/op{len(specs)-1:04d}")
            jobs.append(dict(id=f"instance{instance}/op{i:04d}",deps=deps,cores=cores,
                stage=stage,microbatch=instance,accesses=accesses,shift=shifts,reduce=reductions,
                compute_cycles=math.ceil(cost["total_core_cycles"]/n),
                reduce_compute_cycles=math.ceil(cost["reduction_core_cycles"]/n),
                scratch_bytes_per_core=cost["scratch_bytes_per_core"],work=cost["work"],
                source_operator=spec["index"], cost_summary=cost,
                logical_signature=dict(tensors=spec["tensors"],dims=spec["dims"],variables=spec["variables"],
                    spatial=spec["spatial"],temporal=spec["temporal"],passes=cost["passes"])))
    return dict(schema_version=1,source="L3 canonical aggregate stream cost graph v1",
        lowering="aggregate_stream_v1",execution_model="aggregate",tensors=tensors,jobs=jobs,
        source_audit=audit,independent_instances=instances,
        semantics="dense per-requester sweeps replace affine ordering; explicit ring sharing for spatial replicas, temporal rereads retained; ideal balanced compute; phase accounting, no packet replay")
