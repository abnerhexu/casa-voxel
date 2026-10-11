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


def movement_profile(spec, comp, hw):
    """Return per-core backing-store and remote-SRAM movement for one tiling.

    This is the byte-accounting counterpart of :func:`export_program`.  It
    uses the same canonical dense streams, balanced contiguous core slices,
    ring sharing, and output-reduction policy, but avoids constructing a full
    graph when an experiment only needs movement and capacity information.

    ``vertical_bytes_by_core`` counts required backing-store reads and writes
    assigned to each core.  ``remote_sram_bytes_by_core`` counts payload bytes
    received from another core.  The latter deliberately excludes hop
    weighting so the two arrays have the same unit and may be added.  The
    separate ``noc_byte_hops`` field follows dimension-ordered XY routes.

    A shared-SRAM counterfactual uses the same vertical traffic and pooled
    capacity, while treating the remote flows as internal uniform SRAM
    accesses.  It must not invent a second set of tilings or backing-store
    requests.
    """
    from .system_model import route

    n = int(hw.cores)
    if n <= 0:
        raise ValueError("hardware must contain at least one core")
    cost = summary(spec, comp)
    vertical = [0] * n
    remote = [0] * n
    byte_hops = 0
    flow_count = 0

    def add_flow(src, dst, num_bytes):
        nonlocal byte_hops, flow_count
        num_bytes = int(num_bytes)
        if num_bytes <= 0 or src == dst:
            return
        remote[int(dst)] += num_bytes
        byte_hops += num_bytes * len(route(int(src), int(dst), hw))
        flow_count += 1

    for variable, tensor in enumerate(spec["tensors"]):
        del tensor  # Tensor identity does not change the stream volume.
        size = int(spec["element_bytes"] * math.prod(
            spec["dims"][axis]
            for axis in spec["variables"][variable]
            if axis is not None
        ))
        elements = size // int(spec["element_bytes"])
        spatial_copies = math.prod(
            min(spatial, math.ceil(dim / math.ceil(dim / spatial)))
            for axis, (dim, spatial) in enumerate(
                zip(spec["dims"], spec["spatial"])
            )
            if axis not in spec["variables"][variable]
        )
        sharing = (
            math.gcd(cost["passes"][variable], math.gcd(spatial_copies, n))
            if variable else 1
        )
        for core in range(n):
            left = elements * core // n * int(spec["element_bytes"])
            right = elements * (core + 1) // n * int(spec["element_bytes"])
            segment_bytes = right - left
            if segment_bytes <= 0:
                continue
            if variable == 0:
                # The final value of every output element is materialized once.
                vertical[core] += segment_bytes
                copies = min(spatial_copies, n)
                if copies > 1 and n > 1:
                    add_flow(
                        core,
                        (core + 1) % n,
                        segment_bytes * (copies - 1),
                    )
            else:
                passes = cost["passes"][variable] // sharing
                vertical[core] += segment_bytes * passes
                if sharing > 1:
                    group_start = core // sharing * sharing
                    destination = group_start + (
                        (core + 1 - group_start) % sharing
                    )
                    add_flow(
                        core,
                        destination,
                        segment_bytes * (sharing - 1) * passes,
                    )

    vertical_total = sum(vertical)
    remote_total = sum(remote)
    if vertical_total < 0 or remote_total < 0:
        raise AssertionError("movement counters must be nonnegative")
    return {
        "vertical_bytes_by_core": vertical,
        "remote_sram_bytes_by_core": remote,
        "private_bytes_by_core": [
            vertical[index] + remote[index] for index in range(n)
        ],
        "vertical_bytes": vertical_total,
        "remote_sram_bytes": remote_total,
        "private_bytes": vertical_total + remote_total,
        "noc_byte_hops": int(byte_hops),
        "noc_flow_count": int(flow_count),
        "private_peak_sram_bytes_by_core": [
            int(cost["scratch_bytes_per_core"])
        ] * n,
        "shared_peak_sram_bytes": int(cost["scratch_bytes_per_core"]) * n,
        "cost_summary": cost,
        "semantics": (
            "VOXEL aggregate_stream_v1 canonical dense streams; vertical "
            "reads+writes plus incoming remote-SRAM payload; XY byte-hops "
            "reported separately"
        ),
    }


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
