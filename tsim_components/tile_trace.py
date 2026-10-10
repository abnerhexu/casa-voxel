"""Canonical affine tile lowering for L3, sourced from compiled expressions.

The old cost compiler does not contain a physical instruction schedule. This
module SPECIFIES one: clipped rectangular iteration cells, row-major tensor
views, deduplicated input broadcasts, one-step operand prefetch/shift and
owner-based output reductions. It does not reverse-engineer aggregate bytes.
Convolutions/gathers/non-affine views are rejected rather than guessed.
"""
from collections import Counter, defaultdict
from functools import lru_cache
from itertools import product
import math


class TraceError(ValueError):
    pass


def expression_spec(op, config, index):
    expr = op.expr
    if int(expr.op_type) not in (0, 1, 2, 5):
        raise TraceError(f"operator {index}: non-affine/unsupported op_type={expr.op_type}")
    dims = [int(n) for n in expr.dim_lengths[:-1]]
    if any(d <= 0 for d in dims):
        raise TraceError(f"operator {index}: invalid iteration dimensions")
    spatial, temporal = config
    spatial = [int(v) for v in spatial]
    temporal = [[int(v) for v in row] for row in temporal]
    if len(spatial) != len(dims) or len(temporal) != len(dims):
        raise TraceError("tiling dimensionality mismatch")
    frozen = (tuple(spatial), tuple(tuple(row) for row in temporal))
    if frozen not in expr.config_dict:
        raise TraceError(f"operator {index}: requested tiling was not compiled")
    variables = []
    for variable in expr.variables:
        axes = []
        for axis in variable:
            active = {int(d) for d in axis if 0 <= int(d) < len(dims)}
            if len(active) > 1:
                raise TraceError(f"operator {index}: affine-sum axis needs a dedicated lowering")
            axes.append(next(iter(active)) if active else None)
        if len([d for d in axes if d is not None]) != len(set(d for d in axes if d is not None)):
            raise TraceError("diagonal/repeated tensor axes need a dedicated lowering")
        variables.append(axes)
    ids = [str(op.output_idx)] + [str(v) for v in op.input_idx_list]
    if op.output_idx is None or len(ids) != len(variables) or any(v == "None" for v in ids):
        raise TraceError(f"operator {index}: missing source tensor identity")
    nvars = len(variables)
    if any(len(row) != nvars or min(row) < 1 for row in temporal) or min(spatial) < 1:
        raise TraceError("invalid temporal/spatial factors")
    return dict(index=index, name=op.name, op_type=int(expr.op_type), dims=dims,
                variables=variables, tensors=ids, spatial=spatial, temporal=temporal,
                element_bytes=int(expr.num_byte_per_elem),
                source_hot_bytes=int(expr.config_dict[frozen][0]))


def tensor_size(spec, variable):
    return math.prod(spec["dims"][d] if d is not None else 1 for d in spec["variables"][variable])*spec["element_bytes"]


def audit_specs(specs):
    """Flat reshape views are valid; changing the extent of a produced tensor is not."""
    sizes, producers, errors = {}, {}, []
    for spec in specs:
        for v, tensor in enumerate(spec["tensors"]):
            size = tensor_size(spec, v)
            if tensor in sizes and sizes[tensor] != size:
                errors.append(dict(operator=spec["index"], tensor=tensor, previous_bytes=sizes[tensor], bytes=size))
            sizes.setdefault(tensor, size)
        output = spec["tensors"][0]
        if output in producers:
            errors.append(dict(operator=spec["index"], tensor=output, reason="multiple writers need SSA versioning"))
        producers[output] = spec["index"]
    for spec in specs:
        for tensor in spec["tensors"][1:]:
            if tensor in producers and producers[tensor] >= spec["index"]:
                errors.append(dict(operator=spec["index"], tensor=tensor,
                                   reason="input producer must precede consumer; use SSA versions"))
    return dict(tensor_bytes=sizes, producers=producers, errors=errors,
                valid=not errors, operators=len(specs))


def segments(spec, variable, bounds):
    """Exact contiguous byte intervals of a rectangular row-major tensor tile."""
    axes = spec["variables"][variable]
    shape = [spec["dims"][d] if d is not None else 1 for d in axes]
    ranges = [bounds[d] if d is not None else (0, 1) for d in axes]
    if not shape:
        return ((0, spec["element_bytes"]),)
    strides = [math.prod(shape[i+1:]) for i in range(len(shape))]
    # Collapse whole trailing axes, avoiding one Python object per tensor row.
    last = len(shape)-1
    while last > 0 and ranges[last] == (0, shape[last]):
        last -= 1
    result = []
    for prefix in product(*(range(a, b) for a, b in ranges[:last])):
        offset = (sum(c*s for c, s in zip(prefix, strides)) + ranges[last][0]*strides[last])*spec["element_bytes"]
        size = (ranges[last][1]-ranges[last][0])*strides[last]*spec["element_bytes"]
        if result and result[-1][0]+result[-1][1] == offset:
            result[-1] = (result[-1][0], result[-1][1]+size)
        else:
            result.append((offset, size))
    return tuple(result)


def cells(spec):
    dims, spatial = spec["dims"], spec["spatial"]
    temporal = [max(row) for row in spec["temporal"]]
    spatial_size = [math.ceil(d/s) for d, s in zip(dims, spatial)]
    tile_size = [math.ceil(d/t) for d, t in zip(spatial_size, temporal)]
    for ti in product(*(range(t) for t in temporal)):
        for logical_core, si in enumerate(product(*(range(s) for s in spatial))):
            bounds = tuple((s*n+t*b, min(d, (s+1)*n, s*n+(t+1)*b))
                           for d, s, t, n, b in zip(dims, si, ti, spatial_size, tile_size))
            if all(a < b for a, b in bounds):
                yield logical_core, bounds


def lower_operator(spec, comp, cores, *, max_cells=200000, max_segments=2000000, wave_core_stride=1):
    """Lower cells to bounded parallel waves; accumulated outputs stay in SRAM.

    Only previous-wave operands are retained. A shift in wave k supplies an
    input to wave k+1, so overlap with wave k compute is dependency-correct.
    Output accumulators are live from their first partial to their final write.
    """
    import numpy as np
    waves, wave, occupied, work, logical_cells = [], [], set(), 0, []
    nvars, ndim = len(spec["variables"]), len(spec["dims"])
    original_vars = [np.array([[d if d is not None else ndim] for d in axes]) for axes in spec["variables"]]
    @lru_cache(maxsize=None)
    def cost(shape):
        op = comp.convert_op_simple(dim_len=np.array(shape), variables=original_vars,
              is_ew=spec["op_type"] != 5, ignore_variables=[False]*nvars,
              tensor_id_list=list(range(nvars)), op_type=spec["op_type"])
        return math.ceil(comp.get_total_cycle_for_fused_op([op], [[[1]*nvars for _ in range(ndim)]])[0])
    segment_count = 0
    for count, (logical, bounds) in enumerate(cells(spec), 1):
        if count > max_cells:
            raise TraceError("cell bound exceeded; use --operator-limit or increase the explicit bound")
        core = cores[(logical+len(waves)*wave_core_stride) % len(cores)]
        if core in occupied:
            waves.append(wave)
            wave, occupied = [], set()
            core = cores[(logical+len(waves)*wave_core_stride) % len(cores)]
        parts = [segments(spec, v, bounds) for v in range(nvars)]
        segment_count += sum(len(p) for p in parts)
        if segment_count > max_segments:
            raise TraceError("segment bound exceeded")
        shape = tuple(b-a for a, b in bounds)
        work += math.prod(shape)*(2 if spec["op_type"] == 5 else 1)
        logical_cells.append([list(v) for v in bounds])
        wave.append(dict(core=core, parts=parts, cycles=cost(shape)))
        occupied.add(core)
    if wave:
        waves.append(wave)
    if work != math.prod(spec["dims"])*(2 if spec["op_type"] == 5 else 1):
        raise TraceError("iteration cells do not conserve arithmetic work")
    # Deterministic owner per output tile. All partial contributors are counted
    # before placing the final write, preventing premature accumulator stores.
    owners, remaining = {}, Counter()
    for wave in waves:
        for cell in wave:
            out = cell["parts"][0]
            if out not in owners:
                owners[out] = cell["core"]
            remaining[out] += 1
    steps, demands = [], []
    for wave in waves:
        need = defaultdict(set)
        for cell in wave:
            for v in range(1, nvars):
                need[spec["tensors"][v], cell["parts"][v]].add(cell["core"])
        demands.append({key: (min(users), sorted(users)) for key, users in need.items()})
    seen, scratch, live_outputs = Counter(), 0, set()
    for i, wave in enumerate(waves):
        step = dict(cores=sorted({cell["core"] for cell in wave}), compute_cycles=max(c["cycles"] for c in wave),
                    accesses=[], broadcast=[], shift=[], reduce=[], carry_bytes={}, reduce_compute_cycles=0)
        # Reserve accumulator + current/next input tiles (double buffer) and
        # one local partial output. Deliberate upper bound, not a free cache.
        live_outputs.update(cell["parts"][0] for cell in wave)
        storage = Counter()
        for parts in live_outputs:
            storage[owners[parts]] += sum(n for _, n in parts)
        for demand in (demands[i], demands[i+1] if i+1 < len(waves) else {}):
            for (tensor, parts), (owner, users) in demand.items():
                for core in users:
                    storage[core] += sum(n for _, n in parts)
        for cell in wave:
            storage[cell["core"]] += sum(n for _, n in cell["parts"][0])
        for (tensor, parts), (owner, users) in demands[i].items():
            if i == 0 or (tensor, parts) not in demands[i-1]:
                step["accesses"] += [dict(tensor=tensor, kind="read", core=owner, offset=o, bytes=n) for o, n in parts]
            for dst in users:
                if dst != owner:
                    step["broadcast"].append(dict(src=owner, dst=dst, bytes=sum(n for _, n in parts),
                                                  tensor=tensor, segments=parts))
            if i+1 < len(waves) and (tensor, parts) in demands[i+1]:
                dst = demands[i+1][tensor, parts][0]
                if dst != owner:
                    step["shift"].append(dict(src=owner, dst=dst, bytes=sum(n for _, n in parts),
                                              tensor=tensor, segments=parts, consumer_step=i+1))
        reduction_ops = Counter()
        for cell in wave:
            parts = cell["parts"][0]
            owner, size = owners[parts], sum(n for _, n in parts)
            if cell["core"] != owner:
                step["reduce"].append(dict(src=cell["core"], dst=owner, bytes=size,
                                           tensor=spec["tensors"][0], segments=parts))
            if seen[parts]:
                reduction_ops[owner] += size//spec["element_bytes"]
            seen[parts] += 1
            remaining[parts] -= 1
            if remaining[parts] == 0:
                step["accesses"] += [dict(tensor=spec["tensors"][0], kind="write", core=owner, offset=o, bytes=n)
                                     for o, n in parts]
        step["reduce_compute_cycles"] = math.ceil(max(reduction_ops.values(), default=0)/comp.get_peak_flopc()[1])
        step["reduce_cores"] = sorted(reduction_ops) or step["cores"]
        # carry is intentionally an overestimate in addition to accessed tiles;
        # it makes all retained/receiving buffers visible to the SRAM checker.
        step["carry_bytes"] = dict(storage)
        accessed = Counter()
        for a in step["accesses"]:
            accessed[a["core"]] += a["bytes"]
        scratch = max(scratch, max((storage[c]+accessed[c] for c in cores), default=0))
        steps.append(step)
        live_outputs = {parts for parts in live_outputs if remaining[parts] > 0}
    return dict(steps=steps, accesses=[a for s in steps for a in s["accesses"]], cores=list(cores),
                scratch_bytes_per_core=scratch, compute_cycles=sum(s["compute_cycles"] for s in steps),
                work=work, logical_cells=logical_cells,
                lowering="affine_tile_v1", wave_core_stride=wave_core_stride, source_operator=spec["index"],
                compute_cost_source="voxel Compute.convert_op_simple/get_total_cycle_for_fused_op per clipped cell",
                iteration_shape=spec["dims"], source_name=spec["name"])


def export_program(specs, comp, hw, *, paradigm, instances=1, stages=2, max_cells=200000, max_segments=2000000,
                   wave_core_stride=1):
    if not specs or paradigm not in ("spmd", "dataflow", "compute_shift"):
        raise TraceError("nonempty source and supported paradigm required")
    if type(wave_core_stride) is not int or wave_core_stride < 0:
        raise TraceError("wave core stride must be a nonnegative integer")
    audit = audit_specs(specs)
    if not audit["valid"]:
        raise TraceError(f"source tensor extent/version errors: {audit['errors'][:5]}")
    if instances < 1 or stages < 1 or stages > hw.cores or hw.cores % stages:
        raise TraceError("invalid independent-instance or stage count")
    if any(math.prod(s["spatial"]) > hw.cores for s in specs):
        raise TraceError("compiled spatial tiling exceeds hardware core count")
    templates = []
    for i, spec in enumerate(specs):
        stage = min(stages-1, i*stages//len(specs))
        cores = list(range(stage*(hw.cores//stages), (stage+1)*(hw.cores//stages))) if paradigm == "dataflow" else list(range(hw.cores))
        templates.append(lower_operator(spec, comp, cores, max_cells=max_cells, max_segments=max_segments,
                                        wave_core_stride=wave_core_stride))
    import copy
    tensors, jobs = {}, []
    for instance in range(instances):
        # Replicas here are full independent invocations; never split sequence
        # length or recurrent-state dependencies without an explicit lowering.
        ids = {t: f"instance{instance}/tensor{t}" for t in audit["tensor_bytes"]}
        for t, n in audit["tensor_bytes"].items():
            tensors[ids[t]] = dict(bytes=n, readonly=t not in audit["producers"],
                                    persist=t == specs[-1]["tensors"][0], source_tensor=t)
        for i, template in enumerate(templates):
            j = copy.deepcopy(template)
            j.update(id=f"instance{instance}/op{i:04d}", microbatch=instance,
                     stage=min(stages-1, i*stages//len(specs)))
            # Conservative source operator order preserves all explicit graph
            # dependencies, including side-effect-like state update ordering.
            j["deps"] = [f"instance{instance}/op{i-1:04d}"] if i else []
            if i == 0 and instance and paradigm != "dataflow":
                j["deps"].append(f"instance{instance-1}/op{len(specs)-1:04d}")
            for step in j["steps"]:
                for a in step["accesses"]:
                    a["tensor"] = ids[a["tensor"]]
                for phase in ("broadcast", "shift", "reduce"):
                    for flow in step[phase]:
                        flow["tensor"] = ids[flow["tensor"]]
            j["accesses"] = [a for s in j["steps"] for a in s["accesses"]]
            jobs.append(j)
    return dict(schema_version=1, source="voxel compiler-sourced canonical affine lowering v1",
                lowering="affine_tile_v1", tensors=tensors, jobs=jobs,
                source_audit=audit, independent_instances=instances,
                semantics="canonical schedule, not reconstruction of legacy aggregate traffic; full invocations, not sequence splitting")
