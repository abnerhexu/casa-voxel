"""Bounded-memory, ordered row-run / bulk-link reservation cost model.

Static topological job/phase admission, append-only resource reservations.
Rows within a bank run cannot be interleaved by another request. Repeated
sweeps are counted analytically, not expanded into rows, bursts or packets.
"""
from collections import Counter, defaultdict
from bisect import bisect_right
from functools import lru_cache
import hashlib
import math
from .system_model import Layout, Infeasible, route, validate_graph

KINDS = ("stream", "inter_tensor", "revisit", "other")


class RowIntervals:
    def __init__(self):
        self.spans = []

    def contains(self, row):
        i = bisect_right(self.spans,(row,math.inf))-1
        return i >= 0 and row < self.spans[i][1]

    def add(self, left, right):
        i = bisect_right(self.spans,(left,math.inf))-1
        if i >= 0 and self.spans[i][1] >= right:
            return right-left
        if i < 0 or self.spans[i][1] < left:
            i += 1
        j,overlap,first,last = i,0,left,right
        while j < len(self.spans) and self.spans[j][0] <= right:
            a,b = self.spans[j]
            overlap += max(0,min(last,b)-max(first,a))
            left,right = min(left,a),max(right,b)
            j += 1
        self.spans[i:j] = [(left,right)]
        return overlap


def row_events(seen, previous, previous_tensor, tensor, first, rows, passes):
    """Exact transitions for the declared noninterleaved repeated row run."""
    local = Counter()
    hit, cold = int(previous == first), int(previous < 0)
    first_seen = seen.contains(first)
    if not hit and not cold:
        category = ("revisit" if first_seen else "inter_tensor" if previous_tensor != tensor
                    else "stream" if first == previous+1 else "other")
        local[category] += 1
    old = seen.add(first,first+rows)
    local["revisit"] += old-int(first_seen)
    local["stream"] += rows-1-(old-int(first_seen))
    if rows == 1:
        hit += passes-1
    else:
        local["revisit"] += (passes-1)*rows
    counts = dict(hits=hit,cold=cold,conflicts=sum(local.values()),row_touches=rows*passes)
    assert counts["row_touches"] == counts["hits"]+counts["cold"]+counts["conflicts"]
    return counts,local


def execute(graph, hw, mapping, *, paradigm="compute_shift", chunk_bytes=8192,
            resident=None, counterfactual="none", fixed_bank_order=None,
            trace=False, max_tasks=2_000_000):
    if paradigm not in ("spmd","dataflow","compute_shift") or counterfactual not in ("none","all_conflicts",*KINDS):
        raise ValueError("unknown paradigm/counterfactual")
    if max_tasks < 1 or chunk_bytes < 1:
        raise ValueError("invalid task/chunk bound")
    writers = validate_graph(graph,hw)
    layout = Layout(graph["tensors"],mapping,hw)
    resident = resident or {}
    persistent = [0]*hw.cores
    for tensor,cores in resident.items():
        if tensor not in graph["tensors"] or not cores or len(set(cores)) != len(cores):
            raise ValueError("invalid resident copies")
        for core in cores:
            if type(core) is not int or not 0 <= core < hw.cores:
                raise ValueError("resident core outside mesh")
            persistent[core] += graph["tensors"][tensor]["bytes"]
    if max(persistent) > hw.sram_bytes:
        raise Infeasible("resident copies exceed SRAM")
    ends, busy = defaultdict(int), Counter()
    counts,conflicts,traffic = Counter(),Counter(),Counter()
    channel_bytes,bank_bytes,link_bytes = Counter(),Counter(),Counter()
    open_row,open_tensor = [-1]*hw.banks,[None]*hw.banks
    seen = [RowIntervals() for _ in range(hw.banks)]
    order = hashlib.sha256()
    log,job_start,job_end = [],{},{}
    resident_ready = {}
    peak = persistent[:]
    descriptors = 0
    bank_resources = [f"bank:{b}" for b in range(hw.banks)]
    channel_resources = [f"channel:{c}" for c in range(hw.channels)]

    def reserve(resources, earliest, duration):
        start = max([earliest]+[ends[r] for r in resources])
        finish = start+duration
        for r in resources:
            ends[r] = finish
            busy[r] += duration
        return start,finish

    @lru_cache(maxsize=None)
    def path_resources(src,dst):
        links = route(src,dst,hw)
        return tuple(f"link:{a}:{b}" for a,b in links),tuple(f"{a}:{b}" for a,b in links)

    def network(src,dst,size,earliest,category):
        resources,links = path_resources(src,dst)
        if not resources:
            return earliest
        duration = math.ceil(size/hw.noc_bytepc)
        _,finish = reserve(resources,earliest,duration)
        for link in links:
            link_bytes[link] += size
        traffic["noc_byte_hops"] += size*len(links)
        traffic[category+"_byte_hops"] += size*len(links)
        return finish+len(links)*hw.router_cycles

    def memory(access, earliest, token):
        nonlocal descriptors
        descriptors += 1
        if descriptors > max_tasks:
            raise ValueError("aggregate descriptor bound exceeded")
        passes = access.get("passes",1)
        if type(passes) is not int or passes < 1 or access["kind"] == "write" and passes != 1:
            raise ValueError("invalid aggregate access passes")
        result,total = earliest,0
        # One monotone physical row run per bank. Bank striping changes neither
        # payload bytes nor physical row size. Host chunk size has no physics.
        span = math.ceil((access.get("offset",0)+access["bytes"])/hw.row_bytes)*hw.row_bytes
        for bank,row,rows,size in layout.chunks(access,max(hw.row_bytes,span)):
            total += size
            channel = hw.bank_channels[bank]
            injection = hw.channel_nodes[channel]
            release = earliest
            if access["kind"] == "write":
                release = network(access["core"],injection,size*passes,release,"dram_write")
            event,local = row_events(seen[bank],open_row[bank],open_tensor[bank],access["tensor"],row,rows,passes)
            paid = event["conflicts"] if counterfactual == "none" else (0 if counterfactual == "all_conflicts" else event["conflicts"]-local[counterfactual])
            latency = event["row_touches"]*hw.cl+event["cold"]*hw.rcd+paid*(hw.rp+hw.rcd)
            br,cr = bank_resources[bank],channel_resources[channel]
            start = max(release,ends[br])
            duration = math.ceil(size*passes/hw.channel_bytepc)
            finish = max(start+latency,ends[cr])+duration
            ends[cr] = finish
            busy[cr] += duration
            ends[br] = finish
            busy[br] += finish-start
            open_row[bank],open_tensor[bank] = row+rows-1,access["tensor"]
            order.update(f"{token}|{bank}|{row}|{rows}|{passes};".encode())
            counts.update(event)
            conflicts.update(local)
            traffic[access["kind"]+"_bytes"] += size*passes
            channel_bytes[channel] += size*passes
            bank_bytes[bank] += size*passes
            if access["kind"] == "read":
                finish = network(injection,access["core"],size*passes,finish,"dram_read")
            result = max(result,finish)
            if trace:
                if len(log) >= max_tasks:
                    raise ValueError("aggregate trace bound exceeded")
                log.append(dict(id=token,kind="memory_run",start=start,finish=finish,bank=bank,
                    row=row,rows=rows,passes=passes,bytes=size*passes,conflicts=dict(local)))
        if total != access["bytes"]:
            raise AssertionError("aggregate layout lost bytes")
        return result

    for tensor,cores in sorted(resident.items()):
        if tensor not in writers:
            for core in cores:
                resident_ready[tensor,core] = memory(dict(tensor=tensor,core=core,kind="read",
                    bytes=graph["tensors"][tensor]["bytes"]),0,f"fill/{tensor}/{core}")
    pending = {j["id"]:j for j in graph["jobs"]}
    while pending:
        eligible = [j for j in pending.values() if all(d in job_end for d in j.get("deps",[]))]
        # Fixed topology, independent of counterfactual durations; no claim of
        # free dynamic rescheduling. Aggregate fixed-order contrast is exact
        # for this declared schedule, not the fine backend's scheduler.
        job = min(eligible,key=lambda j:j["id"])
        key = job["id"]
        release = max([job_end[d] for d in job.get("deps",[])]+[0])
        scratch = job.get("scratch_bytes_per_core",0)
        for c in job["cores"]:
            if persistent[c]+scratch > hw.sram_bytes:
                raise Infeasible("aggregate tile scratch plus resident exceeds SRAM")
            peak[c] = max(peak[c],persistent[c]+scratch)
        # Scratch admission is serialized on each core for the whole streaming
        # kernel, while disjoint stage groups may overlap.
        release = max([release]+[ends[f"admit:{c}"] for c in job["cores"]])
        job_start[key] = release
        cursor = release
        for si,step in enumerate(job.get("steps",[job])):
            begin = cursor
            reads = [begin]
            for ai,a in enumerate(step.get("accesses",[])):
                if a["kind"] != "read":
                    continue
                if a["tensor"] in resident:
                    src = min(resident[a["tensor"]],key=lambda c:(len(route(c,a["core"],hw)),c))
                    reads.append(network(src,a["core"],a["bytes"]*a.get("passes",1),
                        max(begin,resident_ready[a["tensor"],src]),"reuse"))
                else:
                    reads.append(memory(a,begin,f"{key}/{si}/r{ai}"))
            cursor = max(reads)
            def flows(phase, at):
                return max([at]+[network(f["src"],f["dst"],f["bytes"],at,phase) for f in step.get(phase,[])])
            cursor = flows("broadcast",cursor)
            shifted = flows("shift",cursor)
            _,computed = reserve([f"core:{c}" for c in step.get("cores",job["cores"])],
                shifted if paradigm == "spmd" else cursor,step["compute_cycles"])
            cursor = flows("reduce",max(shifted,computed))
            _,cursor = reserve([f"core:{c}" for c in step.get("reduce_cores",job["cores"])],cursor,step.get("reduce_compute_cycles",0))
            writes = [cursor]
            for ai,a in enumerate(step.get("accesses",[])):
                if a["kind"] != "write":
                    continue
                if a["tensor"] in resident:
                    for dst in resident[a["tensor"]]:
                        finish = network(a["core"],dst,a["bytes"],cursor,"forward")
                        resident_ready[a["tensor"],dst] = max(finish,resident_ready.get((a["tensor"],dst),0))
                        writes.append(finish)
                if a["tensor"] not in resident or graph["tensors"][a["tensor"]].get("persist",False):
                    writes.append(memory(a,cursor,f"{key}/{si}/w{ai}"))
            cursor = max(writes)
        job_end[key] = cursor
        for c in job["cores"]:
            ends[f"admit:{c}"] = cursor
        del pending[key]
    signature = dict(policy="static_topological_phase_order_v1",sha256=order.hexdigest())
    if fixed_bank_order is not None and fixed_bank_order != signature:
        raise ValueError("aggregate fixed bank order fingerprint mismatch")
    makespan = max(job_end.values())
    size = traffic["read_bytes"]+traffic["write_bytes"]
    energy = dict(dram_transfer=size*hw.dram_pj_byte,row_conflict=counts["conflicts"]*hw.conflict_pj,
        noc_transport=traffic["noc_byte_hops"]*hw.noc_pj_byte_hop,tsv_transfer=size*hw.tsv_pj_byte)
    energy["modeled_movement_total"] = sum(energy.values())
    groups = defaultdict(list)
    for j in graph["jobs"]:
        groups[str(j.get("microbatch",0))].append(job_end[j["id"]])
    completions = sorted(max(v) for v in groups.values())
    return dict(model="l3_aggregate_row_run_v1",paradigm=paradigm,counterfactual=counterfactual,
        chunk_bytes=chunk_bytes,bank_order=signature,bank_allocated_bytes=layout.bank_allocated_bytes,trace=log,
        metrics=dict(cycles=makespan,milliseconds=makespan/(1000*hw.frequency_mhz),row=dict(counts),
            conflict_types={k:conflicts[k] for k in KINDS},traffic=dict(traffic),movement_energy_pj=energy,
            channel_bytes=dict(channel_bytes),bank_bytes=dict(bank_bytes),link_bytes=dict(link_bytes),
            resources={r:dict(busy_cycles=b,utilization=b/max(1,makespan)) for r,b in busy.items()},
            peak_sram_bytes=peak,job_start=job_start,job_end=job_end,microbatch_completions=completions,
            completion_intervals=[b-a for a,b in zip(completions,completions[1:])],aggregate_descriptors=descriptors),
        limitations=["row-run phase accounting, not burst/packet timing; one CL per touched row",
            "static topological admission; fixed-order and reschedule contrasts coincide by construction",
            "noninterleaved repeated sweeps; append-only bulk path reservations, no gap filling",
            "canonical dense stream order may change row conflict mix versus affine tile replay",
            "whole-kernel streaming scratch budget, no fine-grained buffer lifetimes or flow control",
            "no tRAS, refresh, turnaround, compute/static energy or physical vertical distance"])
