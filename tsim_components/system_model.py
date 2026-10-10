"""L3 absolute-time resource model.

This is a deterministic, non-preemptive reservation model, not a DRAM command
or packet simulator. Banks retain open rows across jobs; independent banks may
prepare rows while channel data buses are busy. NoC chunks use store-and-forward
XY routing. The input must provide explicit addresses, requesters, dependencies,
core assignments and compute costs; no execution paradigm is inferred from sizes.
"""
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
import heapq
import math


class Infeasible(ValueError):
    """A well-formed design exceeds a physical capacity."""


@dataclass(frozen=True)
class Hardware:
    cores: int = 256
    mesh_width: int = 16
    banks: int = 256
    channels: int = 80
    row_bytes: int = 8192
    bank_capacity: int = 192 * 1024**3 // 256
    sram_bytes: int = 2 * 1024**2
    channel_bytepc: float = 16.0
    noc_bytepc: float = 16.0
    cl: int = 14
    rcd: int = 14
    rp: int = 14
    router_cycles: int = 1
    dram_burst_bytes: int = 128
    noc_packet_bytes: int = 128
    frequency_mhz: int = 1600
    dram_pj_byte: float = 7.0
    conflict_pj: float = 7270.0
    noc_pj_byte_hop: float = 12.0
    tsv_pj_byte: float = 0.0
    channel_nodes: tuple = ()
    bank_channels: tuple = ()

    def __post_init__(self):
        for key in ("cores", "mesh_width", "banks", "channels", "row_bytes",
                    "bank_capacity", "sram_bytes", "frequency_mhz", "dram_burst_bytes", "noc_packet_bytes"):
            if type(getattr(self, key)) is not int or getattr(self, key) <= 0:
                raise ValueError(f"{key} must be a positive integer")
        if self.cores % self.mesh_width or self.channels > self.cores:
            raise ValueError("rectangular mesh and channels <= cores required")
        if self.dram_burst_bytes > self.row_bytes or self.row_bytes % self.dram_burst_bytes:
            raise ValueError("DRAM burst must divide the physical row")
        for key in ("channel_bytepc", "noc_bytepc"):
            if not math.isfinite(getattr(self, key)) or getattr(self, key) <= 0:
                raise ValueError(f"invalid {key}")
        for key in ("cl", "rcd", "rp", "router_cycles"):
            if type(getattr(self, key)) is not int or getattr(self, key) < 0:
                raise ValueError(f"invalid {key}")
        for key in ("dram_pj_byte", "conflict_pj", "noc_pj_byte_hop", "tsv_pj_byte"):
            if not math.isfinite(getattr(self, key)) or getattr(self, key) < 0:
                raise ValueError(f"invalid {key}")
        if not self.channel_nodes:
            q, r = divmod(self.cores, self.channels)
            groups = [q + (c < r) for c in range(self.channels)]
            starts = [sum(groups[:c]) for c in range(self.channels)]
            object.__setattr__(self, "channel_nodes", tuple(
                s + (n - 1)//2 for s, n in zip(starts, groups)))
        if not self.bank_channels:
            object.__setattr__(self, "bank_channels", tuple(b % self.channels for b in range(self.banks)))
        if (len(self.channel_nodes) != self.channels or
                any(type(c) is not int or not 0 <= c < self.cores for c in self.channel_nodes) or
                len(self.bank_channels) != self.banks or
                any(type(c) is not int or not 0 <= c < self.channels for c in self.bank_channels)):
            raise ValueError("invalid physical channel mapping")


class Calendar:
    """Nonoverlapping half-open intervals; insertion can fill reserved gaps."""
    def __init__(self):
        self.intervals = []

    def available(self, start, duration):
        if duration == 0 or not self.intervals or start >= self.intervals[-1][1]:
            return start
        for left, right, _ in self.intervals:
            if start + duration <= left:
                break
            if start < right:
                start = right
        return start

    def reserve(self, start, duration, owner):
        if duration:
            if self.available(start, duration) != start:
                raise AssertionError("overlapping resource reservation")
            import bisect
            bisect.insort(self.intervals, (start, start + duration, owner))


def route(src, dst, hw):
    path = []
    while src % hw.mesh_width != dst % hw.mesh_width:
        nxt = src + (1 if src % hw.mesh_width < dst % hw.mesh_width else -1)
        path.append((src, nxt))
        src = nxt
    while src != dst:
        nxt = src + (hw.mesh_width if src < dst else -hw.mesh_width)
        path.append((src, nxt))
        src = nxt
    return path


def validate_graph(graph, hw):
    if graph.get("schema_version") != 1 or not graph.get("source"):
        raise ValueError("graph needs schema_version=1 and source provenance")
    tensors = graph["tensors"]
    for name, tensor in tensors.items():
        if not isinstance(name, str) or type(tensor["bytes"]) is not int or tensor["bytes"] <= 0:
            raise ValueError("tensor sizes must be positive integer bytes")
    jobs = {j["id"]: j for j in graph["jobs"]}
    if not jobs or len(jobs) != len(graph["jobs"]):
        raise ValueError("empty graph or duplicate job IDs")
    ancestors, visiting = {}, set()
    def visit(key):
        if key in ancestors:
            return ancestors[key]
        if key in visiting or key not in jobs:
            raise ValueError("cyclic or missing job dependency")
        visiting.add(key)
        result = set()
        for dep in jobs[key].get("deps", []):
            result |= visit(dep) | {dep}
        visiting.remove(key)
        ancestors[key] = result
        return result
    writers = {}
    for key, j in jobs.items():
        visit(key)
        cores = j["cores"]
        if (not cores or len(set(cores)) != len(cores) or
                any(type(c) is not int or not 0 <= c < hw.cores for c in cores)):
            raise ValueError("invalid physical core group")
        if any(type(j.get(k, 0)) is not int or j.get(k, 0) < 0
               for k in ("compute_cycles", "scratch_bytes_per_core", "work")):
            raise ValueError("cost, work and scratch size must be nonnegative integers")
        steps = j.get("steps", [j])
        if not steps or any(type(s.get("compute_cycles")) is not int or s["compute_cycles"] < 0 for s in steps):
            raise ValueError("invalid step compute cost")
        if "steps" in j and j.get("accesses", []) != [a for s in steps for a in s.get("accesses", [])]:
            raise ValueError("step accesses do not match job access catalog")
        for s in steps:
            if not s.get("cores", cores) or not set(s.get("cores", cores)) <= set(cores):
                raise ValueError("step cores outside job core group")
            if (type(s.get("reduce_compute_cycles", 0)) is not int or s.get("reduce_compute_cycles", 0) < 0
                    or not set(s.get("reduce_cores", cores)) <= set(cores)):
                raise ValueError("invalid reduction cost or core group")
            for c, n in s.get("carry_bytes", {}).items():
                if str(c) not in {str(core) for core in cores} or type(n) is not int or n < 0:
                    raise ValueError("invalid retained buffer allocation")
            for flow in [f for phase in ("broadcast", "shift", "reduce") for f in s.get(phase, [])]:
                if (any(type(flow[k]) is not int or not 0 <= flow[k] < hw.cores for k in ("src", "dst"))
                        or type(flow["bytes"]) is not int or flow["bytes"] <= 0):
                    raise ValueError("invalid operator flow")
        for access in j.get("accesses", []):
            tensor = tensors[access["tensor"]]
            if (access["kind"] not in ("read", "write") or
                    type(access["core"]) is not int or access["core"] not in cores or
                    type(access.get("offset", 0)) is not int or access.get("offset", 0) < 0 or
                    type(access["bytes"]) is not int or access["bytes"] <= 0 or
                    access.get("offset", 0) + access["bytes"] > tensor["bytes"]):
                raise ValueError("invalid tile access or range outside allocation")
            if access["kind"] == "write":
                previous = writers.setdefault(access["tensor"], key)
                if previous != key:
                    raise ValueError("use a separate versioned tensor ID for each writer")
                if tensor.get("readonly", False):
                    raise ValueError("write to readonly tensor")
    for key, j in jobs.items():
        for a in j.get("accesses", []):
            if a["kind"] == "read" and a["tensor"] in writers:
                writer = writers[a["tensor"]]
                if writer not in ancestors[key]:
                    raise ValueError("read lacks dependency on its tensor producer")
    # Written tensors must have exact, nonoverlapping full coverage. Partial
    # updates need explicit versions and are intentionally rejected here.
    for tensor, writer in writers.items():
        ranges = sorted((a.get("offset", 0), a.get("offset", 0)+a["bytes"])
                        for a in jobs[writer].get("accesses", [])
                        if a["kind"] == "write" and a["tensor"] == tensor)
        cursor = 0
        for left, right in ranges:
            if left != cursor:
                raise ValueError("producer writes must cover tensor exactly once")
            cursor = right
        if cursor != tensors[tensor]["bytes"]:
            raise ValueError("incomplete producer output")
    return writers


class Layout:
    """Row-striped tensor storage with stable per-bank physical row extents."""
    def __init__(self, tensors, mapping, hw):
        if set(mapping) != set(tensors):
            raise ValueError("placement must cover exactly the graph tensors")
        self.hw, self.mapping, self.base = hw, {}, {}
        cursor = [0] * hw.banks
        for tensor in sorted(tensors):
            banks = tuple(mapping[tensor])
            if (not banks or len(set(banks)) != len(banks) or
                    any(type(b) is not int or not 0 <= b < hw.banks for b in banks)):
                raise ValueError("invalid tensor bank span")
            self.mapping[tensor] = banks
            rows = math.ceil(tensors[tensor]["bytes"] / hw.row_bytes)
            for ordinal, bank in enumerate(banks):
                count = max(0, (rows + len(banks)-1-ordinal)//len(banks))
                self.base[tensor, bank] = cursor[bank]
                cursor[bank] += count
                if cursor[bank]*hw.row_bytes > hw.bank_capacity:
                    raise Infeasible(f"bank {bank} capacity exceeded")
        self.bank_allocated_bytes = [c * hw.row_bytes for c in cursor]

    def chunks(self, access, chunk_bytes):
        # Sub-row chunks retain the same physical row. Larger chunks coalesce
        # consecutive rows in one bank, never reinterpret access granularity.
        row_bytes = self.hw.row_bytes
        if chunk_bytes < row_bytes:
            offset, left = access.get("offset", 0), access["bytes"]
            while left:
                n = min(left, chunk_bytes-offset % chunk_bytes, row_bytes-offset % row_bytes)
                logical_row = offset // row_bytes
                banks = self.mapping[access["tensor"]]
                bank = banks[logical_row % len(banks)]
                yield bank, self.base[access["tensor"], bank]+logical_row//len(banks), 1, n
                offset += n
                left -= n
            return
        start, end = access.get("offset", 0), access.get("offset", 0)+access["bytes"]
        first, last = start//row_bytes, (end-1)//row_bytes
        banks = self.mapping[access["tensor"]]
        rows_per_chunk = chunk_bytes//row_bytes
        for ordinal, bank in enumerate(banks):
            logical = first + (ordinal-first) % len(banks)
            while logical <= last:
                count = min(rows_per_chunk, (last-logical)//len(banks)+1)
                final = logical+(count-1)*len(banks)
                size = count*row_bytes
                if logical == first:
                    size -= start % row_bytes
                if final == last:
                    size -= (last+1)*row_bytes-end
                yield bank, self.base[access["tensor"], bank]+logical//len(banks), count, size
                logical += count*len(banks)


def execute(graph, hw, mapping, *, paradigm="compute_shift", chunk_bytes=8192,
            resident=None, counterfactual="none", fixed_bank_order=None,
            trace=False, max_tasks=2_000_000):
    """Execute an explicit graph, returning system metrics and optional trace.

    Resident copies reserve their full sizes for the complete run. Initial
    fills, producer forwarding and mandatory persistent writes are charged.
    This conservative residency contract avoids free migration or eviction.
    SPMD serializes shift before compute; other modes overlap independent
    compute/shift tasks. Dataflow concurrency comes from explicit job/core DAGs.
    """
    if paradigm not in ("spmd", "dataflow", "compute_shift"):
        raise ValueError("unknown paradigm")
    kinds = ("stream", "inter_tensor", "revisit", "other")
    if counterfactual not in ("none", "all_conflicts", *kinds):
        raise ValueError("unknown conflict counterfactual")
    if type(chunk_bytes) is not int or chunk_bytes <= 0 or max_tasks <= 0:
        raise ValueError("invalid chunk/task bound")
    writers = validate_graph(graph, hw)
    layout = Layout(graph["tensors"], mapping, hw)
    resident = resident or {}
    persistent = [0]*hw.cores
    for tensor, cores in resident.items():
        if tensor not in graph["tensors"] or not cores or len(cores) != len(set(cores)):
            raise ValueError("invalid resident tensor/copies")
        for c in cores:
            if type(c) is not int or not 0 <= c < hw.cores:
                raise ValueError("resident core outside mesh")
            persistent[c] += graph["tensors"][tensor]["bytes"]
    if max(persistent, default=0) > hw.sram_bytes:
        raise Infeasible("persistent SRAM copies exceed per-core capacity")
    for job in graph["jobs"]:
        # Reads join before compute and writes drain afterwards. Consequently
        # all these tiles must fit the declared scratch allocation; a small
        # scratch value cannot silently imply an unmodelled streaming kernel.
        for step in job.get("steps", [job]):
            ranges = defaultdict(list)
            for a in step.get("accesses", []):
                if a["core"] in resident.get(a["tensor"], []):
                    continue
                ranges[a["core"], a["tensor"]].append((a.get("offset", 0), a.get("offset", 0)+a["bytes"]))
            required = Counter({int(c): n for c, n in step.get("carry_bytes", {}).items()})
            if any(type(n) is not int or n < 0 for n in required.values()):
                raise ValueError("invalid carry buffer declaration")
            for (core, tensor), spans in ranges.items():
                end = -1
                for left, right in sorted(spans):
                    required[core] += max(0, right-max(left, end))
                    end = max(end, right)
            if any(n > job.get("scratch_bytes_per_core", 0) for n in required.values()):
                raise Infeasible(f"job {job['id']} declared scratch cannot hold its live nonresident tiles")
    tasks = {}
    def add(key, kind="join", deps=(), **metadata):
        if key in tasks:
            raise ValueError(f"duplicate task {key}")
        if len(tasks) >= max_tasks:
            raise ValueError("task bound exceeded; use a smaller graph or increase max_tasks; batching does not change physics")
        tasks[key] = dict(kind=kind, deps=set(deps), **metadata)
        return key

    def network(prefix, src, dst, size, deps, job, category):
        if src == dst:
            return add(prefix, deps=deps)
        chunks = [add(f"{prefix}/c{index:08d}", "network", deps,
                      src=src, dst=dst, bytes=min(hw.noc_packet_bytes, size-offset), job=job, category=category)
                  for index, offset in enumerate(range(0, size, hw.noc_packet_bytes))]
        return add(prefix, deps=chunks)

    def memory(prefix, access, deps, job):
        completions, previous = [], {}
        total = 0
        def batched_bursts():
            from itertools import islice
            stream = iter(layout.chunks(access, hw.dram_burst_bytes))
            while batch := list(islice(stream, max(1, chunk_bytes//hw.dram_burst_bytes))):
                yield from batch
        for index, (bank, row, count, size) in enumerate(batched_bursts()):
            total += size
            channel = hw.bank_channels[bank]
            injection = hw.channel_nodes[channel]
            chunk_deps = list(deps)
            if bank in previous:
                chunk_deps.append(previous[bank])
            mem_id = f"{prefix}/m{index:08d}"
            if access["kind"] == "write":
                chunk_deps = [network(f"{prefix}/n{index}", access["core"], injection,
                                      size, chunk_deps, job, "dram_write")]
            add(mem_id, "memory", chunk_deps, bank=bank, channel=channel, row=row,
                rows=count, bytes=size, tensor=access["tensor"], direction=access["kind"], job=job)
            previous[bank] = mem_id
            tail = mem_id
            if access["kind"] == "read":
                tail = network(f"{prefix}/n{index}", injection, access["core"], size,
                               [mem_id], job, "dram_read")
            completions.append(tail)
        if total != access["bytes"]:
            raise AssertionError("physical mapping does not conserve bytes")
        return add(f"{prefix}/done", deps=completions)

    # Resident initial fills are real traffic, included in makespan/energy.
    for tensor, cores in sorted(resident.items()):
        if tensor not in writers:
            for c in cores:
                memory(f"fill/{tensor}/{c}", dict(tensor=tensor, kind="read", core=c,
                       bytes=graph["tensors"][tensor]["bytes"]), [], "initial_fill")

    for j in graph["jobs"]:
        key = j["id"]
        start = add(f"{key}/start", "admit", [f"{d}/done" for d in j.get("deps", [])],
                    job=key, cores=j["cores"], scratch=j.get("scratch_bytes_per_core", 0))
        step_done = start
        resident_tails = defaultdict(list)
        for step_index, step in enumerate(j.get("steps", [j])):
            prefix = f"{key}/step{step_index:08d}"
            reads = []
            for i, a in enumerate(step.get("accesses", [])):
                if a["kind"] != "read":
                    continue
                if a["tensor"] in resident:
                    source = min(resident[a["tensor"]], key=lambda c: (len(route(c, a["core"], hw)), c))
                    producer = (f"resident/{a['tensor']}/{source}" if a["tensor"] in writers
                                else f"fill/{a['tensor']}/{source}/done")
                    reads.append(network(f"{prefix}/r{i}", source, a["core"], a["bytes"],
                                         [step_done, producer], key, "reuse"))
                else:
                    reads.append(memory(f"{prefix}/r{i}", a, [step_done], key))
            read_done = add(f"{prefix}/read_done", deps=reads or [step_done])
            def flows(phase, deps):
                tails = [network(f"{prefix}/{phase}{i}", f["src"], f["dst"], f["bytes"], deps, key, phase)
                         for i, f in enumerate(step.get(phase, []))]
                return add(f"{prefix}/{phase}_done", deps=tails or deps)
            broadcast = flows("broadcast", [read_done])
            shift = flows("shift", [broadcast])
            compute = add(f"{prefix}/compute", "compute", [shift if paradigm == "spmd" else broadcast],
                          cores=step.get("cores", j["cores"]), duration=step["compute_cycles"], job=key)
            reduce = flows("reduce", [compute, shift])
            if step.get("reduce_compute_cycles", 0):
                reduce = add(f"{prefix}/reduce_compute", "compute", [reduce],
                             cores=step.get("reduce_cores", step.get("cores", j["cores"])),
                             duration=step["reduce_compute_cycles"], job=key)
            writes = []
            for i, a in enumerate(step.get("accesses", [])):
                if a["kind"] != "write":
                    continue
                tensor = a["tensor"]
                if tensor in resident:
                    for dst in resident[tensor]:
                        tail = network(f"{prefix}/forward{i}/{dst}", a["core"], dst, a["bytes"],
                                       [reduce], key, "forward")
                        resident_tails[tensor, dst].append(tail)
                        writes.append(tail)
                if tensor not in resident or graph["tensors"][tensor].get("persist", False):
                    writes.append(memory(f"{prefix}/w{i}", a, [reduce], key))
            step_done = add(f"{prefix}/done", deps=writes or [reduce])
        for (tensor, dst), deps in resident_tails.items():
            add(f"resident/{tensor}/{dst}", deps=deps)
        add(f"{key}/done", "release", [step_done], cores=j["cores"],
            scratch=j.get("scratch_bytes_per_core", 0), job=key)

    if fixed_bank_order is not None:
        expected = {k for k, t in tasks.items() if t["kind"] == "memory"}
        flat = [k for seq in fixed_bank_order.values() for k in seq]
        if len(flat) != len(set(flat)) or set(flat) != expected:
            raise ValueError("fixed bank order does not cover exactly the memory tasks")
        for bank, sequence in fixed_bank_order.items():
            for key in sequence:
                if tasks[key]["bank"] != int(bank):
                    raise ValueError("fixed order bank mismatch")
            for left, right in zip(sequence, sequence[1:]):
                tasks[right]["deps"].add(left)
    children, indegree = defaultdict(list), {}
    for key, t in tasks.items():
        indegree[key] = len(t["deps"])
        for dep in t["deps"]:
            if dep not in tasks:
                raise ValueError(f"missing task dependency {dep}")
            children[dep].append(key)
    # Validate expanded DAG (including resident dependencies) before execution.
    degrees = dict(indegree)
    queue = [k for k, degree in degrees.items() if not degree]
    visited = 0
    while queue:
        visited += 1
        for child in children[queue.pop()]:
            degrees[child] -= 1
            if not degrees[child]:
                queue.append(child)
    if visited != len(tasks):
        raise ValueError("expanded task graph has a dependency cycle")

    calendars = defaultdict(Calendar)
    bank_end, open_row, open_tensor = [0]*hw.banks, [-1]*hw.banks, [None]*hw.banks
    seen_rows = defaultdict(set)
    bank_order = defaultdict(list)
    scratch_used = [0]*hw.cores
    peak_sram = list(persistent)
    ready = {k: 0 for k, n in indegree.items() if not n}
    events, task_end, task_log = [], {}, []
    counts, conflicts, traffic = Counter(), Counter(), Counter()
    channel_bytes, bank_bytes, link_bytes = Counter(), Counter(), Counter()
    job_start, job_end = {}, {}
    now = 0

    def reserve(resources, earliest, duration, owner):
        start = earliest
        while True:
            moved = max([start] + [calendars[r].available(start, duration) for r in resources])
            if moved == start:
                break
            start = moved
        for resource in resources:
            calendars[resource].reserve(start, duration, owner)
        return start, start+duration

    while ready or events:
        for key in sorted(ready, key=lambda k: (ready[k], k)):
            t, release = tasks[key], ready[key]
            kind = t["kind"]
            start, finish = now, now
            detail = {}
            if kind == "admit":
                if any(persistent[c]+t["scratch"] > hw.sram_bytes for c in t["cores"]):
                    raise Infeasible(f"job {t['job']} scratch plus resident exceeds SRAM")
                if any(persistent[c]+scratch_used[c]+t["scratch"] > hw.sram_bytes for c in t["cores"]):
                    continue
                for c in t["cores"]:
                    scratch_used[c] += t["scratch"]
                    peak_sram[c] = max(peak_sram[c], persistent[c]+scratch_used[c])
                job_start[t["job"]] = now
            elif kind == "release":
                for c in t["cores"]:
                    scratch_used[c] -= t["scratch"]
                job_end[t["job"]] = now
            elif kind == "compute":
                start, finish = reserve([f"core:{c}" for c in t["cores"]], now, t["duration"], key)
            elif kind == "network":
                cursor = now
                links = route(t["src"], t["dst"], hw)
                for src, dst in links:
                    _, cursor = reserve([f"link:{src}:{dst}"], cursor,
                                        math.ceil(t["bytes"]/hw.noc_bytepc)+hw.router_cycles, key)
                    link_bytes[f"{src}:{dst}"] += t["bytes"]
                finish = cursor
                traffic[f"{t['category']}_byte_hops"] += t["bytes"]*len(links)
                traffic["noc_byte_hops"] += t["bytes"]*len(links)
                detail["hops"] = len(links)
            elif kind == "memory":
                bank, row, n = t["bank"], t["row"], t["rows"]
                local = Counter()
                hit, cold = int(open_row[bank] == row), int(open_row[bank] < 0)
                if not hit and not cold:
                    category = ("revisit" if row in seen_rows[bank] else
                                "inter_tensor" if open_tensor[bank] != t["tensor"] else
                                "stream" if row == open_row[bank]+1 else "other")
                    local[category] += 1
                # Internal transitions can be revisits on a repeated sweep.
                revisits = sum(r in seen_rows[bank] for r in range(row+1, row+n))
                local["revisit"] += revisits
                local["stream"] += n-1-revisits
                count = sum(local.values())
                paid = count if counterfactual == "none" else (
                    0 if counterfactual == "all_conflicts" else count-local[counterfactual])
                row_cost = n*hw.cl+cold*hw.rcd+paid*(hw.rp+hw.rcd)
                start = max(now, bank_end[bank])
                bus_start, finish = reserve([f"channel:{t['channel']}"], start+row_cost,
                                            math.ceil(t["bytes"]/hw.channel_bytepc), key)
                calendars[f"bank:{bank}"].reserve(start, finish-start, key)
                bank_end[bank] = finish
                open_row[bank], open_tensor[bank] = row+n-1, t["tensor"]
                seen_rows[bank].update(range(row, row+n))
                bank_order[str(bank)].append(key)
                conflicts.update(local)
                counts.update(hits=hit, cold=cold, conflicts=count, row_touches=n)
                traffic[t["direction"]+"_bytes"] += t["bytes"]
                channel_bytes[t["channel"]] += t["bytes"]
                bank_bytes[bank] += t["bytes"]
                detail.update(conflicts=dict(local), row_ready=start+row_cost, bus_start=bus_start,
                              row=row, rows=n, bank=bank, channel=t["channel"], tensor=t["tensor"],
                              direction=t["direction"], bytes=t["bytes"], cold=cold, hits=hit)
            if trace:
                task_log.append(dict(id=key, kind=kind, job=t.get("job"), release=release,
                                     start=start, finish=finish, **detail))
            heapq.heappush(events, (finish, key))
            del ready[key]
        if not events:
            raise Infeasible("SRAM admission deadlock")
        now = events[0][0]
        while events and events[0][0] == now:
            finish, key = heapq.heappop(events)
            task_end[key] = finish
            for child in children[key]:
                indegree[child] -= 1
                if not indegree[child]:
                    ready[child] = now
    if len(task_end) != len(tasks) or any(scratch_used):
        raise AssertionError("incomplete task execution or leaked scratch")
    makespan = max(task_end.values(), default=0)
    nbytes = traffic["read_bytes"]+traffic["write_bytes"]
    energy = dict(dram_transfer=nbytes*hw.dram_pj_byte,
                  row_conflict=counts["conflicts"]*hw.conflict_pj,
                  noc_transport=traffic["noc_byte_hops"]*hw.noc_pj_byte_hop,
                  tsv_transfer=nbytes*hw.tsv_pj_byte)
    energy["modeled_movement_total"] = sum(energy.values())
    if counts["row_touches"] != counts["hits"]+counts["cold"]+counts["conflicts"]:
        raise AssertionError("row accounting mismatch")
    resource = {key: dict(busy_cycles=sum(b-a for a, b, _ in cal.intervals),
                          utilization=sum(b-a for a, b, _ in cal.intervals)/max(1, makespan))
                for key, cal in calendars.items()}
    groups = defaultdict(list)
    for j in graph["jobs"]:
        groups[str(j.get("microbatch", 0))].append(job_end[j["id"]])
    completions = sorted(max(v) for v in groups.values())
    return dict(model="l3_absolute_reservation_v2", paradigm=paradigm,
                counterfactual=counterfactual, chunk_bytes=chunk_bytes,
                metrics=dict(cycles=makespan, milliseconds=makespan/(1000*hw.frequency_mhz),
                             row=dict(counts), conflict_types={k: conflicts[k] for k in kinds},
                             traffic=dict(traffic), movement_energy_pj=energy,
                             microbatch_completions=completions,
                             completion_intervals=[b-a for a, b in zip(completions, completions[1:])],
                             peak_sram_bytes=peak_sram, job_start=job_start, job_end=job_end,
                             channel_bytes=dict(channel_bytes), bank_bytes=dict(bank_bytes),
                             link_bytes=dict(link_bytes), resources=resource),
                bank_order=dict(bank_order), bank_allocated_bytes=layout.bank_allocated_bytes,
                trace=task_log,
                limitations=["reservation scheduler; not command/packet accurate",
                             "fixed-burst DRAM and fixed-packet store-and-forward NoC",
                             "no tRAS/refresh/turnaround/static or compute energy",
                             "TSV channel service only; no vertical-distance model",
                             "resident copies reserved for entire run; no eviction"])
