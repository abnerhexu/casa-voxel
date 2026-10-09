"""Parsers for VOXEL simulation output files.

All parsers are pure functions on text (or small file reads) so they are
trivially testable.  Summary parsing reads only the head of a log file;
operator parsing streams the whole file.
"""

import re
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# Summary (header block of output_cg_*.log)
# ---------------------------------------------------------------------------

def parse_summary_text(content: str) -> Dict:
    """Parse the header block of a simulation log into flat metrics."""
    lines = content.split("\n")
    summary: Dict = {}

    def _f(pattern, line, group=1, cast=float):
        m = re.search(pattern, line)
        return cast(m.group(group)) if m else None

    first = lines[0] if lines else ""
    for key, pat in (("total_time", r"EXE time \(total, fused\): (\d+)"),
                     ("total_energy_mj", r"Energy \(mJ\): ([\d.eE+-]+)"),
                     ("static_energy_mj", r"Static: ([\d.eE+-]+) mJ"),
                     ("dynamic_energy_mj", r"Dyn\.: ([\d.eE+-]+) mJ")):
        v = _f(pat, first, 1, int if key == "total_time" else float)
        if v is not None:
            summary[key] = v

    def _components(line, out, prefix):
        pats = (("sa", r"SA = ([\d.eE+-]+)"), ("vu", r"VU = ([\d.eE+-]+)"),
                ("sram", r"SRAM= ([\d.eE+-]+)"), ("core", r"Core: ([\d.eE+-]+)"),
                ("dram", r"DRAM = ([\d.eE+-]+)"), ("tsv", r"TSV = ([\d.eE+-]+)"),
                ("noc", r"NoC: ([\d.eE+-]+)"))
        for name, pat in pats:
            v = _f(pat, line)
            if v is not None:
                out[f"{prefix}_{name}_mj"] = v

    if len(lines) > 1:
        _components(lines[1], summary, "static")
    if len(lines) > 2:
        _components(lines[2], summary, "static")
    if len(lines) > 3:
        _components(lines[3], summary, "dynamic")
    if len(lines) > 4:
        _components(lines[4], summary, "dynamic")

    pl = next((line for line in lines[:20] if line.startswith("Power (w):")), "")
    if pl:
        for key, pat in (("avg_power_w", r"Power \(w\): ([\d.eE+-]+)"),
                         ("static_power_w", r"Static: ([\d.eE+-]+) W"),
                         ("dynamic_power_w", r"Dyn\.: ([\d.eE+-]+) W")):
            v = _f(pat, pl)
            if v is not None:
                summary[key] = v
        v = _f(r"Static: [\d.eE+-]+ W \(dram: ([\d.eE+-]+)", pl)
        if v is not None:
            summary["static_power_dram_w"] = v
        v = _f(r"dram: [\d.eE+-]+ logic: ([\d.eE+-]+)", pl)
        if v is not None:
            summary["static_power_logic_w"] = v

    util_line = next((line for line in lines[:20] if line.startswith("Overall Util:")), "")
    if util_line:
        v = _f(r"Overall Util: ([\d.eE+-]+)", util_line)
        if v is not None:
            summary["overall_util"] = v

    cl = next((line for line in lines[:20] if line.startswith("DRAM UTIL")), "")
    if cl:
        m = re.search(r"DRAM UTIL \(%\): ([\d.eE+-]+)/([\d.eE+-]+)", cl)
        if m:
            summary["dram_r_util"] = float(m.group(1))
            summary["dram_w_util"] = float(m.group(2))
        for key, pat in (("sa_util", r"SA_UTIL:([\d.eE+-]+)"),
                         ("vu_util", r"VU_UTIL:([\d.eE+-]+)"),
                         ("noc_util", r"NOC: ([\d.eE+-]+)")):
            v = _f(pat, cl)
            if v is not None:
                summary[key] = v

    fl = next((line for line in lines[:16] if line.startswith("FLOPS:")), "")
    if fl:
        v = _f(r"FLOPS: ([\d.eE+-]+) GFLOPS MM", fl)
        if v is not None:
            summary["mm_gflops"] = v
        v = _f(r"([\d.eE+-]+) GFLOPS VU", fl)
        if v is not None:
            summary["vu_gflops"] = v

    noc_bh_line = next(
        (line for line in lines[:20] if line.startswith("NoC traffic x hops")),
        "",
    )
    if noc_bh_line:
        for key, pat in (
            ("noc_byte_hops", r"Total=([\d.eE+-]+)"),
            ("noc_bcast_byte_hops", r"Broadcast=([\d.eE+-]+)"),
            ("noc_shift_byte_hops", r"Shift=([\d.eE+-]+)"),
            ("noc_reduce_byte_hops", r"Reduce=([\d.eE+-]+)"),
        ):
            value = _f(pat, noc_bh_line)
            if value is not None:
                summary[key] = value

    dram_conflict_line = next(
        (line for line in lines[:20] if line.startswith("DRAM row conflicts")),
        "",
    )
    if dram_conflict_line:
        for key, pat in (
            ("dram_row_conflicts", r"Total=([\d.eE+-]+)"),
            ("dram_r_row_conflicts", r"Read=([\d.eE+-]+)"),
            ("dram_w_row_conflicts", r"Write=([\d.eE+-]+)"),
        ):
            value = _f(pat, dram_conflict_line)
            if value is not None:
                summary[key] = value

    dram_row_buffer_line = next(
        (line for line in lines[:24] if line.startswith("DRAM row buffer")),
        "",
    )
    if dram_row_buffer_line:
        for key, pat in (
            ("dram_row_hits", r"TotalHits=([\d.eE+-]+)"),
            ("dram_row_misses", r"TotalMisses=([\d.eE+-]+)"),
            ("dram_r_row_hits", r"ReadHits=([\d.eE+-]+)"),
            ("dram_r_row_misses", r"ReadMisses=([\d.eE+-]+)"),
            ("dram_w_row_hits", r"WriteHits=([\d.eE+-]+)"),
            ("dram_w_row_misses", r"WriteMisses=([\d.eE+-]+)"),
        ):
            value = _f(pat, dram_row_buffer_line)
            if value is not None:
                summary[key] = value
        placement = re.search(r"Placement=([^\s,]+)", dram_row_buffer_line)
        if placement:
            summary["dram_placement_policy"] = placement.group(1)

    dram_energy_line = next(
        (line for line in lines[:20]
         if line.startswith("DRAM dynamic energy breakdown")),
        "",
    )
    if dram_energy_line:
        for key, pat in (
            ("dynamic_dram_base_mj", r"Base=([\d.eE+-]+)"),
            ("dynamic_dram_row_conflict_mj", r"RowConflict=([\d.eE+-]+)"),
            ("dynamic_dram_mj", r"Total=([\d.eE+-]+)"),
        ):
            value = _f(pat, dram_energy_line)
            if value is not None:
                summary[key] = value

    dram_power_line = next(
        (line for line in lines[:20]
         if line.startswith("DRAM dynamic power (workload average)")),
        "",
    )
    if dram_power_line:
        for key, pat in (
            ("dynamic_power_dram_base_w", r"Base=([\d.eE+-]+)"),
            ("dynamic_power_dram_row_conflict_w", r"RowConflict=([\d.eE+-]+)"),
            ("dynamic_power_dram_w", r"Total=([\d.eE+-]+)"),
        ):
            value = _f(pat, dram_power_line)
            if value is not None:
                summary[key] = value

    return summary


def parse_summary_file(path, head_bytes: int = 4096) -> Optional[Dict]:
    """Parse summary metrics reading only the head of the log file."""
    try:
        with open(path, "r", errors="replace") as f:
            content = f.read(head_bytes)
    except OSError:
        return None
    if "EXE time" not in content:
        return None
    return parse_summary_text(content)


# ---------------------------------------------------------------------------
# Operators
# ---------------------------------------------------------------------------

_OP_HEADER_RE = re.compile(r"^Operator (\d+)=+")
_START_RE = re.compile(
    r"Ld:\s*(\d+)\s*->\s*broadcast\s*(\d+)\s*->\s*comp/sh\s*(\d+)\s*->\s*"
    r"reduce\s*(\d+)(?:\s*->\s*St:\s*(\d+))?\s*->\s*fin\s*(\d+)")
_DUR_RE = re.compile(
    r"Ld:\s*(\d+)\s*->\s*broadcast\s*(\d+)\s*->\s*comp/sh\s*(\d+)/(\d+)\s*->\s*"
    r"reduce\s*(\d+)(?:\s*->\s*St:\s*(\d+))?")
_UTIL_RE = re.compile(
    r"Compute Utilization:\s*([\d.eE+-]+)\s*VU Utilization:\s*([\d.eE+-]+)")
_BYTES_RE = re.compile(r"Write bytes:\s*(\d+)\s*Read bytes:\s*(\d+)")
_NOC_BYTE_HOPS_RE = re.compile(
    r"NoC traffic x hops \(byte-hop\):\s*"
    r"Total=([\d.eE+-]+),\s*Broadcast=([\d.eE+-]+),\s*"
    r"Shift=([\d.eE+-]+),\s*Reduce=([\d.eE+-]+)"
)
_DRAM_CONFLICT_RE = re.compile(
    r"DRAM row conflicts \(ACT\+PRE\):\s*"
    r"Total=([\d.eE+-]+),\s*Read=([\d.eE+-]+),\s*"
    r"Write=([\d.eE+-]+),\s*Energy=([\d.eE+-]+) pJ"
)
_DRAM_DYNAMIC_POWER_RE = re.compile(
    r"DRAM dynamic power \(W\):\s*"
    r"Total=([\d.eE+-]+),\s*Base=([\d.eE+-]+),\s*"
    r"RowConflict=([\d.eE+-]+)"
)
_DRAM_ROW_BUFFER_RE = re.compile(
    r"DRAM row buffer:\s*"
    r"ReadHits=([\d.eE+-]+),\s*ReadMisses=([\d.eE+-]+),\s*"
    r"WriteHits=([\d.eE+-]+),\s*WriteMisses=([\d.eE+-]+),\s*"
    r"Placement=([^\s,]+)"
)
_POWER_RE = re.compile(r"Average Power \(W\):\s*([\d.eE+-]+)")


def parse_operators_text(content: str) -> List[Dict]:
    """Parse per-operator 6-line blocks from a simulation log."""
    operators = []
    lines = content.split("\n")
    n = len(lines)
    for i, line in enumerate(lines):
        m = _OP_HEADER_RE.match(line)
        if not m or i + 4 >= n:
            continue
        op_id = int(m.group(1))
        start_m = _START_RE.search(lines[i + 1])
        dur_m = _DUR_RE.search(lines[i + 2])
        util_m = _UTIL_RE.search(lines[i + 3])
        bytes_m = _BYTES_RE.search(lines[i + 4])
        trailing_lines = lines[i + 5:min(i + 10, n)]
        noc_byte_hops_m = next(
            (_NOC_BYTE_HOPS_RE.search(candidate) for candidate in trailing_lines
             if _NOC_BYTE_HOPS_RE.search(candidate)),
            None,
        )
        power_m = next(
            (_POWER_RE.search(candidate) for candidate in trailing_lines
             if _POWER_RE.search(candidate)),
            None,
        )
        dram_conflict_m = next(
            (_DRAM_CONFLICT_RE.search(candidate) for candidate in trailing_lines
             if _DRAM_CONFLICT_RE.search(candidate)),
            None,
        )
        dram_dynamic_power_m = next(
            (_DRAM_DYNAMIC_POWER_RE.search(candidate)
             for candidate in trailing_lines
             if _DRAM_DYNAMIC_POWER_RE.search(candidate)),
            None,
        )
        dram_row_buffer_m = next(
            (_DRAM_ROW_BUFFER_RE.search(candidate)
             for candidate in trailing_lines
             if _DRAM_ROW_BUFFER_RE.search(candidate)),
            None,
        )
        if not (start_m and dur_m and util_m and bytes_m):
            continue
        operators.append({
            "op_id": op_id,
            "start_ld": int(start_m.group(1)),
            "start_bcast": int(start_m.group(2)),
            "start_comp": int(start_m.group(3)),
            "start_reduce": int(start_m.group(4)),
            "start_store": int(start_m.group(5)) if start_m.group(5)
                           else int(start_m.group(6)),
            "start_finish": int(start_m.group(6)),
            "dur_ld": int(dur_m.group(1)),
            "dur_bcast": int(dur_m.group(2)),
            "dur_comp": int(dur_m.group(3)),
            "dur_shift": int(dur_m.group(4)),
            "dur_reduce": int(dur_m.group(5)),
            "dur_store": int(dur_m.group(6)) if dur_m.group(6) else 0,
            "mm_util": float(util_m.group(1)),
            "vu_util": float(util_m.group(2)),
            "write_bytes": int(bytes_m.group(1)),
            "read_bytes": int(bytes_m.group(2)),
            "noc_byte_hops": float(noc_byte_hops_m.group(1)) if noc_byte_hops_m else 0.0,
            "noc_bcast_byte_hops": float(noc_byte_hops_m.group(2)) if noc_byte_hops_m else 0.0,
            "noc_shift_byte_hops": float(noc_byte_hops_m.group(3)) if noc_byte_hops_m else 0.0,
            "noc_reduce_byte_hops": float(noc_byte_hops_m.group(4)) if noc_byte_hops_m else 0.0,
            "dram_row_conflicts": float(dram_conflict_m.group(1)) if dram_conflict_m else 0.0,
            "dram_r_row_conflicts": float(dram_conflict_m.group(2)) if dram_conflict_m else 0.0,
            "dram_w_row_conflicts": float(dram_conflict_m.group(3)) if dram_conflict_m else 0.0,
            "energy_dram_row_conflict_pj": float(dram_conflict_m.group(4)) if dram_conflict_m else 0.0,
            "dram_dynamic_power_w": float(dram_dynamic_power_m.group(1)) if dram_dynamic_power_m else 0.0,
            "dram_base_dynamic_power_w": float(dram_dynamic_power_m.group(2)) if dram_dynamic_power_m else 0.0,
            "dram_row_conflict_dynamic_power_w": float(dram_dynamic_power_m.group(3)) if dram_dynamic_power_m else 0.0,
            "dram_r_row_hits": float(dram_row_buffer_m.group(1)) if dram_row_buffer_m else 0.0,
            "dram_r_row_misses": float(dram_row_buffer_m.group(2)) if dram_row_buffer_m else 0.0,
            "dram_w_row_hits": float(dram_row_buffer_m.group(3)) if dram_row_buffer_m else 0.0,
            "dram_w_row_misses": float(dram_row_buffer_m.group(4)) if dram_row_buffer_m else 0.0,
            "dram_placement_policy": dram_row_buffer_m.group(5) if dram_row_buffer_m else "unknown",
            "avg_power_w": float(power_m.group(1)) if power_m else 0.0,
        })
    return operators


def parse_operators_file(path) -> List[Dict]:
    with open(path, "r", errors="replace") as f:
        return parse_operators_text(f.read())


# ---------------------------------------------------------------------------
# Overlap log: per-interval active units + dynamic power
# ---------------------------------------------------------------------------

_INTERVAL_RE = re.compile(r"Interval:\s*cycles\s*=\[(\d+),\s*(\d+)\]\s*units\s*=\s*\{([^}]*)\}")
_DYNPOWER_RE = re.compile(r"Dynamic Power \(W\):\s*([\d.eE+-]+)")


def parse_overlap_text(content: str) -> List[Dict]:
    """Parse an overlap_cg_*.log into a list of interval dicts."""
    intervals = []
    pending = None
    for line in content.split("\n"):
        m = _INTERVAL_RE.search(line)
        if m:
            units = [u.strip().strip("'").strip('"')
                     for u in m.group(3).split(",") if u.strip()]
            pending = {"t_start": int(m.group(1)), "t_end": int(m.group(2)),
                       "units": sorted(set(units)), "power_w": 0.0}
            intervals.append(pending)
            continue
        p = _DYNPOWER_RE.search(line)
        if p and pending is not None:
            pending["power_w"] = float(p.group(1))
            pending = None
    return intervals


def parse_overlap_file(path) -> List[Dict]:
    with open(path, "r", errors="replace") as f:
        return parse_overlap_text(f.read())


def downsample_intervals(intervals: List[Dict], max_points: int = 4000) -> List[Dict]:
    """Merge adjacent intervals so the list stays small enough for the UI.

    Merging keeps the time-weighted average power and the union of units.
    """
    if len(intervals) <= max_points:
        return intervals
    factor = (len(intervals) + max_points - 1) // max_points
    merged = []
    for i in range(0, len(intervals), factor):
        chunk = intervals[i:i + factor]
        total_span = sum(max(1, c["t_end"] - c["t_start"]) for c in chunk)
        power = sum(c["power_w"] * max(1, c["t_end"] - c["t_start"])
                    for c in chunk) / total_span
        units = sorted({u for c in chunk for u in c["units"]})
        merged.append({"t_start": chunk[0]["t_start"],
                       "t_end": chunk[-1]["t_end"],
                       "units": units, "power_w": power})
    return merged


# ---------------------------------------------------------------------------
# Top-power log
# ---------------------------------------------------------------------------

def parse_top_power_text(content: str) -> List[Dict]:
    """top_power_cg_*.log has the same interval format, pre-filtered to the
    highest-power intervals."""
    return parse_overlap_text(content)


def parse_top_power_file(path) -> List[Dict]:
    with open(path, "r", errors="replace") as f:
        return parse_top_power_text(f.read())
