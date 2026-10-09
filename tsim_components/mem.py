"""Memory subsystem models for DRAM and SRAM.

Provides cycle-accurate access-cost estimation for DRAM (with row-open /
row-conflict timing) and SRAM, as well as silicon-area estimation helpers
for 3D-stacked and conventional memory technologies.

Key abstractions
----------------
* ``DRAM``  -- models row-based DRAM timing (CAS latency, tRCD, tRP),
               bank contention across cores, and per-core bandwidth.
* ``SRAM``  -- simple bandwidth-limited SRAM model.
* ``get_per_cycle_bytes_per_core_from_DRAM_config`` -- converts chip-level
  DRAM bandwidth into per-core, per-cycle byte count.
* ``get_sram_area_from_size`` / ``get_dram_area_from_size`` -- area
  estimators for technology-exploration sweeps.
"""

import sys
import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union, Set
from math import ceil

# Legacy default number of DRAM banks shared across all cores. New code uses
# ``DRAMGeometry.total_banks``; the constant remains part of the public API.
NUM_BANKS = 256

# Default DRAM timing parameters shared across the codebase.
# These match the values in hw_config/*.json and run_all_tests.py.
DEFAULT_CL   = 14
DEFAULT_TRCD = 14
DEFAULT_TRP  = 14

HBM_PACKAGE_CAPACITY_MB = 16 * 1024
"""Default HBM package capacity used by the TSIM area model."""

HBM_PACKAGE_AREA_MM2 = 87.62745402745404
"""Default HBM package footprint area in mm^2 from the TSIM area model."""


@dataclass(frozen=True)
class DRAMGeometry:
    """Physical DRAM geometry shared by placement and timing models.

    Banks are numbered first by layer and then within a layer. Channels are
    assigned by interleaving global bank IDs, which balances channel bank
    counts to within one while distributing each layer across channels.
    Transaction size is intentionally independent from sustained controller
    bandwidth.
    """

    num_layers: int = 8
    banks_per_layer: int = 32
    num_channels: int = 1
    bytes_per_row: int = 8192
    transaction_bytes: int = 128

    def __post_init__(self) -> None:
        for name in (
            "num_layers", "banks_per_layer", "num_channels",
            "bytes_per_row", "transaction_bytes",
        ):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
    @property
    def total_banks(self) -> int:
        return int(self.num_layers) * int(self.banks_per_layer)

    @property
    def banks_per_channel(self) -> int:
        """Maximum banks assigned to one channel.

        The legacy scalar is retained for callers which size rectangular
        arrays.  For non-divisible geometries use ``bank_counts_per_channel``
        to obtain the exact balanced distribution.
        """
        return int(ceil(self.total_banks / int(self.num_channels)))

    @property
    def bank_counts_per_channel(self) -> Tuple[int, ...]:
        """Exact modulo-interleaved bank count of every channel.

        Global bank ``b`` belongs to channel ``b % num_channels``.  The first
        ``total_banks % num_channels`` channels therefore receive one extra
        bank, which guarantees that channel loads differ by at most one.
        Channels beyond the bank count are legal and receive zero banks.
        """
        quotient, remainder = divmod(self.total_banks, int(self.num_channels))
        return tuple(
            quotient + (1 if channel < remainder else 0)
            for channel in range(int(self.num_channels))
        )

    def decode_bank(self, global_bank_id: int) -> Tuple[int, int, int, int]:
        """Return ``(channel, bank_in_channel, layer, bank_in_layer)``."""
        bank = int(global_bank_id)
        if not 0 <= bank < self.total_banks:
            raise ValueError(
                f"global_bank_id {bank} outside [0, {self.total_banks})"
            )
        layer, bank_in_layer = divmod(bank, int(self.banks_per_layer))
        channel = bank % int(self.num_channels)
        bank_in_channel = bank // int(self.num_channels)
        return channel, bank_in_channel, layer, bank_in_layer

    def as_dict(self) -> Dict[str, Any]:
        return {
            "num_layers": int(self.num_layers),
            "banks_per_layer": int(self.banks_per_layer),
            "total_banks": self.total_banks,
            "num_channels": int(self.num_channels),
            "banks_per_channel": self.banks_per_channel,
            "bank_counts_per_channel": list(self.bank_counts_per_channel),
            "min_banks_per_channel": min(self.bank_counts_per_channel),
            "max_banks_per_channel": max(self.bank_counts_per_channel),
            "bytes_per_row": int(self.bytes_per_row),
            "transaction_bytes": int(self.transaction_bytes),
        }

def get_hbm_package_count(
    dram_size_MB: int,
    package_capacity_MB: int = HBM_PACKAGE_CAPACITY_MB,
) -> int:
    """Return the number of HBM packages needed for a capacity."""
    return max(1, int(ceil(int(dram_size_MB) / max(1, int(package_capacity_MB)))))


def get_hbm_package_footprint_mm(
    package_area_mm2: float = HBM_PACKAGE_AREA_MM2,
    aspect_ratio: float = 1.0,
) -> Tuple[float, float]:
    """Return ``(width_mm, height_mm)`` for one HBM package footprint.

    ``aspect_ratio`` is width / height.  The area is preserved exactly.
    """
    aspect_ratio = max(1e-9, float(aspect_ratio))
    width_mm = (float(package_area_mm2) * aspect_ratio) ** 0.5
    height_mm = float(package_area_mm2) / width_mm
    return width_mm, height_mm


def get_per_cycle_bytes_per_core_from_DRAM_config(num_cores: int,
                                                  total_bandwidth_GBps: float,
                                                  npu_freq_MHz: int,
                                                  bandwidth_unit: str = "GB/s",
                                                  ) -> float:
    """Convert chip-level DRAM bandwidth to per-core, per-cycle byte count.

    Parameters
    ----------
    num_cores : int
        Number of cores sharing the total DRAM bandwidth.
    total_bandwidth_GBps : int
        Aggregate off-chip bandwidth.  The default unit is decimal GB/s.
    npu_freq_MHz : int
        Core clock frequency in MHz.

    Returns
    -------
    float
        Bytes each core can transfer in a single clock cycle.  Fractional
        rates are retained so a channel-count sweep does not silently lose
        bandwidth through integer floor division.
    """
    if int(num_cores) <= 0 or int(npu_freq_MHz) <= 0:
        raise ValueError("num_cores and npu_freq_MHz must be positive")
    normalized_unit = str(bandwidth_unit).strip().lower().replace(" ", "")
    if normalized_unit in {"gb/s", "gbps", "decimal"}:
        bytes_per_gb = 10**9
    elif normalized_unit in {"gib/s", "gibps", "binary"}:
        bytes_per_gb = 2**30
    else:
        raise ValueError(
            f"unsupported DRAM bandwidth unit {bandwidth_unit!r}; "
            "expected GB/s or GiB/s"
        )
    num_byte_per_cycle = (
        float(total_bandwidth_GBps) * bytes_per_gb
        / (float(npu_freq_MHz) * 10**6)
    )
    return num_byte_per_cycle / int(num_cores)

def get_sram_area_from_size(sram_size_KB: int, memtype="3D-SRAM") -> int:
    """Estimate SRAM silicon area (mm^2) for a given capacity.

    Parameters
    ----------
    sram_size_KB : int
        Desired SRAM capacity in KiB.
    memtype : str, optional
        Technology variant.  ``"3D-SRAM"`` uses DRAM-density numbers from
        Meta's AR/VR 3D chip paper.  ``"SRAM"`` uses a piecewise-linear
        regression fitted to McPAT area sweeps (4 KB -- 1 MB); a slope
        discontinuity exists around 640 KB due to McPAT's internal bank
        restructuring.

    Returns
    -------
    float
        Estimated area in mm^2.
    """
    area_sq_mm = -1
    if memtype == "3D-SRAM":
        sram_MB_per_sq_mm = 4  # density from Meta AR/VR 3D chip paper
        area_sq_mm = sram_size_KB / 1024 * sram_MB_per_sq_mm
    elif memtype == "SRAM":
        # Piecewise-linear fit from McPAT area sweep.
        # Slope change at 640 KB is caused by McPAT's internal bank
        # restructuring at that capacity boundary.
        if sram_size_KB <= 640:
            area_sq_mm = 5.74e-04 * sram_size_KB + 0.133
        else:
            area_sq_mm = 6.7e-04 * sram_size_KB + 0.322
    else:
        print( "area was not set! Invalid memtype!")
        exit(-1)
    return area_sq_mm

def get_dram_area_from_size(dram_size_MB: int, memtype="3D-DRAM") -> int:
    """Estimate DRAM silicon area (mm^2) for a given capacity.

    Parameters
    ----------
    dram_size_MB : int
        Desired DRAM capacity in MiB.
    memtype : str, optional
        ``"3D-DRAM"`` -- uses density of 8.4 MB/mm^2 from
        `<https://openreview.net/pdf?id=P4LViaB8g0>`_.
        ``"HBM"`` -- each HBM die is 16 GB; die area (~87.6 mm^2) was
        measured from an A100 die photo (GPU die 826 mm^2, HBM die
        proportionally scaled from pixel measurements).

    Returns
    -------
    float
        Estimated area in mm^2.  For HBM, the result is rounded up to
        the next whole die.
    """
    area_sq_mm = -1
    if memtype == "3D-DRAM":
        MB_per_sq_mm = 8.4  # 3D-stacked DRAM density
        area_sq_mm = dram_size_MB / MB_per_sq_mm
    elif memtype == "HBM":
        mem_per_die_MB = 16 * 1024  # 16 GiB per HBM die
        area_per_die_sq_mm = 87.62745402745404  # from A100 die-photo measurement
        # Round up to whole HBM dies.
        area_sq_mm = area_per_die_sq_mm * ceil(dram_size_MB / mem_per_die_MB)
    else:
        print( "area was not set! Invalid memtype!")
        exit(-1)
    return area_sq_mm

class SRAM:
    """Simple bandwidth-limited SRAM model.

    Access latency is computed purely from the data volume divided by the
    sustained bandwidth (bytes per cycle).  No row/bank modelling.

    Parameters
    ----------
    bandwidth_bytepc : float
        Sustained read/write bandwidth in bytes per cycle.
    num_layers : int, optional
        Number of 3D-stacked SRAM layers (default 4).
    """

    def __init__(self,
                 bandwidth_bytepc: float,
                 num_layers: int = 4) -> None:
        self.bandwidth_bytepc = bandwidth_bytepc
        self.num_layers = num_layers

    def num_cycle_of_access(self, num_bytes: int) -> float:
        """Return the number of cycles to transfer *num_bytes* at the
        configured bandwidth."""
        return num_bytes / self.bandwidth_bytepc

class DRAM:
    """Cycle-level DRAM access-cost model with row-open / row-conflict timing.

    The model accounts for three key DRAM timing parameters:

    * **CL** (CAS Latency) -- column-access strobe to data.
    * **tRCD** (RAS-to-CAS Delay) -- row activation to column command.
    * **tRP** (Row Precharge) -- minimum time to close a row before
      opening a new one.

    A *row reopen* costs ``CL + tRCD + tRP`` cycles.  The number of
    reopens is determined by both the access granularity (how data is
    tiled across cores) and the row size.  Bank contention is modelled
    by scaling reopen cost by ``num_cores / NUM_BANKS``.

    Parameters
    ----------
    CL, tRCD, tRP : int
        DRAM timing parameters in core clock cycles.
    bytes_per_row : int
        Size of one DRAM row (row buffer) in bytes.
    bytes_per_cycle : int
        Data-bus width in bytes (burst transfer per cycle).
    num_cores : int
        Total number of cores sharing the DRAM.
    num_layers : int, optional
        Number of 3D-stacked DRAM layers (default 8).
    use_sram : bool, optional
        If True, bypass DRAM timing and return a simple fixed-rate
        estimate (used for SRAM-only design-point sweeps).
    """

    def __init__(self,
                 CL: int,
                 tRCD: int,
                 tRP: int,
                 bytes_per_row: int,
                 bytes_per_cycle: int,
                 num_cores: int,
                 num_layers: int = 8,
                 use_sram: bool = False,
                 precise: bool = False,
                 num_banks_per_channel: Optional[int] = None,
                 banks_per_layer: int = 32,
                 num_channels: int = 1,
                 transaction_bytes: int = 128,
                 geometry: Optional[DRAMGeometry] = None,
                 tRRD: int = 4,
                 tFAW: int = 20,
                 tRFC: int = 350,
                 tREFI: int = 12480,
                 precise_cache_size: int = 4096,
                 ultra_precise: bool = False,
                 ultra_backend: Optional[Any] = None,
                 ultra_cache_size: int = 8192,
                 lock_cores_per_bank: float = 0,
                 soft_cores_per_bank: bool = True,
                 capacity_bytes: Optional[int] = None,
                ) -> None:
        self.CL: int = CL
        self.tRCD: int = tRCD
        self.tRP: int = tRP
        # Full row-reopen penalty: activate + column-access + precharge.
        self.reopen: int = CL + tRCD + tRP
        # ``num_banks_per_channel`` is a legacy constructor argument. When it
        # is explicitly supplied without a geometry, preserve its historical
        # single-channel meaning while representing it through DRAMGeometry.
        if geometry is None:
            if num_banks_per_channel is not None:
                legacy_total_banks = int(num_banks_per_channel) * int(num_channels)
                if legacy_total_banks % int(num_layers) == 0:
                    legacy_layers = int(num_layers)
                    legacy_banks_per_layer = legacy_total_banks // legacy_layers
                else:
                    legacy_layers = 1
                    legacy_banks_per_layer = legacy_total_banks
                geometry = DRAMGeometry(
                    num_layers=legacy_layers,
                    banks_per_layer=legacy_banks_per_layer,
                    num_channels=int(num_channels),
                    bytes_per_row=int(bytes_per_row),
                    transaction_bytes=int(transaction_bytes),
                )
            else:
                geometry = DRAMGeometry(
                    num_layers=int(num_layers),
                    banks_per_layer=int(banks_per_layer),
                    num_channels=int(num_channels),
                    bytes_per_row=int(bytes_per_row),
                    transaction_bytes=int(transaction_bytes),
                )
        elif int(bytes_per_row) != int(geometry.bytes_per_row):
            raise ValueError(
                "bytes_per_row disagrees with geometry.bytes_per_row: "
                f"{bytes_per_row} != {geometry.bytes_per_row}"
            )

        self.geometry: DRAMGeometry = geometry
        self.bytes_per_row: int = int(geometry.bytes_per_row)
        # Aggregate per-core service rate across all channels.  Keep the
        # historical attribute name as an alias because a large part of TSim
        # reads it directly.
        self.total_bytes_per_cycle: float = float(bytes_per_cycle)
        if self.total_bytes_per_cycle <= 0:
            raise ValueError("bytes_per_cycle must be positive")
        self.bytes_per_cycle: float = self.total_bytes_per_cycle
        self.num_cores: int = num_cores
        self.num_layers: int = int(geometry.num_layers)
        self.banks_per_layer: int = int(geometry.banks_per_layer)
        self.num_channels: int = int(geometry.num_channels)
        self.num_banks: int = int(geometry.total_banks)
        # Compatibility attribute used by callers that inspect the old name.
        self.num_banks_per_channel: int = int(geometry.banks_per_channel)
        self.transaction_bytes: int = int(geometry.transaction_bytes)
        self.capacity_bytes: Optional[int] = (
            int(capacity_bytes) if capacity_bytes is not None else None
        )
        self.use_sram: bool = use_sram
        # Switch: when non-zero, lock cores_per_bank to this fixed value
        # (e.g. 2 = pin to the default 256-core / 128-bank ratio) instead of
        # the actual num_cores/NUM_BANKS. 0 (default) = off; any model can opt
        # in via the constructor / CLI flag.
        self.lock_cores_per_bank: float = lock_cores_per_bank
        # Soft mode: compress cores_per_bank above 2 via sqrt(2*raw).
        self.soft_cores_per_bank: bool = soft_cores_per_bank
        # Pre-compute: immutable per DRAM instance, called in hot paths.
        self._cpb: float = self._compute_cores_per_bank()

        # --- Precise (request-level) DRAM mode ---
        # When ``precise=True``, ``num_cycle_of_access`` synthesizes a stream
        # of burst-sized requests and drives a per-bank state machine that
        # tracks open rows, tRRD/tFAW activation throttling, and amortized
        # tRFC refresh penalty. Results
        # are coalesced via a (num_bytes, granularity, need_init) cache so
        # structurally equivalent access patterns reuse a prior simulation.
        # Defaults are off — when ``precise=False`` the original analytical
        # fast path is used unchanged.
        self.precise: bool = precise
        self.tRRD: int = tRRD
        self.tFAW: int = tFAW
        self.tRFC: int = tRFC
        self.tREFI: int = tREFI
        self._precise_cache_size: int = precise_cache_size
        self._precise_cache: Dict[Tuple[int, int, bool], int] = {}

        # --- Ultra-precise (external simulator) DRAM mode ---
        # Defers latency estimation to a real cycle-accurate DRAM simulator
        # (Ramulator 2.0 or DRAMsim3) via subprocess. The backend object is
        # discovered/instantiated by the caller (icbm_launch.get_hw_modules)
        # and dropped here; if it failed to instantiate we silently leave
        # ``ultra_precise=False`` and the caller may downgrade to
        # ``precise=True``. Same coalescing key as the precise path: the
        # synthesized stream is fully determined by (num_bytes, granularity,
        # need_init), so per-key caching avoids re-simulating
        # signature.
        self.ultra_precise: bool = ultra_precise and (ultra_backend is not None)
        self._ultra_backend: Optional[Any] = ultra_backend if self.ultra_precise else None
        self._ultra_cache_size: int = ultra_cache_size
        self._ultra_cache: Dict[Tuple[int, int, bool], int] = {}
        if self.ultra_precise:
            self._populate_from_dram_cache()

    @property
    def channel_bytes_per_cycle(self) -> float:
        """Per-channel service rate under the fixed total-bandwidth budget."""
        return self.total_bytes_per_cycle / self.num_channels

    def bandwidth_geometry(self) -> Dict[str, float]:
        """Return the decoupled aggregate/channel bandwidth description."""
        return {
            "total_bytes_per_cycle": self.total_bytes_per_cycle,
            "channel_bytes_per_cycle": self.channel_bytes_per_cycle,
            "num_channels": self.num_channels,
            "transaction_bytes": self.transaction_bytes,
        }

    def tiling_cache_signature(self) -> Tuple[object, ...]:
        """Return DRAM properties that may change tiling selection.

        Channel count and per-channel bandwidth are intentionally excluded:
        a channel-count sweep reuses tiling and reruns only placement/timing.
        Aggregate bandwidth remains in the key because bandwidth sweeps must
        be allowed to choose a different tiling.
        """
        return (
            "dram-tiling-v1",
            float(self.total_bytes_per_cycle),
            int(self.CL),
            int(self.tRCD),
            int(self.tRP),
            int(self.bytes_per_row),
            int(self.num_banks),
            int(self.transaction_bytes),
            float(self._cpb),
            bool(self.use_sram),
        )

    def num_cycle_of_access_for_tiling(
        self,
        num_bytes: int,
        access_granularity_bytes: int,
        need_init: bool = False,
    ) -> int:
        """Channel-neutral analytical cost used only for tiling selection.

        Final execution uses the placement-aware session.  Keeping this
        search estimate independent of channel topology lets channel sweeps
        reuse a previously selected tiling while aggregate-bandwidth sweeps
        still obtain distinct cache entries.
        """
        if num_bytes <= 0:
            return 0
        if self.use_sram:
            return int(num_bytes / 3.57)
        granularity = max(1, int(access_granularity_bytes))
        num_reopen = max(
            ceil(num_bytes / granularity),
            ceil(num_bytes / self.bytes_per_row),
        )
        if not need_init and num_reopen <= 1:
            max_granularity = min(granularity, self.bytes_per_row)
            num_reopen = num_bytes / max(1, max_granularity)
        row_cycles = num_reopen * self.reopen * self._cpb
        row_cycles = max(0.0, row_cycles - self.tRP)
        transfer_cycles = ceil(num_bytes / self.total_bytes_per_cycle)
        return int(max(1, row_cycles + transfer_cycles))

    def new_execution_session(
        self,
        placement_policy: str = "software_aware",
        replication_factor: int = 1,
        frfcfs_window: Optional[int] = None,
    ):
        """Create a stateful placement-aware timing/conflict session.

        The import is local to keep the low-level memory model independent of
        placement policy implementation details at module import time.
        """
        from tsim_components.dram_scheduler import DRAMExecutionSession
        return DRAMExecutionSession(
            self,
            placement_policy=placement_policy,
            replication_factor=replication_factor,
            frfcfs_window=(
                int(frfcfs_window) if frfcfs_window is not None
                else int(getattr(self, "frfcfs_window", 32))
            ),
        )

    def _populate_from_dram_cache(self) -> None:
        """Populate ultra cache from pre-computed dram_cache files (``*.dcache``).

        Only called when ultra_precise is active with a valid backend
        (no fallback).  Each cached trace is converted to a cache entry:
        (num_bytes, granularity, need_init) -> total_cycles.
        """
        try:
            from tsim_components.dram_external import load_dram_cache
            cache_data = load_dram_cache()
        except Exception:
            return
        for trace in cache_data:
            if not trace:
                continue
            n = len(trace)
            # need_init: first access is read (rw=0) -> need_init=True
            need_init = (trace[0][1] == 0)
            # granularity: XOR of adjacent address deltas
            prev = trace[0][0]
            gran = 0
            for i in range(1, min(n, 64)):  # sample first 64 entries
                gran |= (trace[i][0] ^ prev)
                prev = trace[i][0]
            if gran == 0:
                gran = 1
            # total cycles = sum of latencies
            cycles = sum(e[2] for e in trace)
            key = (n, gran, need_init)
            self._ultra_cache[key] = cycles

    def _compute_cores_per_bank(self) -> float:
        """Effective cores-per-bank used for bank-contention row scaling.

        Computed once in ``__init__`` and cached as ``self._cpb``.

        Modes (priority order):
          * ``lock_cores_per_bank`` (non-zero) -> pin to that fixed value.
        """
        if self.lock_cores_per_bank:
            return float(self.lock_cores_per_bank)
        if self.soft_cores_per_bank:
            return max(1.0, (self.num_cores / self.num_banks) ** .8)
        return max(1.0, self.num_cores / self.num_banks)

    def _cores_per_bank(self) -> float:
        """Return the cached cores-per-bank value (computed in __init__).

        Kept for backward compatibility; prefer ``self._cpb`` directly.
        """
        return self._cpb

    def num_cycle_of_access(self, num_bytes: int,
                            access_granularity_bytes: int,
                            need_init: bool = False) -> float:
        """Estimate the total cycle cost for a single DRAM access.

        The cost has two components:

        1. **Row-reopen overhead** -- each non-contiguous access to a new
           DRAM row incurs a full reopen penalty (``CL + tRCD + tRP``).
           The number of reopens is the *maximum* of the reopens implied
           by the access granularity and by the physical row size, then
           scaled by bank contention (``num_cores / NUM_BANKS``).
        2. **Column transfer time** -- the raw number of bus cycles to
           move the data (``ceil(num_bytes / bytes_per_cycle)``).

        Parameters
        ----------
        num_bytes : int
            Total bytes to transfer for this access.
        access_granularity_bytes : int
            Contiguous chunk size per access (determined by tensor tiling).
        need_init : bool, optional
            Whether the first access requires a fresh row activation
            (True) or can piggy-back on an already-open row (False).

        Returns
        -------
        int
            Estimated access latency in clock cycles (minimum 1).
        """
        if num_bytes == 0:
            return 0
        if self.use_sram:
            # Bypass DRAM model: use a simple fixed-rate estimate for
            # SRAM-only design points (empirical ~3.57 bytes/cycle).
            return int(num_bytes / 3.57)
        if self.ultra_precise:
            # External backends currently accept one isolated-channel trace.
            # Use the internal channel-aware path for multi-channel geometry
            # so concurrency is not lost.
            if self.num_channels > 1:
                return self._precise_num_cycle_of_access(
                    num_bytes, access_granularity_bytes, need_init
                )
            return self._ultra_precise_num_cycle_of_access(num_bytes,
                                                           access_granularity_bytes,
                                                           need_init)
        if self.precise:
            return self._precise_num_cycle_of_access(num_bytes,
                                                     access_granularity_bytes,
                                                     need_init)

        cycle = 0

        # --- Row-reopen count estimation ---
        # Reopens from access granularity: each granularity-sized chunk
        # may land in a different DRAM row.
        num_reopen_granularity = (num_bytes + access_granularity_bytes - 1) // access_granularity_bytes
        # Reopens from row size: data spanning multiple rows forces reopens.
        num_reopen_row_limit = (num_bytes + self.bytes_per_row - 1) // self.bytes_per_row
        # The binding constraint determines the actual reopen count.
        num_reopen = max(num_reopen_granularity, num_reopen_row_limit)

        if need_init == False and num_reopen <= 1:
            # Continuing from an already-open row: the access fits within
            # one row, so compute a fractional reopen cost proportional to
            # the fraction of the row/granularity actually used.
            max_access_granularity = min(access_granularity_bytes, self.bytes_per_row)
            num_reopen = 1 / (max_access_granularity // num_bytes)

        # Row work on different channels can proceed independently. The
        # slowest channel receives the ceiling of the striped reopen count.
        if num_reopen < 1:
            channel_reopens = num_reopen
        else:
            channel_reopens = ceil(num_reopen / self.num_channels)
        cycle += channel_reopens * self.reopen

        # --- Bank contention scaling ---
        # Multiple cores sharing the same bank serialize their row activations.
        cycle *= self._cpb
        # Subtract one tRP: the very last access does not need to precharge
        # for a subsequent row (pipeline overlap with next command).
        cycle -= self.tRP

        # --- Column (burst) transfer time ---
        num_cols = (num_bytes + self.bytes_per_cycle - 1) // self.bytes_per_cycle
        cycle += num_cols

        return int(max(1, cycle))

    def num_row_conflicts_of_access(self, num_bytes: int,
                                    access_granularity_bytes: int,
                                    need_init: bool = False) -> int:
        """Estimate ACT+PRE row-conflict pairs for one DRAM access.

        The access stream uses the same chunking assumption as the precise
        timing model: each contiguous granularity-sized chunk occupies a new
        row and chunks are striped round-robin across the banks.  The first
        row opened in a bank is an initial row miss (ACT only), not a row
        conflict.  Each later row mapped to that bank replaces an open row
        and therefore contributes one PRE+ACT pair.

        ``need_init`` affects whether the first access can hit a pre-opened
        row, but it does not change the number of later row replacements, so
        it is accepted for API symmetry with :meth:`num_cycle_of_access`.
        SRAM-only configurations report no DRAM conflicts.
        """
        del need_init
        if num_bytes <= 0 or self.use_sram:
            return 0

        bytes_per_burst = max(1, int(self.transaction_bytes))
        bytes_per_row = max(bytes_per_burst, int(self.bytes_per_row))
        granularity = max(
            bytes_per_burst,
            min(int(access_granularity_bytes), bytes_per_row),
        )
        bursts_per_chunk = max(1, ceil(granularity / bytes_per_burst))
        num_bursts = (int(num_bytes) + bytes_per_burst - 1) // bytes_per_burst
        num_chunks = (num_bursts + bursts_per_chunk - 1) // bursts_per_chunk
        num_banks = max(1, int(self.num_banks))
        return max(0, num_chunks - num_banks)

    def _precise_num_cycle_of_access(self,
                                     num_bytes: int,
                                     access_granularity_bytes: int,
                                     need_init: bool) -> int:
        """Request-level DRAM latency estimator with bank-state simulation.

        Synthesizes a burst-sized request stream from
        ``(num_bytes, access_granularity_bytes)`` and runs it against a
        finite-state model of ``num_banks_per_channel`` banks. Captures:

        * Row-buffer hits vs. row-conflicts (CL / tRCD / tRP).
        * Bank-level parallelism — independent banks can pipeline
          activations and data transfers.
        * tRRD (minimum interval between activations on the same bank)
          and tFAW (max 4 activations within a sliding window).
        * Amortized refresh penalty — the long-run fraction of cycles
          spent in tRFC stalls (tRFC / tREFI).

        Bank contention from multiple cores sharing the same bank is
        applied on top, mirroring the heuristic of the fast path: only
        the row-overhead component scales by ``num_cores / NUM_BANKS``,
        the burst component is bandwidth-bound and does not.

        Results are cached by ``(num_bytes, granularity, need_init)`` —
        the synthetic stream is fully determined by these parameters, so
        the cache key plays the same role as the paper's match-key
        signature for repeated traces.
        """
        ag = access_granularity_bytes
        if ag <= 0:
            ag = self.bytes_per_cycle

        cache_key = (int(num_bytes), int(ag), bool(need_init))
        cached = self._precise_cache.get(cache_key)
        if cached is not None:
            cycles = cached
        else:
            cycles = self._simulate_request_stream(num_bytes, ag, need_init)
            if len(self._precise_cache) < self._precise_cache_size:
                self._precise_cache[cache_key] = cycles

        # Bank contention (mirrors fast path): when more cores than banks,
        # row-related work serializes; bus throughput does not.
        cores_per_bank = self._cpb
        if cores_per_bank > 1:
            burst_floor = (num_bytes + self.bytes_per_cycle - 1) // self.bytes_per_cycle
            row_overhead = max(0, cycles - burst_floor)
            cycles = burst_floor + row_overhead * cores_per_bank

        return int(max(1, cycles))

    def _simulate_request_stream(self,
                                 num_bytes: int,
                                 access_granularity: int,
                                 need_init: bool) -> int:
        """Drive the per-bank state machine over a synthetic burst stream.

        Address layout assumption: each ``access_granularity``-sized chunk
        opens a fresh row, chunks round-robin across all physical banks, and
        global banks are interleaved across channels. Each channel has an
        independent data bus whose bandwidth is the fixed total bandwidth
        divided by the number of channels. The access completes when its
        slowest channel completes.
        """
        transaction_bytes = max(1, int(self.transaction_bytes))
        bpr = max(transaction_bytes, int(self.bytes_per_row))
        # Each chunk maps to one row. Don't let granularity exceed a row
        # (would imply spanning multiple rows in one chunk — not modelled).
        ag_eff = max(transaction_bytes, min(int(access_granularity), bpr))
        bursts_per_chunk = max(1, ceil(ag_eff / transaction_bytes))
        num_bursts = max(1, ceil(int(num_bytes) / transaction_bytes))
        num_chunks = (num_bursts + bursts_per_chunk - 1) // bursts_per_chunk

        total_banks = max(1, self.num_banks)
        banks_per_channel = self.geometry.banks_per_channel
        # Per-channel bank state and independent shared data buses.
        bank_open_row: List[List[int]] = [
            [-1] * banks_per_channel for _ in range(self.num_channels)
        ]
        bank_free_time: List[List[int]] = [
            [0] * banks_per_channel for _ in range(self.num_channels)
        ]
        last_act_per_bank: List[List[int]] = [
            [-self.tRRD] * banks_per_channel for _ in range(self.num_channels)
        ]
        act_windows: List[List[int]] = [[] for _ in range(self.num_channels)]
        bus_free_time: List[int] = [0] * self.num_channels

        # If continuing from a previously open row, pre-open bank 0 row 0
        # so the first chunk hits without paying the activation penalty.
        if not need_init:
            bank_open_row[0][0] = 0

        for chunk_idx in range(num_chunks):
            global_bank = chunk_idx % total_banks
            channel, bank, _layer, _bank_in_layer = \
                self.geometry.decode_bank(global_bank)
            row = chunk_idx // total_banks
            bursts_remaining = num_bursts - chunk_idx * bursts_per_chunk
            bursts_this = min(bursts_per_chunk, bursts_remaining)
            bytes_remaining = int(num_bytes) - chunk_idx * bursts_per_chunk * transaction_bytes
            bytes_this = min(
                max(0, bytes_remaining),
                bursts_this * transaction_bytes,
            )

            if bank_open_row[channel][bank] != row:
                # Need (precharge if open) + activate + tRCD before CAS.
                t_ready = bank_free_time[channel][bank]
                if bank_open_row[channel][bank] != -1:
                    t_pre_done = t_ready + self.tRP
                else:
                    t_pre_done = t_ready
                # tRRD: min interval to previous activation on this bank.
                t_act = max(
                    t_pre_done,
                    last_act_per_bank[channel][bank] + self.tRRD,
                )
                # tFAW is enforced independently by each channel.
                act_window = act_windows[channel]
                if len(act_window) >= 4:
                    t_act = max(t_act, act_window[-4] + self.tFAW)
                act_window.append(t_act)
                if len(act_window) > 4:
                    # Keep only the trailing four — that's all tFAW needs.
                    act_windows[channel] = act_window[-4:]
                last_act_per_bank[channel][bank] = t_act
                bank_open_row[channel][bank] = row
                t_cas_cmd = t_act + self.tRCD
            else:
                # Row hit — go straight to CAS.
                t_cas_cmd = bank_free_time[channel][bank]

            # The 128-byte transaction size is independent from bandwidth.
            # A channel may therefore need multiple cycles to transfer one
            # transaction when the channel count is increased under a fixed
            # aggregate bandwidth budget.
            transfer_cycles = max(
                1, ceil(bytes_this / self.channel_bytes_per_cycle)
            )
            t_data_first = max(
                t_cas_cmd + self.CL,
                bus_free_time[channel],
            )
            t_data_last = t_data_first + transfer_cycles
            bus_free_time[channel] = t_data_last
            bank_free_time[channel][bank] = t_data_last

        cur_time = max(bus_free_time, default=0)

        # Amortized refresh: the fraction of time spent in tRFC stalls.
        # A real DRAM stalls one rank for tRFC cycles every tREFI cycles;
        # over a long run that's a tRFC/tREFI multiplicative slowdown.
        # For short accesses this rounds to ~zero overhead.
        if self.tREFI > 0:
            cur_time += (cur_time * self.tRFC) // self.tREFI

        return cur_time

    def _ultra_precise_num_cycle_of_access(self,
                                           num_bytes: int,
                                           access_granularity_bytes: int,
                                           need_init: bool) -> int:
        """Defer to the configured external DRAM simulator backend.

        Each unique ``(num_bytes, granularity, need_init)`` tuple is fed to
        the backend exactly once; results are cached for subsequent calls.
        On any subprocess failure we fall back to the pure-Python precise
        simulator and warn once per process so users notice.
        """
        ag = access_granularity_bytes
        if ag <= 0:
            ag = self.bytes_per_cycle

        cache_key = (int(num_bytes), int(ag), bool(need_init))
        cached = self._ultra_cache.get(cache_key)
        if cached is not None:
            cycles = cached
        else:
            try:
                cycles = self._ultra_backend.simulate_stream(
                    num_bytes=num_bytes,
                    access_granularity=ag,
                    need_init=need_init,
                    bytes_per_cycle=self.bytes_per_cycle,
                    bytes_per_row=self.bytes_per_row,
                    num_banks_per_channel=self.num_banks_per_channel,
                )
            except Exception as e:
                if not getattr(self, "_ultra_failed_once", False):
                    print(
                        f"WARNING: ultra-precise DRAM backend "
                        f"{getattr(self._ultra_backend, 'name', '?')} "
                        f"failed ({e}); falling back to the pure-Python "
                        f"precise simulator for this and subsequent calls.",
                        file=sys.stderr, flush=True)
                    self._ultra_failed_once = True
                # Permanent downgrade to avoid hammering a broken subprocess.
                self.ultra_precise = False
                if not self.precise:
                    self.precise = True
                return self._precise_num_cycle_of_access(num_bytes, ag, need_init)
            if len(self._ultra_cache) < self._ultra_cache_size:
                self._ultra_cache[cache_key] = cycles

        # Same bank-contention scaling as precise/fast: row-overhead serializes
        # across cores sharing a bank, burst transfer is bandwidth-bound.
        cores_per_bank = self._cpb
        if cores_per_bank > 1:
            burst_floor = (num_bytes + self.bytes_per_cycle - 1) // self.bytes_per_cycle
            row_overhead = max(0, cycles - burst_floor)
            cycles = burst_floor + row_overhead * cores_per_bank
        return int(max(1, cycles))

    def get_dram_access_list(self, tensor_shapes: List[np.ndarray],
                             temporal_var_replicas: List[int],
                             core_group_size: int,
                             num_byte_per_elem: int,
                             return_granularity: bool = False,
                             return_tot_bytes_per_core: bool = False,
                             return_cycles_and_bytes: bool = False,
                             return_cycles_bytes_granularity: bool = False,
                             return_cycles_bytes_granularity_conflicts: bool = False,
                             opti_intra_mapping: bool = False,
                             bad_mapping: bool = False,
                             for_tiling: bool = False):
        """Compute per-tensor DRAM access costs for a list of tensors.

        For each tensor the method calculates:

        * **total bytes per core** -- element count (after replica
          division) times ``num_byte_per_elem``.
        * **access granularity** -- contiguous byte chunk each core
          touches, derived from the tensor's largest dimension, the
          replica count, and the core-group size.

        What is returned depends on the ``return_*`` flags (exactly one
        should be True, or all False for the default cycle-count mode):

        * ``return_granularity`` -- list of access granularities (bytes).
        * ``return_tot_bytes_per_core`` -- list of total bytes per core.
        * ``return_cycles_and_bytes`` -- tuple of (cycles_list, bytes_list).
        * ``return_cycles_bytes_granularity`` -- tuple of (cycles_list,
          bytes_list, granularity_list).
        * ``return_cycles_bytes_granularity_conflicts`` -- tuple of
          (cycles_list, bytes_list, granularity_list, row_conflicts_list).
        * *(default)* -- list of access cycle counts.

        Parameters
        ----------
        tensor_shapes : list of np.ndarray
            Shape arrays for [output, input0, input1, ...].
        temporal_var_replicas : list of int
            Temporal reuse factor per tensor (divides total bytes).
        core_group_size : int
            Number of cores cooperating on the same tile.
        num_byte_per_elem : int
            Element width (e.g. 2 for FP16).
        opti_intra_mapping : bool, optional
            If True, assume an optimised intra-core mapping where the
            full core-group accesses data contiguously (no row reopens
            on first access).
        bad_mapping : bool, optional
            If True, simulate a sub-optimal mapping where input tensors
            use only half their last dimension for granularity.
        for_tiling : bool, optional
            Use the channel-neutral aggregate-bandwidth estimator. This is
            reserved for tiling selection; final execution should leave it
            False and use the placement-aware execution session.
        """
        dram_access_list: List[float] = []
        detailed = return_cycles_bytes_granularity or return_cycles_bytes_granularity_conflicts
        dram_bytes_list: List[int] = [] if (return_cycles_and_bytes or detailed) else None
        dram_granularity_list: List[int] = [] if detailed else None
        dram_row_conflicts_list: List[int] = [] if return_cycles_bytes_granularity_conflicts else None

        for i, (shape, replica) in enumerate(zip(tensor_shapes, temporal_var_replicas)):
            # Collapse singleton dimensions so they don't inflate the
            # granularity calculation.
            shape1 = shape[shape > 1]
            if len(shape1) == 0:
                shape1 = np.array([1])

            # Total bytes this core must load/store for the tensor.
            tot_access_bytes: int = ceil(shape1.prod() * num_byte_per_elem / replica)

            if opti_intra_mapping:
                # Optimised mapping: all cores in the group access a
                # single contiguous region -- granularity equals the
                # entire per-core block times the group size.
                dram_access_granularity_byte = tot_access_bytes * core_group_size
            else:
                # Standard mapping: granularity is based on the tensor's
                # largest dimension, adjusted for temporal reuse and the
                # core-group cooperative factor.
                if bad_mapping and i:
                    # Sub-optimal: input tensors divide last dim by cpb.
                    last_dim = max(1, shape1[-1] // self._cpb)
                else:
                    last_dim = max(shape1)

                remaining_dims = shape.prod() / last_dim
                # Compute how much the replica factor shrinks the
                # contiguous last-dimension chunk.
                last_dim_div = 2 * replica / remaining_dims
                last_dim_div = max(1, last_dim_div)
                last_dim = last_dim // last_dim_div
                # Scale by core-group: cooperating cores access
                # adjacent elements, widening the contiguous region.
                last_dim = last_dim * core_group_size
                dram_access_granularity_byte = last_dim * num_byte_per_elem
                # Ensure at least one element width.
                dram_access_granularity_byte = max(num_byte_per_elem, dram_access_granularity_byte)
            if return_granularity:
                dram_access_list.append(dram_access_granularity_byte)
            elif return_tot_bytes_per_core:
                dram_access_list.append(tot_access_bytes)
            elif return_cycles_and_bytes:
                need_init = not opti_intra_mapping
                cycle_fn = (
                    self.num_cycle_of_access_for_tiling
                    if for_tiling else self.num_cycle_of_access
                )
                dram_access_cycles = cycle_fn(tot_access_bytes,
                                              dram_access_granularity_byte,
                                              need_init)
                dram_access_list.append(dram_access_cycles)
                dram_bytes_list.append(tot_access_bytes)
            elif detailed:
                need_init = not opti_intra_mapping
                cycle_fn = (
                    self.num_cycle_of_access_for_tiling
                    if for_tiling else self.num_cycle_of_access
                )
                dram_access_cycles = cycle_fn(tot_access_bytes,
                                              dram_access_granularity_byte,
                                              need_init)
                dram_access_list.append(dram_access_cycles)
                dram_bytes_list.append(tot_access_bytes)
                dram_granularity_list.append(int(dram_access_granularity_byte))
                if return_cycles_bytes_granularity_conflicts:
                    dram_row_conflicts_list.append(
                        self.num_row_conflicts_of_access(
                            tot_access_bytes,
                            dram_access_granularity_byte,
                            need_init,
                        )
                    )
            else:
                need_init = not opti_intra_mapping
                cycle_fn = (
                    self.num_cycle_of_access_for_tiling
                    if for_tiling else self.num_cycle_of_access
                )
                dram_access_cycles = cycle_fn(tot_access_bytes,
                                              dram_access_granularity_byte,
                                              need_init)
                dram_access_list.append(dram_access_cycles)
        if return_cycles_and_bytes:
            return dram_access_list, dram_bytes_list
        if return_cycles_bytes_granularity:
            return dram_access_list, dram_bytes_list, dram_granularity_list
        if return_cycles_bytes_granularity_conflicts:
            return (dram_access_list, dram_bytes_list, dram_granularity_list,
                    dram_row_conflicts_list)
        return dram_access_list
