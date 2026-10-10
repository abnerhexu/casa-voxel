"""Counterfactual zero-duration stores retain physical traffic and energy."""
import pytest
from tsim_components.tsim_analysis_lib import FusedOperatorExecLog


def make_log(mode, write_bytes=200, write_cycles=0, finish=4):
    energy = (16640, 0, 0, dict(sa=0, vu=0, noc=0, sram=0, dram=16640, tsv=0))
    return FusedOperatorExecLog(
        0, 1, 2, 3, 4, 0, finish, 0,
        (100, write_bytes), (1, write_cycles), (1, 1, 0, 0, 0), (1, 0, 1),
        energy, (1, 1, 0, 0), (0, 0), 128, 1500,
        spatial_meta={"dram_row_buffer": {"counterfactual": mode}},
    )


@pytest.mark.parametrize("mode", ["ideal_memory", "ideal_dram"])
def test_idealized_store_keeps_bytes_energy_and_finish(mode):
    log = make_log(mode)
    assert log.dram_w_bytes == 200
    assert log.dram_st_dur == 0
    assert log.energy_dram == 16640
    assert log.t_finish == log.t_reduce_start + log.reduce_dur
    log.t_finish += 1
    with pytest.raises(AssertionError, match="Finish time"):
        log.sanity_check()


@pytest.mark.parametrize("mode", ["none", "ideal_dram_noc", "ideal_row_switch", "unknown"])
def test_nonidealized_store_still_rejected(mode):
    with pytest.raises(AssertionError, match="write bytes"):
        make_log(mode)
    assert make_log(mode, write_bytes=0).dram_w_bytes == 0
    assert make_log(mode, write_cycles=1, finish=5).dram_st_dur == 1
