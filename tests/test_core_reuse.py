import copy
from dataclasses import replace
import numpy as np
import pytest
from tsim_components.comp import Compute
from tsim_components.aggregate_trace import summary
from tsim_components.core_reuse import lower_program, intermediate_residents, traffic_summary
from tsim_components.system_model import Hardware, execute
from tsim_components.tile_trace import export_program


def compute(**kwargs):
    cfg = dict(ew_pad_len=4, mm_pad_shape=[4]*3, ew_reuse_num=4,
               mm_reuse_list=[4, 4, 4], ew_flopc=4, mm_flopc=32,
               load_store_bw_bytepc=8, byte_per_elem=2, mm_init_cycle=4)
    return Compute(**dict(cfg, **kwargs))


def spec():
    return dict(index=0, name="mm", op_type=5, dims=[16]*3,
                variables=[[0,1],[0,2],[2,1]], tensors=["out","a","b"],
                spatial=[2,2,1], temporal=[[2]*3,[1]*3,[1]*3],
                element_bytes=2, source_hot_bytes=1)


def hardware():
    return Hardware(cores=4, mesh_width=2, banks=4, channels=4, row_bytes=128,
                    sram_bytes=2**20, channel_bytepc=16, noc_bytepc=2)


def test_input_a_policy_and_legacy_default():
    low = compute(mm_input_a_reuse_policy="configured")
    high = compute(mm_input_a_reuse_policy="configured", mm_reuse_list=[4,16,4])
    assert low.get_mm_load_store_cycles_from_padded_bkmn((1,16,16,16))[1] == 4*high.get_mm_load_store_cycles_from_padded_bkmn((1,16,16,16))[1]
    np.testing.assert_equal(compute().get_mm_load_store_cycles_from_padded_bkmn((1,16,16,16)), high.get_mm_load_store_cycles_from_padded_bkmn((1,16,16,16)))
    assert summary(spec(), low)["local_sram_read_bytes"] > summary(spec(), high)["local_sram_read_bytes"]
    legacy = compute()
    del legacy.mm_input_a_reuse_policy  # Attribute absent in old pickles.
    np.testing.assert_equal(legacy.get_mm_load_store_cycles_from_padded_bkmn((1,16,16,16)),
                            compute().get_mm_load_store_cycles_from_padded_bkmn((1,16,16,16)))


@pytest.mark.parametrize("backend", ["aggregate", "affine"])
def test_paired_counterfactuals_conserve_traffic_and_dependencies(backend):
    hw, c = hardware(), compute()
    g = (lower_program([spec()],c,hw,paradigm="compute_shift") if backend == "aggregate"
         else export_program([spec()],c,hw,paradigm="compute_shift",stages=1))
    mapping = {t:list(range(hw.banks)) for t in g["tensors"]}
    original = copy.deepcopy(g)
    rows = [execute(g,hw,mapping,noc_ideal=ideal)["metrics"] for ideal in ("none","reuse","all")]
    assert rows[0]["traffic"] == rows[1]["traffic"] == rows[2]["traffic"]
    assert rows[0]["link_bytes"] == rows[2]["link_bytes"]
    assert rows[2]["cycles"] <= rows[0]["cycles"]
    assert execute(g,hw,mapping,overlap=False)["metrics"]["cycles"] >= rows[0]["cycles"]
    assert g == original


def test_only_active_cores_divide_work_and_residency_is_capacity_checked():
    s, hw = spec(), hardware()
    s["spatial"] = [1]*3
    g = lower_program([s],compute(),hw,paradigm="compute_shift")
    assert g["jobs"][0]["cores"] == [0]
    assert g["jobs"][0]["compute_cycles"] == summary(s,compute())["total_core_cycles"]
    second = dict(spec(), index=1, tensors=["last","out","b"])
    g = lower_program([spec(),second],compute(),hw,paradigm="dataflow",instances=2)
    assert set(g["jobs"][0]["cores"]).isdisjoint(g["jobs"][1]["cores"])
    assert intermediate_residents(g, replace(hw,sram_bytes=1)) == {}
    resident = intermediate_residents(g,hw)
    assert resident
    mapping = {t:list(range(hw.banks)) for t in g["tensors"]}
    measured = execute(g,hw,mapping,paradigm="dataflow",resident=resident)["metrics"]
    no_reuse = execute(g,hw,mapping,paradigm="dataflow")["metrics"]
    assert traffic_summary(measured)["vertical_bytes"] < traffic_summary(no_reuse)["vertical_bytes"]
