import unittest

import numpy as np

from icbm_DNNProgram import (
    DRAM_ACT_PRE_ENERGY_PJ,
    DNNProgram,
    STACKED_3D_DRAM_PJ_PER_BYTE,
)
from tsim_components.mem import DRAM
from tsim_components.tsim_analysis_lib import FusedOperatorExecLog
from web_profiler.server.parsers import parse_operators_text, parse_summary_text


class DRAMRowConflictEnergyTest(unittest.TestCase):
    def make_dram(self, *, banks=4, use_sram=False):
        return DRAM(
            CL=14,
            tRCD=14,
            tRP=14,
            bytes_per_row=256,
            bytes_per_cycle=16,
            num_cores=4,
            num_banks_per_channel=banks,
            transaction_bytes=16,
            use_sram=use_sram,
        )

    def test_conflicts_begin_when_a_bank_open_row_is_replaced(self):
        dram = self.make_dram(banks=4)

        self.assertEqual(dram.num_row_conflicts_of_access(4 * 256, 256), 0)
        self.assertEqual(dram.num_row_conflicts_of_access(5 * 256, 256), 1)
        self.assertEqual(dram.num_row_conflicts_of_access(20 * 64, 64), 16)

    def test_sram_mode_has_no_dram_row_conflicts(self):
        dram = self.make_dram(banks=1, use_sram=True)
        self.assertEqual(dram.num_row_conflicts_of_access(4096, 64), 0)

    def test_detailed_access_result_includes_conflict_count(self):
        dram = self.make_dram(banks=1)
        cycles, byte_counts, granularities, conflicts = dram.get_dram_access_list(
            [np.array([1024])],
            [1],
            core_group_size=1,
            num_byte_per_elem=1,
            return_cycles_bytes_granularity_conflicts=True,
        )

        self.assertEqual(len(cycles), 1)
        self.assertEqual(byte_counts, [1024])
        # The reported tensor granularity can exceed a physical row; the
        # conflict estimator clamps it to bytes_per_row internally.
        self.assertEqual(granularities, [512])
        self.assertEqual(conflicts, [3])

    def test_worse_mapping_produces_more_conflicts_at_equal_bytes(self):
        dram = DRAM(
            CL=14, tRCD=14, tRP=14,
            bytes_per_row=256, bytes_per_cycle=16,
            num_cores=256, num_banks_per_channel=4,
            transaction_bytes=16,
        )
        tensors = [np.array([64, 64]), np.array([64, 64])]
        common = dict(
            tensor_shapes=tensors,
            temporal_var_replicas=[1, 1],
            core_group_size=1,
            num_byte_per_elem=2,
            return_cycles_bytes_granularity_conflicts=True,
        )
        _, good_bytes, _, good_conflicts = dram.get_dram_access_list(
            **common, bad_mapping=False
        )
        _, bad_bytes, _, bad_conflicts = dram.get_dram_access_list(
            **common, bad_mapping=True
        )

        self.assertEqual(good_bytes[1], bad_bytes[1])
        self.assertGreater(bad_conflicts[1], good_conflicts[1])

    def test_energy_adds_7_27_nj_per_conflict(self):
        _, _, _, components = DNNProgram.get_fused_op_energy_from_scratch(
            object(),
            [],
            iter(()),
            dram_r_traffic=100,
            dram_w_traffic=200,
            dram_row_conflicts=2,
        )

        expected_base = 300 * STACKED_3D_DRAM_PJ_PER_BYTE
        expected_conflict = 2 * DRAM_ACT_PRE_ENERGY_PJ
        self.assertEqual(components["dram"], expected_base + expected_conflict)
        self.assertEqual(expected_conflict, 14_540)

    def test_operator_log_retains_energy_breakdown_and_counts(self):
        energy = (
            16_640,
            0,
            0,
            {"sa": 0, "vu": 0, "noc": 0, "sram": 0,
             "dram": 16_640, "tsv": 0},
        )
        op = FusedOperatorExecLog(
            0, 1, 2, 3, 4, 0, 5, 0,
            (100, 200), (1, 1), (1, 1, 0, 0, 0), (1, 0, 1),
            energy, (1, 1, 0, 0), (0, 0), 128, 1500,
            spatial_meta={
                "dram_row_conflicts": {"read": 1, "write": 1, "total": 2},
                "dram_energy_breakdown_pj": {
                    "base_transfer": 2_100,
                    "row_conflict": 14_540,
                },
            },
        )

        self.assertEqual(op.dram_row_conflicts, 2)
        self.assertEqual(op.energy_dram_base, 2_100)
        self.assertEqual(op.energy_dram_row_conflict, 14_540)
        self.assertGreater(op.dram_row_conflict_dynamic_power_W, 0)

    def test_summary_and_operator_parsers_expose_conflict_metrics(self):
        summary = parse_summary_text("\n".join([
            "EXE time (total, fused): 100,Energy (mJ): 1, Static: 0.4 mJ, Dyn.: 0.6 mJ",
            "Static Energy: SA = 0 mJ",
            "Static Energy: DRAM = 0 mJ",
            "Dynamic Energy: SA = 0 mJ",
            "Dynamic Energy: DRAM = 0.3 mJ, TSV = 0 mJ",
            "DRAM row conflicts (ACT+PRE): Total=30, Read=20, Write=10, EnergyPerConflict=7.27 nJ",
            "DRAM dynamic energy breakdown: Base=0.0819 mJ, RowConflict=0.2181 mJ, Total=0.3 mJ",
            "DRAM dynamic power (workload average): Base=0.819 W, RowConflict=2.181 W, Total=3 W",
            "Power (w): 1, Static: 0 W (dram: 0 logic: 0), Dyn.: 1 W",
            "Overall Util: 0.5",
            "DRAM UTIL (%): 1/2 (R/W), SA_UTIL:0.1, VU_UTIL:0.2, NOC: 0.3",
        ]))
        self.assertEqual(summary["dram_row_conflicts"], 30.0)
        self.assertEqual(summary["dram_r_row_conflicts"], 20.0)
        self.assertEqual(summary["dram_w_row_conflicts"], 10.0)
        self.assertEqual(summary["dynamic_dram_base_mj"], 0.0819)
        self.assertEqual(summary["dynamic_dram_row_conflict_mj"], 0.2181)
        self.assertEqual(summary["dynamic_power_dram_row_conflict_w"], 2.181)
        self.assertEqual(summary["dynamic_power_dram_w"], 3.0)
        self.assertEqual(summary["dynamic_power_w"], 1.0)

        operators = parse_operators_text("\n".join([
            "Operator 0===================",
            "START TIMES:  Ld: 0 -> broadcast 1 -> comp/sh 2 -> reduce 3 -> fin 4",
            "DURATIONS:    Ld: 1 -> broadcast 1 -> comp/sh 1/1 -> reduce 1",
            "Compute Utilization: 0.5 VU Utilization: 0.25",
            "Write bytes: 16 Read bytes: 32",
            "NoC traffic x hops (byte-hop): Total=1, Broadcast=1, Shift=0, Reduce=0",
            "DRAM row conflicts (ACT+PRE): Total=3, Read=2, Write=1, Energy=21810 pJ",
            "DRAM dynamic power (W): Total=3, Base=0.819, RowConflict=2.181",
            "Average Power (W): 2.5",
        ]))
        self.assertEqual(operators[0]["dram_row_conflicts"], 3.0)
        self.assertEqual(operators[0]["energy_dram_row_conflict_pj"], 21_810.0)
        self.assertEqual(operators[0]["dram_row_conflict_dynamic_power_w"], 2.181)
        self.assertEqual(operators[0]["avg_power_w"], 2.5)


if __name__ == "__main__":
    unittest.main()
