import unittest
from types import SimpleNamespace
from unittest.mock import patch

from tsim_components.noc import NoC, Topo
from tsim_components.mem import DRAM
from tsim_simple import run_tsim
from web_profiler.server.parsers import parse_operators_text, parse_summary_text


class NoCTrafficHopsTest(unittest.TestCase):
    def setUp(self):
        self.noc = NoC(
            bandwidth_bytepc=16,
            topology=Topo.MESH,
            nodes=list(range(4)),
        )

    def test_byte_hops_match_default_mapping_math(self):
        tensor_sizes = [100, 200]
        temporal_replicas = [2, 4]
        spatial_replicas = [2, 4]
        shift_info = (
            1,
            [],
            [],
            [],
            [[(10, 2, 3), (5, 1, 3)]],
        )

        old_cycles = self.noc.get_total_cycles_from_expression(
            tensor_sizes,
            temporal_replicas,
            spatial_replicas,
            shift_info,
            num_bytes_per_elem=2,
        )
        cycles, byte_hops = self.noc.get_total_cycles_and_traffic_hops_from_expression(
            tensor_sizes,
            temporal_replicas,
            spatial_replicas,
            shift_info,
            num_bytes_per_elem=2,
        )

        self.assertEqual(cycles, old_cycles)
        # Output reduce: (100 / 2) elements * 1 transfer * 2 B * 1 hop.
        self.assertEqual(byte_hops[2], 100.0)
        # Input broadcast: (200 / 4) elements * 3 transfers * 2 B * 1 hop.
        self.assertEqual(byte_hops[0], 300.0)
        # Shift transfers use the same greedy hop state as cycle estimation:
        # 10 * (2*3) * 2 B * 2 hops + 5 * (1*3) * 2 B * 4 hops.
        self.assertEqual(byte_hops[1], 360.0)
        self.assertEqual(sum(byte_hops), 760.0)

    def test_dimension_ordered_mesh_path_is_x_then_y(self):
        self.assertEqual(
            self.noc.get_dimension_ordered_path(0, 3),
            [0, 1, 3],
        )
        self.assertEqual(
            self.noc.get_dimension_ordered_path(3, 0),
            [3, 2, 0],
        )

    def test_spmd_reclassifies_cycles_not_traffic(self):
        args = (
            [100, 200],
            [2, 4],
            [2, 4],
            (1, [], [], [], [[(10, 2, 3)]]),
        )
        normal_cycles, normal_byte_hops = \
            self.noc.get_total_cycles_and_traffic_hops_from_expression(*args)
        spmd_cycles, spmd_byte_hops = \
            self.noc.get_total_cycles_and_traffic_hops_from_expression(
                *args, spmd_compiler=True
            )

        self.assertEqual(spmd_byte_hops, normal_byte_hops)
        self.assertEqual(spmd_cycles[0], normal_cycles[0] + normal_cycles[1])
        self.assertEqual(spmd_cycles[1], 0)
        self.assertEqual(spmd_cycles[2], normal_cycles[2] + normal_cycles[1])

    def test_no_replication_has_zero_hop_weighted_traffic(self):
        _, byte_hops = self.noc.get_total_cycles_and_traffic_hops_from_expression(
            [64, 128],
            [1, 1],
            [1, 1],
            (0, [], [], [], []),
        )
        self.assertEqual(byte_hops, (0.0, 0.0, 0.0))

    def test_new_metric_is_parsed_from_summary_and_operator_logs(self):
        summary_text = "\n".join([
            "EXE time (total, fused): 100,Energy (mJ): 1, Static: 0.4 mJ, Dyn.: 0.6 mJ",
            "Static Energy: SA = 0 mJ",
            "Static Energy: DRAM = 0 mJ",
            "Dynamic Energy: SA = 0 mJ",
            "Dynamic Energy: DRAM = 0 mJ",
            "Power (w): 1, Static: 0 W (dram: 0 logic: 0), Dyn.: 1 W",
            "Overall Util: 0.5",
            "DRAM UTIL (%): 1/2 (R/W), SA_UTIL:0.1, VU_UTIL:0.2, NOC: 0.3",
            "NoC traffic x hops (byte-hop): Total=760, Broadcast=300, Shift=360, Reduce=100",
            "FLOPS: 1 GFLOPS MM, 2 GFLOPS VU",
        ])
        summary = parse_summary_text(summary_text)
        self.assertEqual(summary["noc_byte_hops"], 760.0)
        self.assertEqual(summary["noc_bcast_byte_hops"], 300.0)
        self.assertEqual(summary["noc_shift_byte_hops"], 360.0)
        self.assertEqual(summary["noc_reduce_byte_hops"], 100.0)
        self.assertEqual(summary["mm_gflops"], 1.0)

        operator_text = "\n".join([
            "Operator 0===================",
            "START TIMES:  Ld: 0 -> broadcast 1 -> comp/sh 2 -> reduce 3 -> fin 4",
            "DURATIONS:    Ld: 1 -> broadcast 1 -> comp/sh 1/1 -> reduce 1",
            "Compute Utilization: 0.5 VU Utilization: 0.25",
            "Write bytes: 16 Read bytes: 32",
            "NoC traffic x hops (byte-hop): Total=760, Broadcast=300, Shift=360, Reduce=100",
            "Average Power (W): 2.5",
        ])
        operators = parse_operators_text(operator_text)
        self.assertEqual(len(operators), 1)
        self.assertEqual(operators[0]["noc_byte_hops"], 760.0)
        self.assertEqual(operators[0]["avg_power_w"], 2.5)

    def test_run_tsim_exposes_aggregate_and_per_operator_metrics(self):
        class FakeExpr:
            def get_sub_op_var_sizes(self, temporal, spatial, return_size=True):
                return [100, 200]

            def get_temporal_var_replicas(self, temporal, spatial):
                return [2, 4]

            def get_spatial_var_replicas(self, temporal, spatial):
                return [2, 4]

            def get_shift_info(self, temporal, spatial, output_special_format_for_tsim=False):
                return (1, [], [], [], [[(10, 2, 3), (5, 1, 3)]])

        op = SimpleNamespace(expr=FakeExpr(), num_byte_per_elem=2)
        op_log = SimpleNamespace()

        class FakeProgram:
            ops = [op]
            tot_num_cores = 4

            def get_best_config_by_max_mem_size(self, *args):
                return [2, 2], [[1, 1], [1, 1]]

            def get_fused_exec_time(self, **kwargs):
                stats = {
                    "exec_time": 100,
                    "exec_energy": 0,
                    "comp_energy": 0,
                    "sss_energy": 0,
                    "sa_energy": 0,
                    "vu_energy": 0,
                    "noc_energy": 0,
                    "sram_energy": 0,
                    "dram_energy": 0,
                    "tsv_energy": 0,
                    "dram_r_util": 0,
                    "dram_w_util": 0,
                    "sa_flops": 0,
                    "vu_flops": 0,
                }
                return stats, [op_log]

            def get_fused_dram_bytes_only(self, **kwargs):
                return [(0, 0)]

        compute = SimpleNamespace(mm_pad_shape=[1, 1, 1], ew_pad_len=1)

        class FakeComputeOp:
            comp = compute

            def fuse_ops(self, ops, *args, **kwargs):
                return [[item] for item in ops]

            def compute_costs(self, *args, **kwargs):
                return [(1, 1, 0, 0, 0)]

        dram = DRAM(
            CL=14,
            tRCD=14,
            tRP=14,
            bytes_per_row=1024,
            bytes_per_cycle=16,
            num_cores=4,
        )
        with patch("tsim_simple.get_overlap", return_value=[]):
            _, stats, logs, _ = run_tsim(
                prog=FakeProgram(),
                total_sram_byte_per_core=4096,
                exe_space=2048,
                dram=dram,
                noc=self.noc,
                comp_op=FakeComputeOp(),
                core_group_size=1,
                dram_bandwidth_GBps=128,
                npu_freq_MHz=1500,
                tot_layers=1,
                sim_layers=0,
                aggregate_scale=4,
            )

        self.assertEqual(stats["noc_byte_hops"], 3040.0)
        self.assertEqual(stats["noc_bcast_byte_hops"], 1200.0)
        self.assertEqual(stats["noc_shift_byte_hops"], 1440.0)
        self.assertEqual(stats["noc_reduce_byte_hops"], 400.0)
        self.assertEqual(stats["operator_noc_byte_hops"], 3040.0)
        self.assertEqual(stats["dram_noc_byte_hops"], 0.0)
        self.assertEqual(stats["noc_energy"], 3040.0 * 2.0)
        self.assertEqual(logs[0].energy_noc, 760.0 * 2.0)
        self.assertEqual(logs[0].noc_byte_hops, 760.0)
        self.assertEqual(stats["experiment"]["metrics"]["noc_byte_hops"]["total"], 3040.0)
        self.assertEqual(stats["experiment"]["workload"]["total_layers"], 4)
        self.assertEqual(stats["experiment"]["workload"]["simulated_layers"], 1)
        self.assertEqual(stats["experiment"]["workload"]["extrapolation_factor"], 4.0)
        self.assertEqual(len(stats["experiment"]["tiling"]), 1)


if __name__ == "__main__":
    unittest.main()
