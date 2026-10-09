import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from tsim_components.experiment_output import (
    EXPERIMENT_SCHEMA_VERSION,
    build_experiment_record,
    write_experiment_record_files,
    write_experiment_records_jsonl,
)
from tsim_components.mem import DRAM


class ExperimentOutputTest(unittest.TestCase):
    def make_inputs(self, placement="channel_aware"):
        dram = DRAM(
            CL=14, tRCD=14, tRP=14,
            bytes_per_row=8192, bytes_per_cycle=32,
            num_cores=256, num_layers=8, banks_per_layer=16,
            num_channels=8, transaction_bytes=128,
            capacity_bytes=192 * 1024**3,
        )
        hardware = SimpleNamespace(
            mem=dram,
            comp=SimpleNamespace(mm_pad_shape=[32, 32], ew_pad_len=32),
            noc=SimpleNamespace(
                topology=SimpleNamespace(value="mesh"),
                bandwidth_bytepc=16,
            ),
            num_cores=256,
            core_grp_size=8,
            sram_size=2 * 1024**2,
            exe_sram_size=1024**2,
            npu_freq_mHz=1600,
            dram_bw_GBps=12 * 1024,
        )
        stats = {
            "exec_time": 1600,
            "exec_energy": 50_000,
            "dram_r_bytes": 1000,
            "dram_w_bytes": 200,
            "dram_r_row_hits": 4,
            "dram_w_row_hits": 1,
            "dram_r_row_misses": 3,
            "dram_w_row_misses": 2,
            "dram_r_row_conflicts": 7,
            "dram_w_row_conflicts": 5,
            "dram_base_energy": 8400,
            "dram_row_conflict_energy": 87_240,
            "dram_energy": 95_640,
            "noc_bcast_byte_hops": 10,
            "noc_shift_byte_hops": 20,
            "noc_reduce_byte_hops": 30,
            "noc_byte_hops": 60,
        }
        log = SimpleNamespace(
            op_id=0,
            t_dram_ld_start=0,
            t_finish=1600,
            dram_ld_dur=100,
            dram_st_dur=20,
            bcast_dur=30,
            shift_dur=40,
            reduce_dur=10,
            comp_dur=80,
            dram_r_bytes=1000,
            dram_w_bytes=200,
            dram_r_row_hits=4,
            dram_w_row_hits=1,
            dram_r_row_misses=3,
            dram_w_row_misses=2,
            dram_r_row_conflicts=7,
            dram_w_row_conflicts=5,
            energy_dram_base=8400,
            energy_dram_row_conflict=87_240,
            energy_dram=95_640,
            noc_bcast_byte_hops=10,
            noc_shift_byte_hops=20,
            noc_reduce_byte_hops=30,
            noc_byte_hops=60,
            dram_placement_policy=placement,
            spatial_meta={
                "dram_access_records": [{
                    "tensor_id": 42, "stage": "read", "address": 4096,
                    "allocation_bytes": 8192, "total_bytes": 1024,
                    "bank_ids": [0, 8], "channel_ids": [0],
                }],
            },
        )
        return hardware, stats, [log]

    def build_record(self, placement="channel_aware"):
        hardware, stats, logs = self.make_inputs(placement)
        return build_experiment_record(
            workload_name="qwen3.5-9b-prefill-b16-s4096",
            hardware=hardware,
            stats=stats,
            operator_logs=logs,
            partitions=[(((1, 1),), (8, 32))],
            operator_names=["matmul_0"],
            placement_policy=placement,
            total_layers=32,
            simulated_layers=32,
            tiling_cache={"hits": 1, "misses": 1, "entries": 1},
        )

    def test_record_contains_l1_metrics_and_architecture(self):
        record = self.build_record()

        self.assertEqual(record["schema_version"], EXPERIMENT_SCHEMA_VERSION)
        self.assertEqual(record["architecture"]["dram"]["total_banks"], 128)
        self.assertEqual(record["architecture"]["dram"]["num_channels"], 8)
        self.assertEqual(record["architecture"]["dram"]["capacity_bytes"], 192 * 1024**3)
        self.assertEqual(record["architecture"]["noc"]["link_bandwidth_bytes_per_cycle"], 16)
        self.assertEqual(record["metrics"]["end_to_end_time_ms"], 0.001)
        self.assertEqual(record["metrics"]["row_buffer"]["total"]["conflicts"], 12)
        self.assertEqual(record["metrics"]["dram_dynamic_energy_pj"]["total"], 95_640)
        self.assertEqual(record["operators"][0]["placement"]["bank_ids"], [0, 8])
        self.assertEqual(
            record["operators"][0]["placement"]["tensor_accesses"][0]["address"],
            4096,
        )
        self.assertEqual(record["tiling"][0]["spatial"], [8, 32])

    def test_configuration_id_changes_with_placement(self):
        self.assertNotEqual(
            self.build_record("software_aware")["configuration_id"],
            self.build_record("channel_aware")["configuration_id"],
        )

    def test_jsonl_writer_preserves_every_design_point(self):
        records = [self.build_record("uniform"), self.build_record("channel_aware")]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = write_experiment_records_jsonl(
                Path(tmpdir) / "experiment_results.jsonl", records
            )
            loaded = [json.loads(line) for line in path.read_text().splitlines()]

        self.assertEqual(len(loaded), 2)
        self.assertEqual(
            [item["architecture"]["dram"]["placement_policy"] for item in loaded],
            ["uniform", "channel_aware"],
        )

    def test_per_configuration_files_do_not_collide_across_sweeps(self):
        records = [self.build_record("uniform"), self.build_record("channel_aware")]
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = write_experiment_record_files(tmpdir, records)

            self.assertEqual(len(paths), 2)
            self.assertEqual(len({path.name for path in paths}), 2)
            self.assertTrue(all(path.exists() for path in paths))


if __name__ == "__main__":
    unittest.main()
