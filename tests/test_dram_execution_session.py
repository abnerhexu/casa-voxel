import unittest

from tsim_components.mem import DRAM
from tsim_components.noc import NoC, Topo


def make_dram(*, banks=1, channels=1, row_bytes=256, bytepc=32, tras=0):
    return DRAM(
        CL=14,
        tRCD=14,
        tRP=14,
        tRAS=tras,
        bytes_per_row=row_bytes,
        bytes_per_cycle=bytepc,
        num_cores=1,
        num_layers=1,
        banks_per_layer=banks,
        num_channels=channels,
        transaction_bytes=128,
        soft_cores_per_bank=False,
        lock_cores_per_bank=1,
    )


def record(stage, tensor_index, size):
    return {
        "tensor_id": tensor_index,
        "subop_index": 0,
        "tensor_index": tensor_index,
        "tensor_role": "input" if stage == "read" else "output",
        "stage": stage,
        "total_bytes": size,
    }


class DRAMExecutionSessionTest(unittest.TestCase):
    def test_tras_delays_earliest_precharge(self):
        result = make_dram(tras=34).new_execution_session(
            placement_policy="uniform", replication_factor=1
        ).schedule_records([record("read", 0, 3 * 256)], op_index=0)

        self.assertEqual(result.read.row_conflicts, 2)
        self.assertEqual(result.read.stall_causing_row_conflicts, 2)
        # ACT@0, PRE@34, ACT@48, PRE@82, ACT@96, data-ready@124,
        # followed by 24 transfer cycles.
        self.assertEqual(result.read.cycles, 148)
        self.assertEqual(result.read.as_dict()[
            "stall_causing_row_conflict_ratio"
        ], 1.0)

    def test_ready_alternative_bank_prevents_conflict_stall_attribution(self):
        from tsim_components.dram_scheduler import DRAMRowRun

        session = make_dram(banks=2, channels=1).new_execution_session(
            placement_policy="uniform", replication_factor=1
        )
        session.policy = "fcfs"
        session._open_rows[0] = 0
        runs = [
            DRAMRowRun(0, 0, 0, 1, 1, 128),
            DRAMRowRun(1, 1, 0, 0, 1, 128),
        ]
        result = session._schedule_stage(runs, [{}, {}])

        self.assertEqual(result.row_conflicts, 1)
        self.assertEqual(result.stall_causing_row_conflicts, 0)

    def test_one_event_stream_provides_timing_and_row_counts(self):
        session = make_dram().new_execution_session(
            placement_policy="uniform", replication_factor=1
        )
        result = session.schedule_records(
            [record("read", 0, 1024)], op_index=0
        )

        self.assertEqual(result.read.row_misses, 1)
        self.assertEqual(result.read.row_conflicts, 3)
        self.assertEqual(result.read.num_bytes, 1024)
        self.assertEqual(result.records[0]["total_row_conflicts"], 3)
        self.assertEqual(result.records[0]["placement_policy"], "uniform")
        self.assertEqual(result.records[0]["bank_ids"], [0])
        # Four CLs, one tRCD, three PRE+tRCD pairs, and 32 data cycles.
        self.assertEqual(result.read.cycles, 4 * 14 + 14 + 3 * 28 + 32)

    def test_open_rows_persist_between_fused_operators(self):
        session = make_dram().new_execution_session(
            placement_policy="address_trace", replication_factor=1
        )
        first = session.schedule_records([record("read", 0, 128)], op_index=0)
        second = session.schedule_records([record("read", 0, 128)], op_index=1)

        self.assertEqual(first.read.row_misses, 1)
        self.assertEqual(first.read.row_conflicts, 0)
        self.assertEqual(second.read.row_misses, 0)
        self.assertEqual(second.read.row_hits, 1)
        self.assertEqual(second.read.row_conflicts, 0)
        self.assertEqual(first.records[0]["address"], second.records[0]["address"])
        self.assertEqual(first.records[0]["bank_ids"], second.records[0]["bank_ids"])
        self.assertEqual(first.read.cycles - second.read.cycles, 14)

    def test_read_and_write_share_the_same_bank_state(self):
        session = make_dram().new_execution_session(
            placement_policy="uniform", replication_factor=1
        )
        result = session.schedule_records(
            [record("read", 0, 128), record("write", 0, 128)], op_index=0
        )

        self.assertEqual(result.read.row_misses, 1)
        self.assertEqual(result.write.row_hits, 1)
        self.assertEqual(result.write.row_conflicts, 0)
        self.assertEqual(result.records[1]["total_row_conflicts"], 0)

    def test_global_prepare_allocates_distinct_non_overlapping_tensors(self):
        session = make_dram(banks=4, channels=1).new_execution_session(
            placement_policy="address_trace", replication_factor=1
        )
        records = [record("read", 10, 256), record("read", 11, 512)]
        for item in records:
            item["allocation_bytes"] = item["total_bytes"]
        session.prepare_records(records)

        first = session.schedule_records([records[0]], op_index=0).records[0]
        second = session.schedule_records([records[1]], op_index=1).records[0]
        self.assertNotEqual(first["address"], second["address"])
        self.assertGreaterEqual(second["address"], first["address"] + 256)

    def test_tiling_granularity_changes_row_conflicts_at_equal_bytes(self):
        coarse_record = record("read", 0, 1024)
        coarse_record["access_granularity_bytes"] = 256
        fine_record = record("read", 0, 1024)
        fine_record["access_granularity_bytes"] = 128

        coarse = make_dram().new_execution_session("uniform").schedule_records(
            [coarse_record], op_index=0
        )
        fine = make_dram().new_execution_session("uniform").schedule_records(
            [fine_record], op_index=0
        )

        self.assertEqual(coarse.read.num_bytes, fine.read.num_bytes)
        self.assertEqual(coarse.read.row_conflicts, 3)
        self.assertEqual(fine.read.row_conflicts, 7)
        self.assertGreater(fine.read.cycles, coarse.read.cycles)

    def test_row_runs_are_bounded_by_records_times_banks(self):
        dram = make_dram(banks=8, channels=4, row_bytes=1024)
        session = dram.new_execution_session(
            placement_policy="channel_aware", replication_factor=256
        )
        records = [record("read", 0, 16 * 1024**3)]
        runs, _plan = session._row_runs(records, op_index=0)

        self.assertLessEqual(len(runs), dram.geometry.total_banks)
        self.assertEqual(sum(run.num_bytes for run in runs), 16 * 1024**3)

    def test_noc_aware_session_records_selected_channels(self):
        noc = NoC(16, Topo.MESH, list(range(8)))
        dram = make_dram(banks=4, channels=2)
        session = dram.new_execution_session(
            placement_policy="noc_aware",
            replication_factor=1,
            noc=noc,
        )
        left = record("read", 0, 128)
        left["requester_core_weights"] = [[0, 128]]
        right = record("read", 1, 128)
        right["requester_core_weights"] = [[7, 128]]
        session.prepare_records([left, right])
        result = session.schedule_records([right], op_index=0)

        self.assertEqual(result.records[0]["placement_policy"], "noc_aware")
        self.assertEqual(result.records[0]["channel_ids"], [1])
        self.assertEqual(set(result.records[0]["bank_ids"]), {1, 3})
        self.assertEqual(result.records[0]["channel_noc_nodes"], [5])
        self.assertEqual(result.read.noc_byte_hops, 256.0)
        self.assertEqual(result.read.noc_max_hops, 2)
        self.assertEqual(result.read.noc_cycles, 30)

    def test_dram_noc_distance_changes_byte_hops_and_latency(self):
        noc = NoC(16, Topo.MESH, list(range(8)))

        def schedule_for(core):
            dram = make_dram(banks=2, channels=2, bytepc=1024)
            session = dram.new_execution_session(
                placement_policy="address_trace",
                replication_factor=1,
                noc=noc,
            )
            access = record("read", core, 128)
            access["address"] = 0
            access["allocation_bytes"] = 128
            access["requester_core_weights"] = [[core, 128]]
            return session.schedule_records([access], op_index=0)

        near = schedule_for(0)
        far = schedule_for(7)
        self.assertEqual(near.read.noc_byte_hops, 128.0)
        self.assertEqual(far.read.noc_byte_hops, 384.0)
        self.assertEqual(near.read.noc_cycles, 29)
        self.assertEqual(far.read.noc_cycles, 31)
        self.assertLess(near.read.noc_cycles, far.read.noc_cycles)
        self.assertLess(near.read.cycles, far.read.cycles)


if __name__ == "__main__":
    unittest.main()
