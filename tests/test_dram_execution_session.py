import unittest

from tsim_components.mem import DRAM


def make_dram(*, banks=1, channels=1, row_bytes=256, bytepc=32):
    return DRAM(
        CL=14,
        tRCD=14,
        tRP=14,
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


if __name__ == "__main__":
    unittest.main()
