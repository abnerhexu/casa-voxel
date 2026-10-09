import copy
import unittest
from test_dram_execution_session import make_dram, record


class FrozenReplayTest(unittest.TestCase):
    def session(self, dram, **options):
        dram.execution_options = options
        return dram.new_execution_session(placement_policy="uniform")

    def test_instrumentation_and_frozen_replay(self):
        records = [record("read", 1, 2048), record("write", 1, 256)]
        plain = self.session(make_dram(banks=4, channels=2)).schedule_records(records, 0)
        captured = self.session(make_dram(banks=4, channels=2), instrument=True, trace=True)
        measured = captured.schedule_records(records, 0)
        self.assertEqual(plain.read.cycles, measured.read.cycles)
        replay = self.session(make_dram(banks=4, channels=1), frozen=captured.snapshot,
                              instrument=True, trace=True)
        result = replay.schedule_records(records, 0)
        self.assertEqual(result.read.num_bytes, plain.read.num_bytes)
        for stats in (measured.read, measured.write, result.read):
            r = stats.resource
            self.assertEqual(r["channel_time"], r["transfer"] + r["tail_idle"] + r["unclassified_idle"])
            self.assertEqual(sum(r["channel_bytes"]), stats.num_bytes)
            for c in range(len(r["channel_bytes"])):
                events = sorted((e for e in stats.trace if e["channel_id"] == c), key=lambda e: e["transfer_start"])
                self.assertTrue(all(a["finish"] <= b["transfer_start"] for a, b in zip(events, events[1:])))
        bad = copy.deepcopy(records)
        bad[0]["total_bytes"] += 1
        with self.assertRaisesRegex(ValueError, "mismatch"):
            self.session(make_dram(banks=4, channels=2), frozen=captured.snapshot).schedule_records(bad, 0)

    def test_frozen_rejects_geometry_and_capacity(self):
        records = [record("read", 1, 2048)]
        session = self.session(make_dram())
        session.schedule_records(records, 0)
        with self.assertRaisesRegex(ValueError, "geometry"):
            self.session(make_dram(banks=2), frozen=session.snapshot).schedule_records(records, 0)
        dram = make_dram()
        dram.capacity_bytes = 100
        with self.assertRaisesRegex(ValueError, "capacity"):
            self.session(dram, frozen=session.snapshot).schedule_records(records, 0)

    def test_split_conservation_and_fixed_order_counts(self):
        records = [record("read", 1, 1025), record("read", 2, 2048), record("read", 1, 1025)]
        counts = []
        for budget in (None, 1, 4, 16):
            for speed in (8, 64):
                session = self.session(make_dram(banks=2, bytepc=speed), trace=True,
                                       row_budget=budget, fixed_bank_order=True)
                result = session.schedule_records(records, 0)
                self.assertEqual(result.read.num_bytes, sum(r["total_bytes"] for r in records))
                self.assertEqual(sum(e["num_bytes"] for e in result.read.trace), result.read.num_bytes)
                counts.append((result.read.row_hits, result.read.row_misses, result.read.row_conflicts))
        self.assertEqual(len(set(counts)), 1)

    def test_fcfs_and_age_protection(self):
        from tsim_components.dram_scheduler import DRAMRowRun
        runs = [DRAMRowRun(0, 0, 0, 0, 100, 12800)] + [
            DRAMRowRun(i, 1, 0, i, 1, 128) for i in range(1, 10)]
        fcfs = self.session(make_dram(banks=2), trace=True, policy="fcfs")
        out = fcfs._schedule_stage(runs, [{} for _ in runs])
        self.assertEqual([e["input_index"] for e in out.trace], list(range(10)))
        aged = self.session(make_dram(banks=2), trace=True, policy="age_protected", max_bypass=2)
        out = aged._schedule_stage(runs, [{} for _ in runs])
        self.assertEqual(out.trace[2]["input_index"], 0)
        self.assertEqual(out.trace[2]["selection_reason"], "age_protection")

    def test_stage_aware_requires_metadata_and_preserves_bytes(self):
        dram = make_dram(banks=4, channels=2)
        dram.execution_options = {"instrument": True}
        records = [record("read", 1, 1024), record("write", 2, 1024)]
        for r in records:
            r["execution_stage_id"] = r["stage"]
        session = dram.new_execution_session(placement_policy="stage_aware_no_distance")
        out = session.schedule_records(records, 0)
        self.assertEqual(out.read.num_bytes + out.write.num_bytes, 2048)
