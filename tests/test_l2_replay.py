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
