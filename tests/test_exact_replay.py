import unittest

from tsim_components.exact_replay import replay_exact_placement
from tsim_components.system_model import Hardware


class ExactReplayTest(unittest.TestCase):
    def test_exact_block_replay_conserves_tsv_noc_and_replicas(self):
        records = [
            {
                "demand_id": "0:A:left", "tensor_id": "A", "kind": "read",
                "replica_id": 0, "portal": 0, "portal_node": 0, "bank": 0,
                "num_bytes": 512, "allocation_bytes": 256, "live_bytes": 128,
                "live_start": 0, "live_end": 2, "consumers": [0, 1, 2, 3],
                "assigned_consumers": [0, 1], "producers": [],
            },
            {
                "demand_id": "0:A:left", "tensor_id": "A", "kind": "read",
                "replica_id": 1, "portal": 1, "portal_node": 3, "bank": 1,
                "num_bytes": 512, "allocation_bytes": 256, "live_bytes": 128,
                "live_start": 0, "live_end": 2, "consumers": [0, 1, 2, 3],
                "assigned_consumers": [2, 3], "producers": [],
            },
            {
                "demand_id": "0:C:owner", "tensor_id": "C", "kind": "write",
                "replica_id": 0, "portal": 1, "portal_node": 3, "bank": 3,
                "num_bytes": 256, "allocation_bytes": 256, "live_bytes": 128,
                "live_start": 1, "live_end": 3, "consumers": [0],
                "assigned_consumers": [0], "producers": [0, 1, 2, 3],
            },
        ]
        hardware = Hardware(
            cores=4, mesh_width=2, banks=4, channels=2,
            row_bytes=256, bank_capacity=1 << 20, sram_bytes=4096,
            channel_bytepc=64.0, noc_bytepc=16.0,
            cl=2, rcd=3, rp=3, dram_burst_bytes=128,
            noc_packet_bytes=128, channel_nodes=(0, 3),
        )
        result = replay_exact_placement(
            records, hardware, compute_cores=(0, 1, 2, 3),
            compute_cycles=20, scratch_bytes_per_core=256,
            source_policy="joint_minimax", trace=True,
        )
        replay = result["exact_replay"]
        self.assertTrue(replay["traffic_invariants_passed"])
        self.assertEqual(replay["channel_load_bytes"], [512, 768])
        self.assertEqual(replay["critical_tsv_bytes"], 768)
        self.assertEqual(sum(result["metrics"]["channel_bytes"].values()), 1280)
        self.assertGreater(result["metrics"]["cycles"], 0)

    def test_wrong_bank_channel_mapping_is_rejected(self):
        record = {
            "demand_id": "d", "tensor_id": "A", "kind": "read",
            "replica_id": 0, "portal": 1, "portal_node": 1, "bank": 0,
            "num_bytes": 128, "allocation_bytes": 128, "live_bytes": 128,
            "live_start": 0, "live_end": 1, "consumers": [1],
            "assigned_consumers": [1], "producers": [],
        }
        hardware = Hardware(
            cores=2, mesh_width=2, banks=2, channels=2, row_bytes=128,
            bank_capacity=1024, sram_bytes=1024, channel_nodes=(0, 1),
        )
        with self.assertRaisesRegex(ValueError, "channel declaration|bank is not attached"):
            replay_exact_placement(
                [record], hardware, compute_cores=(0, 1), compute_cycles=1,
                scratch_bytes_per_core=128, source_policy="local_single",
            )


if __name__ == "__main__":
    unittest.main()
