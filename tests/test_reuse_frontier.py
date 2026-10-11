import unittest

from tsim_components.mem import DRAM
from tsim_components.noc import NoC, Topo
from tsim_components.reuse_frontier import ReuseChoice, replay_reuse_plan


class FakeLog:
    def __init__(self):
        self.bcast_dur = 40
        self.shift_dur = 80
        self.reduce_dur = 60
        self.comp_dur = 100
        self.noc_bcast_byte_hops = 400
        self.noc_shift_byte_hops = 800
        self.noc_reduce_byte_hops = 600
        self.spatial_meta = {
            "dram_access_records": [
                {
                    "version": 1,
                    "source": "test",
                    "subop_index": 0,
                    "tensor_index": 1,
                    "tensor_role": "input",
                    "tensor_id": "input",
                    "stage": "read",
                    "allocation_bytes": 4096,
                    "bytes_per_core": 128,
                    "total_bytes": 4096,
                    "requester_core_weights": [[i, 128] for i in range(32)],
                    "access_granularity_bytes": 128,
                },
                {
                    "version": 1,
                    "source": "test",
                    "subop_index": 0,
                    "tensor_index": 0,
                    "tensor_role": "output",
                    "tensor_id": "output",
                    "stage": "write",
                    "allocation_bytes": 4096,
                    "bytes_per_core": 128,
                    "total_bytes": 4096,
                    "requester_core_weights": [[i, 128] for i in range(32)],
                    "access_granularity_bytes": 128,
                },
            ]
        }


def make_dram():
    return DRAM(
        CL=14, tRCD=14, tRP=14, tRAS=34,
        bytes_per_row=8192, bytes_per_cycle=64,
        num_cores=32, num_layers=8, banks_per_layer=16,
        num_channels=16, transaction_bytes=128,
        capacity_bytes=192 * 1024**3,
        soft_cores_per_bank=False, lock_cores_per_bank=1,
    )


def make_noc():
    return NoC(
        bandwidth_bytepc=16, topology=Topo.MESH,
        nodes=list(range(32)), use_sram=False,
    )


class ReuseFrontierTest(unittest.TestCase):
    def test_partial_sum_is_write_then_read(self):
        choice = ReuseChoice(4096, 2048, 4096, 2048)
        result = replay_reuse_plan(
            [FakeLog()], make_dram(), make_noc(), [choice]
        )
        self.assertEqual(result["additional_tsv_bytes"], 6144)
        self.assertEqual(result["removed_noc_bytes"], 4096)
        self.assertEqual(result["input_reroute_bytes"], 2048)
        self.assertEqual(result["reduction_reroute_bytes"], 2048)
        phases = result["per_operator"][0]["phase_cycles"]
        self.assertGreater(phases["partial_sum_write"], 0)
        self.assertGreater(phases["partial_sum_read"], 0)
        self.assertEqual(phases["noc_broadcast"], 20)
        self.assertEqual(phases["noc_reduce"], 30)

    def test_no_noc_endpoint_has_only_compute_and_dram(self):
        choice = ReuseChoice(4096, 4096, 4096, 4096)
        result = replay_reuse_plan(
            [FakeLog()], make_dram(), make_noc(), [choice]
        )
        phases = result["per_operator"][0]["phase_cycles"]
        self.assertEqual(phases["noc_broadcast"], 0)
        self.assertEqual(phases["compute_shift"], 100)
        self.assertEqual(phases["noc_reduce"], 0)
        self.assertEqual(result["operator_noc_byte_hops"], 0)

    def test_rejects_non_transaction_aligned_choice(self):
        with self.assertRaisesRegex(ValueError, "transaction aligned"):
            replay_reuse_plan(
                [FakeLog()], make_dram(), make_noc(),
                [ReuseChoice(128, 1, 0, 0)],
            )


if __name__ == "__main__":
    unittest.main()
