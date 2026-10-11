import unittest
from types import SimpleNamespace

from tsim_components.aggregate_trace import movement_profile


class FakeCompute:
    def convert_op_simple(self, **_kwargs):
        return object()

    def get_total_cycle_for_fused_op(self, *_args, **_kwargs):
        return [1]

    def get_peak_flopc(self):
        return (1, 1)


class SkiSlopeMovementTest(unittest.TestCase):
    def test_profile_conserves_vertical_and_remote_payloads(self):
        spec = {
            "index": 0,
            "name": "matmul",
            "op_type": 5,
            "dims": [4, 4, 4],
            "variables": [[0, 1], [0, 2], [2, 1]],
            "tensors": ["out", "a", "b"],
            "spatial": [2, 2, 1],
            "temporal": [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
            "element_bytes": 2,
            "source_hot_bytes": 1,
        }
        hw = SimpleNamespace(cores=4, mesh_width=2)
        result = movement_profile(spec, FakeCompute(), hw)

        self.assertEqual(
            result["vertical_bytes"],
            sum(result["vertical_bytes_by_core"]),
        )
        self.assertEqual(
            result["remote_sram_bytes"],
            sum(result["remote_sram_bytes_by_core"]),
        )
        self.assertEqual(
            result["private_bytes"],
            sum(result["private_bytes_by_core"]),
        )
        self.assertGreater(result["remote_sram_bytes"], 0)
        self.assertGreaterEqual(
            result["noc_byte_hops"], result["remote_sram_bytes"]
        )

    def test_zero_hop_flows_do_not_count_as_remote(self):
        spec = {
            "index": 0,
            "name": "elementwise",
            "op_type": 2,
            "dims": [4],
            "variables": [[0], [0]],
            "tensors": ["out", "in"],
            "spatial": [4],
            "temporal": [[1, 1]],
            "element_bytes": 2,
            "source_hot_bytes": 1,
        }
        result = movement_profile(
            spec, FakeCompute(), SimpleNamespace(cores=4, mesh_width=2)
        )
        self.assertEqual(result["remote_sram_bytes"], 0)
        self.assertEqual(result["noc_byte_hops"], 0)


if __name__ == "__main__":
    unittest.main()
