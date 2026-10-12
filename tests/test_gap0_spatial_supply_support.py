import unittest

from tsim_components.dram_scheduler import _locality_matched_dram_flows
from tsim_components.noc import NoC, Topo
from tsim_components.system_model import Hardware


class Gap0SpatialSupplySupportTest(unittest.TestCase):
    def test_system_model_allows_more_channels_than_cores(self):
        hardware = Hardware(
            cores=4,
            mesh_width=2,
            banks=16,
            channels=8,
        )

        self.assertEqual(len(hardware.channel_nodes), 8)
        self.assertTrue(all(0 <= node < 4 for node in hardware.channel_nodes))

    def test_explicit_channel_requester_flow_is_not_rematched(self):
        noc = NoC(
            bandwidth_bytepc=16,
            topology=Topo.MESH,
            nodes=list(range(4)),
        )
        flows = _locality_matched_dram_flows(
            noc,
            channel_nodes=(0, 3),
            channel_weights={0: 64, 1: 64},
            demand=((0, 64), (3, 64)),
            # Deliberately cross the mesh.  The old greedy matcher would
            # replace this with the two zero-hop assignments.
            explicit_flows=((0, 3, 64), (1, 0, 64)),
        )

        self.assertEqual([(c, r, b) for c, r, b, _ in flows], [
            (0, 3, 64.0),
            (1, 0, 64.0),
        ])
        self.assertTrue(all(len(path) == 3 for *_, path in flows))

    def test_explicit_flow_must_conserve_every_channel(self):
        noc = NoC(
            bandwidth_bytepc=16,
            topology=Topo.MESH,
            nodes=list(range(4)),
        )
        with self.assertRaisesRegex(ValueError, "per-channel supply"):
            _locality_matched_dram_flows(
                noc,
                channel_nodes=(0, 3),
                channel_weights={0: 64, 1: 64},
                demand=((0, 64), (3, 64)),
                explicit_flows=((0, 0, 64), (0, 3, 64)),
            )


if __name__ == "__main__":
    unittest.main()
