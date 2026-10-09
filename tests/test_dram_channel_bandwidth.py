import unittest

from tsim_components.mem import DRAM


def make_dram(*, channels, total_bytepc=32, precise=True):
    return DRAM(
        CL=14,
        tRCD=14,
        tRP=14,
        bytes_per_row=1024,
        bytes_per_cycle=total_bytepc,
        num_cores=256,
        num_layers=1,
        banks_per_layer=8,
        num_channels=channels,
        transaction_bytes=128,
        precise=precise,
        soft_cores_per_bank=False,
        lock_cores_per_bank=1,
        tRFC=0,
    )


class DRAMChannelBandwidthTest(unittest.TestCase):
    def test_total_bandwidth_is_split_evenly_across_channels(self):
        dram = make_dram(channels=4, total_bytepc=32)

        self.assertEqual(dram.total_bytes_per_cycle, 32)
        self.assertEqual(dram.channel_bytes_per_cycle, 8)
        self.assertEqual(
            dram.channel_bytes_per_cycle * dram.num_channels,
            dram.total_bytes_per_cycle,
        )
        self.assertEqual(dram.bandwidth_geometry()["transaction_bytes"], 128)

    def test_transaction_size_is_independent_from_total_bandwidth(self):
        slow = make_dram(channels=1, total_bytepc=16, precise=False)
        fast = make_dram(channels=1, total_bytepc=64, precise=False)

        self.assertEqual(
            slow.num_row_conflicts_of_access(10 * 128, 128),
            fast.num_row_conflicts_of_access(10 * 128, 128),
        )
        self.assertEqual(slow.num_row_conflicts_of_access(10 * 128, 128), 2)

    def test_channels_run_concurrently_under_fixed_total_bandwidth(self):
        one_channel = make_dram(channels=1)
        four_channels = make_dram(channels=4)

        one_cycle = one_channel.num_cycle_of_access(8 * 128, 128, True)
        four_cycle = four_channels.num_cycle_of_access(8 * 128, 128, True)

        # Total bandwidth is identical. The multi-channel design can overlap
        # independent ACT/CL latency, while each channel transfers at 1/4 BW.
        self.assertLess(four_cycle, one_cycle)
        self.assertGreaterEqual(one_cycle, 8 * 128 / 32)
        self.assertGreaterEqual(four_cycle, 8 * 128 / 32)


if __name__ == "__main__":
    unittest.main()
