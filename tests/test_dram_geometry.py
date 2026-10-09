import unittest

from tsim_components.mem import DRAM, DRAMGeometry


class DRAMGeometryTest(unittest.TestCase):
    def test_default_geometry_is_eight_by_sixteen(self):
        dram = DRAM(
            CL=14,
            tRCD=14,
            tRP=14,
            bytes_per_row=8192,
            bytes_per_cycle=32,
            num_cores=256,
        )

        self.assertEqual(dram.num_layers, 8)
        self.assertEqual(dram.banks_per_layer, 16)
        self.assertEqual(dram.num_banks, 128)
        self.assertEqual(dram.num_channels, 1)
        self.assertEqual(dram.num_banks_per_channel, 128)
        self.assertEqual(dram.transaction_bytes, 128)

    def test_channels_partition_one_shared_bank_geometry(self):
        geometry = DRAMGeometry(
            num_layers=8,
            banks_per_layer=16,
            num_channels=8,
            bytes_per_row=4096,
            transaction_bytes=128,
        )

        self.assertEqual(geometry.total_banks, 128)
        self.assertEqual(geometry.banks_per_channel, 16)
        self.assertEqual(geometry.decode_bank(0), (0, 0, 0, 0))
        self.assertEqual(geometry.decode_bank(17), (1, 2, 1, 1))
        self.assertEqual(geometry.decode_bank(127), (7, 15, 7, 15))

    def test_invalid_channel_partition_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "must be divisible"):
            DRAMGeometry(
                num_layers=8,
                banks_per_layer=16,
                num_channels=3,
                bytes_per_row=8192,
            )

    def test_legacy_bank_count_remains_supported(self):
        dram = DRAM(
            CL=14,
            tRCD=14,
            tRP=14,
            bytes_per_row=256,
            bytes_per_cycle=16,
            num_cores=4,
            num_banks_per_channel=4,
        )

        self.assertEqual(dram.num_banks, 4)
        self.assertEqual(dram.num_banks_per_channel, 4)
        self.assertEqual(dram.num_row_conflicts_of_access(5 * 256, 256), 1)


if __name__ == "__main__":
    unittest.main()
