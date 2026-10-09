import unittest

from tsim_components.dram_placement import (
    SUPPORTED_DRAM_PLACEMENTS,
    build_placement_plan,
    channel_aware_placements,
    clear_placement_cache,
    placement_cache_info,
    record_signature,
    software_aware_placements,
)
from tsim_thermal.trace import TraceConfig, resolve_dram_bank_mapping


def make_records():
    return [
        {
            "subop_index": 0,
            "tensor_index": 0,
            "tensor_role": "input",
            "stage": "read",
            "total_bytes": 800,
        },
        {
            "subop_index": 0,
            "tensor_index": 1,
            "tensor_role": "weight",
            "stage": "read",
            "total_bytes": 1600,
        },
        {
            "subop_index": 0,
            "tensor_index": 0,
            "tensor_role": "output",
            "stage": "write",
            "total_bytes": 800,
        },
    ]


class DRAMPlacementTest(unittest.TestCase):
    def test_channel_aware_is_an_independent_supported_policy(self):
        self.assertIn("software_aware", SUPPORTED_DRAM_PLACEMENTS)
        self.assertIn("channel_aware", SUPPORTED_DRAM_PLACEMENTS)
        self.assertEqual(
            resolve_dram_bank_mapping(
                TraceConfig(dram_bank_mapping="channel-aware")
            ),
            "channel_aware",
        )

    def test_channel_aware_plan_is_deterministic_and_conserves_bytes(self):
        records = make_records()
        first = channel_aware_placements(records, 16, 4, seed=7)
        second = channel_aware_placements(records, 16, 4, seed=7)

        self.assertEqual(first, second)
        for record in records:
            placement = first[record_signature(record)]
            weights = placement.bank_weights(record["total_bytes"])
            self.assertAlmostEqual(sum(weights.values()), record["total_bytes"])
            self.assertTrue(all(0 <= bank < 16 for bank in weights))
            self.assertEqual(
                set(placement.channel_ids),
                {bank % 4 for bank in placement.bank_ids},
            )

    def test_channel_count_changes_placement_without_changing_records(self):
        records = make_records()
        two_channels = channel_aware_placements(records, 16, 2, seed=3)
        four_channels = channel_aware_placements(records, 16, 4, seed=3)

        key = record_signature(records[1])
        self.assertNotEqual(two_channels[key].bank_ids, four_channels[key].bank_ids)
        self.assertGreater(
            len(four_channels[key].channel_ids),
            len(two_channels[key].channel_ids),
        )

    def test_channel_aware_preserves_separate_read_write_cursors(self):
        records = [
            {
                "subop_index": 0, "tensor_index": 0,
                "tensor_role": "tensor", "stage": "read",
                "total_bytes": 1024,
            },
            {
                "subop_index": 0, "tensor_index": 0,
                "tensor_role": "tensor", "stage": "write",
                "total_bytes": 1024,
            },
        ]
        plan = channel_aware_placements(records, 32, 4, seed=0)

        read = plan[record_signature(records[0])]
        write = plan[record_signature(records[1])]
        self.assertTrue(set(read.bank_ids).isdisjoint(write.bank_ids))

    def test_software_aware_remains_a_separate_contiguous_policy(self):
        records = make_records()
        software = software_aware_placements(records, 16, seed=0)
        channel = channel_aware_placements(records, 16, 4, seed=0)
        key = record_signature(records[1])

        self.assertEqual(software[key].policy, "software_aware")
        self.assertEqual(channel[key].policy, "channel_aware")
        self.assertNotEqual(software[key].bank_ids, channel[key].bank_ids)

    def test_invalid_channel_geometry_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "must be divisible"):
            channel_aware_placements(make_records(), 16, 3)

    def test_shared_builder_supports_every_public_policy(self):
        records = make_records()
        for policy in SUPPORTED_DRAM_PLACEMENTS:
            with self.subTest(policy=policy):
                plan = build_placement_plan(
                    records, total_banks=16, num_channels=4,
                    policy=policy, seed=11, stripe_bytes=128,
                )
                self.assertEqual(len(plan), len(records))
                for placement in plan.values():
                    self.assertEqual(placement.policy, policy)
                    self.assertTrue(placement.bank_ids)

    def test_placement_cache_is_keyed_by_policy_and_channel_geometry(self):
        records = make_records()
        clear_placement_cache()
        build_placement_plan(records, 16, 2, "software_aware", seed=1)
        first = placement_cache_info()
        build_placement_plan(records, 16, 2, "software_aware", seed=1)
        second = placement_cache_info()
        build_placement_plan(records, 16, 4, "software_aware", seed=1)
        build_placement_plan(records, 16, 4, "channel_aware", seed=1)
        final = placement_cache_info()

        self.assertEqual(first.misses, 1)
        self.assertEqual(second.hits, 1)
        self.assertEqual(final.misses, 3)

    def test_global_tensor_identity_coalesces_read_and_write(self):
        records = [
            {
                "tensor_id": 42, "subop_index": 0, "tensor_index": 0,
                "tensor_role": "output", "stage": "write",
                "total_bytes": 128, "allocation_bytes": 1024,
                "address": 4096,
            },
            {
                "tensor_id": 42, "subop_index": 3, "tensor_index": 1,
                "tensor_role": "input", "stage": "read",
                "total_bytes": 512, "allocation_bytes": 1024,
                "address": 4096,
            },
        ]
        plan = build_placement_plan(
            records, total_banks=16, num_channels=4,
            policy="address_trace", stripe_bytes=128,
        )
        self.assertEqual(len(plan), 1)
        self.assertEqual(
            plan[record_signature(records[0])],
            plan[record_signature(records[1])],
        )


if __name__ == "__main__":
    unittest.main()
