import unittest

from tsim_components.dram_placement import (
    SUPPORTED_DRAM_PLACEMENTS,
    balanced_channel_core_groups,
    build_placement_plan,
    channel_aware_placements,
    channel_injection_nodes,
    clear_placement_cache,
    placement_cache_info,
    record_signature,
    software_aware_placements,
)
from tsim_components.noc import NoC, Topo
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


def make_mesh(num_cores=8):
    return NoC(
        bandwidth_bytepc=16,
        topology=Topo.MESH,
        nodes=list(range(num_cores)),
    )


def with_requesters(records, core=0):
    enriched = []
    for record in records:
        item = dict(record)
        item["requester_core_weights"] = [[core, item["total_bytes"]]]
        enriched.append(item)
    return enriched


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

    def test_noc_aware_is_an_independent_supported_policy(self):
        self.assertIn("noc_aware", SUPPORTED_DRAM_PLACEMENTS)
        self.assertEqual(
            resolve_dram_bank_mapping(
                TraceConfig(dram_bank_mapping="noc-aware")
            ),
            "noc_aware",
        )

    def test_channels_evenly_partition_dimension_ordered_cores(self):
        self.assertEqual(
            balanced_channel_core_groups(3, 8),
            ((0, 1, 2), (3, 4, 5), (6, 7)),
        )
        self.assertEqual(channel_injection_nodes(3, 8), (1, 4, 6))

    def test_more_channels_than_cores_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "cannot bind 9 DRAM channels"):
            balanced_channel_core_groups(9, 8)
        with self.assertRaisesRegex(ValueError, "cannot bind 9 DRAM channels"):
            build_placement_plan(
                with_requesters(make_records()),
                total_banks=16,
                num_channels=9,
                policy="noc_aware",
                noc=make_mesh(8),
            )

    def test_noc_aware_prefers_nearby_channel_injection_nodes(self):
        records = [
            {
                "tensor_id": "left",
                "stage": "read",
                "total_bytes": 100,
                "requester_core_weights": [[0, 100]],
            },
            {
                "tensor_id": "right",
                "stage": "read",
                "total_bytes": 100,
                "requester_core_weights": [[7, 100]],
            },
        ]
        plan = build_placement_plan(
            records,
            total_banks=2,
            num_channels=2,
            policy="noc_aware",
            noc=make_mesh(8),
        )

        self.assertEqual(plan[record_signature(records[0])].channel_ids, (0,))
        self.assertEqual(plan[record_signature(records[1])].channel_ids, (1,))

    def test_noc_aware_preserves_software_aware_bank_spans(self):
        records = with_requesters(make_records())
        software = software_aware_placements(records, total_banks=16)
        noc_aware = build_placement_plan(
            records,
            total_banks=16,
            num_channels=4,
            policy="noc_aware",
            noc=make_mesh(8),
        )
        self.assertEqual(
            {
                signature: len(placement.bank_ids)
                for signature, placement in software.items()
            },
            {
                signature: len(placement.bank_ids)
                for signature, placement in noc_aware.items()
            },
        )

    def test_noc_aware_cache_includes_requester_traffic(self):
        left = with_requesters(make_records(), core=0)
        right = with_requesters(make_records(), core=7)
        clear_placement_cache()
        left_plan = build_placement_plan(
            left, 3, 2, "noc_aware", noc=make_mesh(8)
        )
        right_plan = build_placement_plan(
            right, 3, 2, "noc_aware", noc=make_mesh(8)
        )

        self.assertNotEqual(left_plan, right_plan)
        self.assertEqual(placement_cache_info().misses, 2)

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

    def test_non_divisible_channel_geometry_is_balanced_and_valid(self):
        plan = channel_aware_placements(make_records(), 16, 3)
        self.assertEqual(len(plan), len(make_records()))
        for placement in plan.values():
            self.assertTrue(placement.bank_ids)
            self.assertTrue(all(0 <= bank < 16 for bank in placement.bank_ids))
            self.assertEqual(
                set(placement.channel_ids),
                {bank % 3 for bank in placement.bank_ids},
            )

    def test_shared_builder_supports_every_public_policy(self):
        records = make_records()
        for policy in SUPPORTED_DRAM_PLACEMENTS:
            with self.subTest(policy=policy):
                policy_records = (
                    with_requesters(records) if policy == "noc_aware" else records
                )
                plan = build_placement_plan(
                    policy_records, total_banks=16, num_channels=4,
                    policy=policy, seed=11, stripe_bytes=128,
                    noc=make_mesh(8) if policy == "noc_aware" else None,
                )
                self.assertEqual(len(plan), len(records))
                for placement in plan.values():
                    self.assertEqual(placement.policy, policy)
                    self.assertTrue(placement.bank_ids)

    def test_address_trace_and_hbm_interleave_use_distinct_start_rules(self):
        records = make_records()
        for ordinal, item in enumerate(records):
            item["address"] = ordinal * 4096
        address_plan = build_placement_plan(
            records, 16, 4, "address_trace", stripe_bytes=128
        )
        interleave_plan = build_placement_plan(
            records, 16, 4, "hbm_interleave", stripe_bytes=128
        )
        self.assertTrue(any(
            address_plan[key].bank_ids != interleave_plan[key].bank_ids
            for key in address_plan
        ))

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
