import unittest
from types import SimpleNamespace
from unittest.mock import patch

from tsim_components.tiling_search import (
    RankedOperatorTiling,
    enumerate_motif_tiling_candidates,
)


def candidate(label, score):
    return RankedOperatorTiling(
        config=((label,), ((1,),)),
        score_cycles=score,
        compute_noc_cycles=score - 1,
        transfer_lower_bound_cycles=score // 2,
        hot_memory_bytes=128,
    )


class MotifTilingSearchTest(unittest.TestCase):
    def test_k_best_product_is_sorted_unique_and_bounded(self):
        prog = SimpleNamespace(ops=["a", "b", "c"])
        pools = {
            "a": [candidate("a0", 1), candidate("a1", 3)],
            "b": [candidate("b0", 10), candidate("b1", 12)],
            "c": [candidate("c0", 100), candidate("c1", 105)],
        }

        with patch(
            "tsim_components.tiling_search.rank_operator_tilings",
            side_effect=lambda op, *args, **kwargs: pools[op],
        ):
            results = enumerate_motif_tiling_candidates(
                prog,
                dram=object(),
                noc=object(),
                core_group_size=8,
                max_memory_bytes=1024,
                limit=5,
                per_operator_limit=2,
            )

        self.assertEqual([item.score_cycles for item in results], [111, 113, 113, 115, 116])
        self.assertEqual([item.rank for item in results], [1, 2, 3, 4, 5])
        self.assertEqual(len({item.candidate_id for item in results}), 5)
        self.assertEqual(len(results[0].configs), 3)

    def test_empty_program_has_no_candidates(self):
        results = enumerate_motif_tiling_candidates(
            SimpleNamespace(ops=[]),
            dram=object(),
            noc=object(),
            core_group_size=8,
            max_memory_bytes=1024,
        )
        self.assertEqual(results, [])


if __name__ == "__main__":
    unittest.main()
