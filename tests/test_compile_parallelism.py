import unittest
from unittest.mock import patch

from icbm_DNNProgram import _classify_and_setup_ops


class _FakeExpression:
    def __init__(self, light=False):
        self._light = light
        self.dim_lengths = [4096, 4096, 4096, 1]
        self.num_cores = [256]

    def is_light_op(self):
        return self._light

    def get_util_threshold(self):
        return 0.97


class CompileParallelismTest(unittest.TestCase):
    @staticmethod
    def classify(exprs, requested, cpu_cap=128, available_gb=128):
        with (
            patch("icbm_DNNProgram._get_compile_thread_cap", return_value=cpu_cap),
            patch("icbm_DNNProgram._get_available_memory_gb", return_value=available_gb),
        ):
            return _classify_and_setup_ops(
                exprs,
                is_intra_mode=True,
                num_threads=requested,
            )

    def test_requested_budget_is_split_across_and_within_heavy_ops(self):
        result = self.classify([_FakeExpression() for _ in range(12)], 12)
        threads_per_heavy = result[3]
        heavy_batch_size = result[4]
        worker_budget = result[9]

        self.assertEqual(worker_budget, 12)
        self.assertEqual(threads_per_heavy, 4)
        self.assertEqual(heavy_batch_size, 3)
        self.assertLessEqual(threads_per_heavy * heavy_batch_size, worker_budget)

    def test_cpu_cap_limits_requested_budget(self):
        result = self.classify([_FakeExpression() for _ in range(12)], 64, cpu_cap=8)
        self.assertEqual(result[9], 8)
        self.assertLessEqual(result[3] * result[4], 8)

    def test_memory_cap_limits_requested_budget(self):
        result = self.classify(
            [_FakeExpression() for _ in range(12)],
            64,
            cpu_cap=64,
            available_gb=6,
        )
        self.assertEqual(result[9], 6)
        self.assertLessEqual(result[3] * result[4], 6)

    def test_light_ops_use_one_inner_worker_to_avoid_nested_oversubscription(self):
        result = self.classify([_FakeExpression(light=True) for _ in range(7)], 4)
        self.assertEqual(result[2], 1)
        self.assertEqual(result[7], 7)
        self.assertEqual(result[9], 4)

    def test_rejects_nonpositive_budget(self):
        with self.assertRaisesRegex(ValueError, "positive"):
            self.classify([_FakeExpression()], 0)


if __name__ == "__main__":
    unittest.main()
