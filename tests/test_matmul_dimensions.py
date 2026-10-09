import unittest
import math
import numpy as np

from tsim_components import utils
from t10_TensorExpression import TensorExpression


class MatmulDimensionInferenceTest(unittest.TestCase):
    def test_standard_matrix_multiply(self):
        variables = [[[0], [1]], [[0], [2]], [[2], [1]]]
        self.assertEqual(
            utils.dim_var_to_bkmn([64, 128, 256], variables),
            (1, 256, 64, 128),
        )

    def test_qkv_projection_flattens_batch_sequence_and_heads(self):
        # output[B,S,N,D] = input[B,S,H] * weight[H,N,D]
        variables = [
            [[0], [1], [2], [3]],
            [[0], [1], [4]],
            [[4], [2], [3]],
        ]
        self.assertEqual(
            utils.dim_var_to_bkmn([16, 4096, 32, 128, 4096], variables),
            (1, 4096, 16 * 4096, 32 * 128),
        )

    def test_attention_output_projection_supports_two_k_dimensions(self):
        # output[B,S,H] = activation[B,S,V,D] * weight[V,D,H]
        variables = [
            [[0], [1], [2]],
            [[0], [1], [3], [4]],
            [[3], [4], [2]],
        ]
        self.assertEqual(
            utils.dim_var_to_bkmn([4, 1024, 4096, 32, 128], variables),
            (1, 32 * 128, 4 * 1024, 4096),
        )

    def test_batched_attention_keeps_shared_batch_dimensions(self):
        # output[B,V,Q,K] = query[B,V,Q,H] * key[B,V,H,K]
        variables = [
            [[0], [1], [2], [3]],
            [[0], [1], [2], [4]],
            [[0], [1], [4], [3]],
        ]
        self.assertEqual(
            utils.dim_var_to_bkmn([4, 32, 1024, 2048, 128], variables),
            (4 * 32, 128, 1024, 2048),
        )

    def test_qkv_projection_energy_path_uses_generalized_dimensions(self):
        dimensions = [2, 1, 32, 128, 4096]
        variables = [
            [[0], [1], [2], [3]],
            [[0], [1], [4]],
            [[4], [2], [3]],
        ]
        expression = TensorExpression(
            TensorExpression.OP_TYPE_MATMUL,
            dimensions,
            variables,
            num_cores=[256],
        )
        cycles, energy, breakdown = \
            expression.get_comp_time_energy_per_iter_helper(
                np.array(dimensions)
            )

        self.assertTrue(math.isfinite(cycles))
        self.assertTrue(math.isfinite(energy))
        self.assertGreater(breakdown["sa"], 0)


if __name__ == "__main__":
    unittest.main()
