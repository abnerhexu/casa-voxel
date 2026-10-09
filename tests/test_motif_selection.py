import unittest

from icbm_DNNProgram import DNNProgram, TensorOperator


class MotifSelectionTest(unittest.TestCase):
    @staticmethod
    def make_operator(index):
        return TensorOperator(
            f"op_{index}", 0,
            dim_lengths=[1],
            variables=[[[0]], [[0]]],
            num_cores=[1],
            ignore_variables=[True, False],
            output_idx=index,
            input_idx_list=[1000 + index],
        )

    def make_program(self):
        ops = [self.make_operator(index) for index in range(24)]
        return DNNProgram(
            num_cores=[1], ops=ops, name="motif-test", output_dir="/tmp"
        )

    def test_selects_contiguous_layers_without_renumbering_operators(self):
        program = self.make_program()
        original = list(program.ops)
        program.select_layer_motif(6, 2, 3)

        self.assertEqual(program.ops, original[8:20])
        self.assertEqual(program.motif_start_layer, 2)
        self.assertEqual(program.motif_layers, 3)
        self.assertEqual(program.source_total_layers, 6)

    def test_rejects_nonuniform_layer_width(self):
        program = DNNProgram(
            num_cores=[1],
            ops=[self.make_operator(index) for index in range(10)],
            name="motif-test-invalid",
            output_dir="/tmp",
        )
        with self.assertRaisesRegex(ValueError, "evenly"):
            program.select_layer_motif(4, 0, 2)


if __name__ == "__main__":
    unittest.main()
