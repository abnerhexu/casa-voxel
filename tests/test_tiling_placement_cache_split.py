import tempfile
import types
import unittest
from types import SimpleNamespace

from icbm_DNNProgram import DNNProgram
from tsim_components.mem import DRAM


def make_dram(channels=1, bytepc=32):
    return DRAM(
        CL=14,
        tRCD=14,
        tRP=14,
        bytes_per_row=1024,
        bytes_per_cycle=bytepc,
        num_cores=8,
        num_layers=1,
        banks_per_layer=8,
        num_channels=channels,
        transaction_bytes=128,
        soft_cores_per_bank=False,
        lock_cores_per_bank=1,
    )


def make_operator():
    config_a = ((1,), ((1,),))
    config_b = ((2,), ((2,),))
    expr = SimpleNamespace(config_dict={
        config_a: (128, 100, 0, 0),
        config_b: (256, 80, 0, 0),
    })
    return SimpleNamespace(
        name="test-op",
        op_type=1,
        dim_lengths=[8, 8],
        variables=[[[0]], [[1]]],
        num_byte_per_elem=2,
        ignore_variables=[False, False],
        expr=expr,
    )


class TilingPlacementCacheSplitTest(unittest.TestCase):
    def make_program(self, output_dir):
        program = DNNProgram(num_cores=[8], output_dir=output_dir)
        program._test_dram_calls = 0

        def fake_get_dram_time(this, fused_op, dram, core_group, partition_it,
                               for_tiling=False):
            this._test_dram_calls += 1
            temporal, spatial = next(partition_it)
            cycles = 50 if spatial == (2,) else 100
            return (cycles, 0, 0, 0, 0, 0, False, 0, 0, 0, [], [])

        program.get_dram_time = types.MethodType(fake_get_dram_time, program)
        return program

    def test_channel_and_placement_changes_reuse_tiling(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            program = self.make_program(tmpdir)
            op = make_operator()
            first = program.get_best_config_by_max_mem_size(
                op, make_dram(channels=1), 512, 8
            )
            program.dram_placement_policy = "channel_aware"
            second = program.get_best_config_by_max_mem_size(
                op, make_dram(channels=4), 512, 8
            )

            self.assertEqual(first, second)
            self.assertEqual(program._test_dram_calls, 2)
            self.assertEqual(program.tiling_cache_info(), {
                "hits": 1, "misses": 1, "entries": 1,
            })

    def test_bandwidth_and_sram_changes_create_new_tiling_entries(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            program = self.make_program(tmpdir)
            op = make_operator()
            program.get_best_config_by_max_mem_size(op, make_dram(bytepc=32), 512, 8)
            program.get_best_config_by_max_mem_size(op, make_dram(bytepc=64), 512, 8)
            program.get_best_config_by_max_mem_size(op, make_dram(bytepc=64), 256, 8)

            self.assertEqual(program.tiling_cache_info()["misses"], 3)
            self.assertEqual(program.tiling_cache_info()["entries"], 3)

    def test_compiled_noc_candidate_cost_change_invalidates_tiling(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            program = self.make_program(tmpdir)
            op = make_operator()
            program.get_best_config_by_max_mem_size(op, make_dram(), 512, 8)
            op.expr.config_dict = {
                config: (values[0], values[1] + 10, values[2], values[3])
                for config, values in op.expr.config_dict.items()
            }
            program.get_best_config_by_max_mem_size(op, make_dram(), 512, 8)

            self.assertEqual(program.tiling_cache_info()["misses"], 2)
            self.assertEqual(program.tiling_cache_info()["entries"], 2)

    def test_tiling_cache_can_be_persisted_without_placement_state(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = f"{tmpdir}/tiling.cache"
            op = make_operator()
            first_program = self.make_program(tmpdir)
            expected = first_program.get_best_config_by_max_mem_size(
                op, make_dram(channels=1), 512, 8
            )
            first_program.save_tiling_cache(cache_path)

            second_program = self.make_program(tmpdir)
            self.assertEqual(second_program.load_tiling_cache(cache_path), 1)
            actual = second_program.get_best_config_by_max_mem_size(
                op, make_dram(channels=4), 512, 8
            )
            self.assertEqual(actual, expected)
            self.assertEqual(second_program._test_dram_calls, 0)

    def test_programs_without_new_cache_fields_remain_usable(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            program = self.make_program(tmpdir)
            del program._tiling_selection_cache
            del program._tiling_cache_hits
            del program._tiling_cache_misses
            del program._tiling_candidate_fingerprints

            result = program.get_best_config_by_max_mem_size(
                make_operator(), make_dram(), 512, 8
            )
            self.assertTrue(result)
            self.assertEqual(program.tiling_cache_info()["misses"], 1)


if __name__ == "__main__":
    unittest.main()
