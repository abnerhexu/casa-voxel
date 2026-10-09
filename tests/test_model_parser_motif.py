import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


class ModelParserMotifTest(unittest.TestCase):
    def test_selects_source_layers_before_operator_partitioning(self):
        voxel_root = Path(__file__).resolve().parents[1]
        parser = voxel_root / "models" / "model_parser.py"
        source_model = voxel_root / "models" / "original" / "qwen-3.5-9b.json"

        with tempfile.TemporaryDirectory() as temp_dir:
            workdir = Path(temp_dir)
            (workdir / "original").mkdir()
            shutil.copy2(source_model, workdir / "original" / source_model.name)

            subprocess.run(
                [
                    sys.executable,
                    str(parser),
                    source_model.name,
                    "1",
                    "2",
                    "512",
                    "--total-layers",
                    "32",
                    "--motif-start-layer",
                    "0",
                    "--motif-layers",
                    "4",
                ],
                cwd=workdir,
                check=True,
            )

            parsed_path = workdir / "parsed" / "parsed_qwen-3.5-9b-motif0+4.json"
            texpr_path = workdir / "TExpr" / "TExpr_qwen-3.5-9b-b1-motif0+4.json"
            self.assertTrue(parsed_path.is_file())
            self.assertTrue(texpr_path.is_file())
            with parsed_path.open() as handle:
                self.assertEqual(len(json.load(handle)), 4 * 16)
            with texpr_path.open() as handle:
                self.assertGreater(len(json.load(handle)), 0)


if __name__ == "__main__":
    unittest.main()
