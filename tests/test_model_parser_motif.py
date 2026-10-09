import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


class ModelParserMotifTest(unittest.TestCase):
    def test_isolates_concurrent_artifacts_by_configuration_tag(self):
        voxel_root = Path(__file__).resolve().parents[1]
        parser = voxel_root / "models" / "model_parser.py"
        source_model = voxel_root / "models" / "original" / "qwen-3.5-9b.json"

        with tempfile.TemporaryDirectory() as temp_dir:
            workdir = Path(temp_dir)
            (workdir / "original").mkdir()
            shutil.copy2(source_model, workdir / "original" / source_model.name)

            processes = []
            for capacity, artifact_tag in (
                ("256", "qwen9-prefill-sram1MiB"),
                ("512", "qwen9-prefill-sram2MiB"),
            ):
                command = [
                    sys.executable,
                    str(parser),
                    source_model.name,
                    "1",
                    "2",
                    capacity,
                    "--total-layers",
                    "32",
                    "--motif-start-layer",
                    "0",
                    "--motif-layers",
                    "4",
                    "--artifact-tag",
                    artifact_tag,
                ]
                processes.append(
                    subprocess.Popen(
                        command,
                        cwd=workdir,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.PIPE,
                        text=True,
                    )
                )

            for process in processes:
                _stdout, stderr = process.communicate(timeout=30)
                self.assertEqual(process.returncode, 0, stderr)

            for artifact_tag in (
                "qwen9-prefill-sram1MiB",
                "qwen9-prefill-sram2MiB",
            ):
                parsed_path = workdir / "parsed" / f"parsed_{artifact_tag}.json"
                texpr_path = workdir / "TExpr" / f"TExpr_{artifact_tag}.json"
                self.assertTrue(parsed_path.is_file())
                self.assertTrue(texpr_path.is_file())
                with parsed_path.open() as handle:
                    self.assertEqual(len(json.load(handle)), 4 * 16)
                with texpr_path.open() as handle:
                    self.assertGreater(len(json.load(handle)), 0)

            self.assertEqual(list((workdir / "parsed").glob(".*.tmp")), [])
            self.assertEqual(list((workdir / "TExpr").glob(".*.tmp")), [])


if __name__ == "__main__":
    unittest.main()
