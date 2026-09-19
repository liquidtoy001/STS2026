"""Exercise the actual export bootstrap without writing to a Kaggle directory."""

import ast
import json
import os
from pathlib import Path
import tempfile
import unittest

from tools.build_kaggle_eval import build


class KaggleExportTests(unittest.TestCase):
    def test_bootstrap_reconstructs_sources_and_cells_compile(self):
        output = build()
        root = Path(__file__).resolve().parents[1]
        notebook = json.loads(output.read_text(encoding="utf-8"))
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                ast.parse(cell["source"])
                self.assertEqual(cell["outputs"], [])
                self.assertIsNone(cell["execution_count"])
        environment = dict(os.environ)
        try:
            with tempfile.TemporaryDirectory() as directory:
                code = notebook["cells"][1]["source"].replace(
                    "/kaggle/working/sts2026-eval-runtime", Path(directory).as_posix(),
                )
                exec(compile(code, "kaggle_bootstrap", "exec"), {})
                for relative in ("notebooks/vits-light_v1.ipynb", "sts_eval/__init__.py",
                                 "sts_eval/reconstruction.py"):
                    self.assertEqual((Path(directory) / relative).read_bytes(),
                                     (root / relative).read_text(encoding="utf-8").encode("utf-8"))
        finally:
            os.environ.clear()
            os.environ.update(environment)


if __name__ == "__main__":
    unittest.main()
