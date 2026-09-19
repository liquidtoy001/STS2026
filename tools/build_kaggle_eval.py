"""Export the reviewed notebook and shared evaluator as one uploadable notebook.

The bootstrap materializes exact source snapshots in /kaggle/working for imports
and provenance. Edit repository sources, then rebuild; never edit the export.
Run from any directory: python tools/build_kaggle_eval.py
"""

import json
from pathlib import Path


def build():
    root = Path(__file__).resolve().parents[1]
    notebook_path = root / "notebooks/vits-light_v1.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    snapshots = {relative: (root / relative).read_text(encoding="utf-8") for relative in (
        "notebooks/vits-light_v1.ipynb", "sts_eval/__init__.py", "sts_eval/reconstruction.py",
    )}
    bootstrap = '''# Generated from repository sources. Rebuild to update the evaluator.
import os
from pathlib import Path

runtime_root = Path("/kaggle/working/sts2026-eval-runtime")
runtime_root.mkdir(parents=True, exist_ok=True)
'''
    bootstrap += f"source_snapshots = {snapshots!r}\n"
    bootstrap += '''for relative, content in source_snapshots.items():
    destination = runtime_root / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(content, encoding="utf-8", newline="\\n")
os.environ["STS_REPO_ROOT"] = str(runtime_root)
os.environ["STS_RUN_ROOT"] = "/kaggle/working/sts2026-eval-runs"
'''
    notebook["cells"][:0] = [
        {"cell_type": "markdown", "metadata": {}, "source":
         "# Old-model reconstruction evaluation — Kaggle entry\n\n"
         "Attach LJSpeech first and enable a GPU. Check DATA_DIR below before running. "
         "The final training cell runs 2,000 steps; this is a training-batch reconstruction "
         "experiment, not a held-out or TTS benchmark. Download sts2026-eval-runs after completion. "
         "Shared evaluator code is bundled automatically from the repository.\n"},
        {"cell_type": "code", "metadata": {}, "execution_count": None,
         "outputs": [], "source": bootstrap},
    ]
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
    output = root / "output/kaggle/vits-light_v1_eval.ipynb"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
    print(output)
    return output


if __name__ == "__main__":
    build()
