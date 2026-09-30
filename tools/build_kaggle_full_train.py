"""Export notebooks/vits-full-train.ipynb as two uploadable Kaggle notebooks.

One notebook per arm of the GAN comparison: gan0 (GAN_WEIGHT = 0) and gan1
(GAN_WEIGHT = 1). Everything else is identical. The shared evaluator source is
bundled, so no extra Kaggle dataset is needed besides LJSpeech.
Run from any directory: python tools/build_kaggle_full_train.py
"""

import json
from pathlib import Path

ARMS = {"gan0": "0.0", "gan1": "1.0"}
GAN_LINE = 'GAN_WEIGHT = float(os.environ.get("STS_GAN_WEIGHT", "1.0"))'


def build():
    root = Path(__file__).resolve().parents[1]
    notebook_path = root / "notebooks/vits-full-train.ipynb"
    snapshots = {relative: (root / relative).read_text(encoding="utf-8") for relative in (
        "notebooks/vits-full-train.ipynb", "sts_eval/__init__.py", "sts_eval/reconstruction.py",
    )}
    bootstrap = '''# Generated from repository sources. Rebuild to update; never edit the export.
import os
from pathlib import Path

runtime_root = Path(os.environ.get("STS_RUNTIME_ROOT", "/kaggle/working/sts2026-runtime"))
runtime_root.mkdir(parents=True, exist_ok=True)
'''
    bootstrap += f"source_snapshots = {snapshots!r}\n"
    bootstrap += '''for relative, content in source_snapshots.items():
    destination = runtime_root / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(content, encoding="utf-8", newline="\\n")
os.environ["STS_REPO_ROOT"] = str(runtime_root)
os.environ.setdefault("STS_RUN_ROOT", "/kaggle/working/sts2026-full-runs")
'''
    outputs = []
    for arm, weight in ARMS.items():
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        replaced = 0
        for cell in notebook["cells"]:
            if cell["cell_type"] != "code":
                continue
            source = "".join(cell["source"])
            if GAN_LINE in source:
                source = source.replace(GAN_LINE, GAN_LINE.replace('"1.0"', f'"{weight}"'))
                replaced += 1
            cell["source"] = source
            cell["execution_count"] = None
            cell["outputs"] = []
        if replaced != 1:
            raise RuntimeError(f"Expected one GAN_WEIGHT line, found {replaced}.")
        notebook["cells"][:0] = [
            {"cell_type": "markdown", "metadata": {}, "source":
             f"# STS2026 full training — {arm} (GAN_WEIGHT = {weight})\n\n"
             "1. Add Input: the LJSpeech-1.1 dataset. The notebook finds `metadata.csv` and `wavs/` itself.\n"
             "2. Settings: Accelerator = GPU, Internet on is not required.\n"
             "3. Save Version → **Save & Run All (Commit)**. The run continues after you close the browser.\n"
             "4. The log prints speed and ETA every 1,000 iterations. The run stops cleanly at 11.3 h "
             "if it is not finished; see the repository docs to resume.\n"
             "5. When it finishes, download `sts2026-full-runs/` from the Output tab and upload it to the team Drive.\n"},
            {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": bootstrap},
        ]
        output = root / f"output/kaggle/vits-full-train_{arm}.ipynb"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
        outputs.append(output)
        print(output)
    return outputs


if __name__ == "__main__":
    build()
