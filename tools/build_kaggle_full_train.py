"""Export notebooks/vits-full-train.ipynb as one uploadable Kaggle notebook per TTS_0.10 arm.

Each export differs only in the default STS_ARM (R0, R0b, E0, G1-G3, A1, A2, L1, O1, D1;
see docs/ABLATION_TTS_0.10.md; T011, docs/TTS_0.11.md). The shared evaluator and the held-out splits are bundled,
so no extra Kaggle dataset is needed besides LJSpeech.
Run from any directory: python tools/build_kaggle_full_train.py
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.notebook_defs import literal_assignment  # noqa: E402

ARM_LINE = 'ARM = os.environ.get("STS_ARM", "R0")'
SOURCES = ("notebooks/vits-full-train.ipynb", "sts_eval/__init__.py", "sts_eval/reconstruction.py",
           "sts_eval/splits/ljspeech_heldout_100.txt", "sts_eval/splits/ljspeech_long_100.txt")


def build():
    notebook_path = ROOT / "notebooks/vits-full-train.ipynb"
    arms = literal_assignment("ARMS")
    snapshots = {relative: (ROOT / relative).read_text(encoding="utf-8") for relative in SOURCES}
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
    for arm, spec in arms.items():
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        replaced = 0
        for cell in notebook["cells"]:
            source = "".join(cell["source"])
            if cell["cell_type"] == "code":
                if ARM_LINE in source:
                    source = source.replace(ARM_LINE, ARM_LINE.replace('"R0")', f'"{arm}")'))
                    replaced += 1
                cell["execution_count"] = None
                cell["outputs"] = []
            cell["source"] = source
        if replaced != 1:
            raise RuntimeError(f"Expected one ARM line, found {replaced}.")
        reverted = ", ".join(f"`{name}`" for name in spec["revert"]) or "none"
        notebook["cells"][:0] = [
            {"cell_type": "markdown", "metadata": {}, "source":
             f"# STS2026 TTS_0.10 — arm {arm}: {spec['title']}\n\n"
             f"Reverted to TTS_0.02: {reverted}. Seed {spec['seed']}.\n\n"
             "1. Add Input: the LJSpeech-1.1 dataset. The notebook finds `metadata.csv` and `wavs/` itself.\n"
             "2. Settings: Accelerator = GPU. Internet is not required.\n"
             "3. Save Version → **Save & Run All (Commit)**. The run continues after you close the browser.\n"
             "4. The log prints losses, speed and ETA every 1,000 steps. A run stops cleanly at 11.3 h if "
             "it is not finished; see docs/ABLATION_TTS_0.10.md to resume.\n"
             "5. When it finishes, download `sts2026-full-runs/` from the Output tab and upload it to the "
             "team Drive as described in docs/ABLATION_TTS_0.10.md.\n"},
            {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": bootstrap},
        ]
        output = ROOT / f"output/kaggle/tts010_{arm}.ipynb"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
        outputs.append(output)
        print(output)
    return outputs


if __name__ == "__main__":
    build()
