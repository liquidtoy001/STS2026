"""Train one TTS_0.10 ablation arm without Jupyter (GPU servers, Windows, tmux/nohup/Slurm).

Executes every code cell of notebooks/vits-full-train.ipynb in this process, exactly as
"Run All" would, with the arm and paths given here. Example:

    python tools/train_tts010.py --arm G1 --data-dir /data/LJSpeech-1.1

The run folder is printed at the start and written under --run-root
(default output/full-train/). Resume a stopped run with --resume <run folder>.
"""

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.notebook_defs import code_cells, literal_assignment, load_definitions  # noqa: E402

# DataLoader workers started with "spawn" (Windows, macOS) import this module and unpickle the
# dataset; the notebook classes must therefore exist at module level here.
load_definitions(globals())


def main():
    arms = literal_assignment("ARMS")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arm", required=True, choices=sorted(arms))
    parser.add_argument("--data-dir", required=True, help="LJSpeech-1.1 folder with metadata.csv and wavs/")
    parser.add_argument("--run-root", default=str(ROOT / "output" / "full-train"))
    parser.add_argument("--resume", default=None, help="run folder of a stopped run of the same arm")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--time-budget-hours", type=float, default=11.3)
    parser.add_argument("--smoke-steps", type=int, default=None,
                        help="SMOKE TEST ONLY: train this many steps on whatever clips exist")
    args = parser.parse_args()

    for key in [key for key in os.environ if key.startswith("STS_")]:
        del os.environ[key]  # never inherit a stale arm, resume path or smoke-test setting
    os.environ.update(
        STS_ARM=args.arm, STS_DATA_DIR=str(Path(args.data_dir).resolve()), STS_REPO_ROOT=str(ROOT),
        STS_RUN_ROOT=str(Path(args.run_root).resolve()), STS_NUM_WORKERS=str(args.num_workers),
        STS_TIME_BUDGET_HOURS=str(args.time_budget_hours), MPLBACKEND="Agg",
    )
    if args.resume:
        os.environ["STS_RESUME_FROM"] = str(Path(args.resume).resolve())
    if args.smoke_steps:
        every = str(max(1, args.smoke_steps // 2))
        os.environ.update(STS_ALLOW_SUBSET="1", STS_TOTAL_STEPS=str(args.smoke_steps), STS_EVAL_EVERY=every,
                          STS_SAVE_EVERY=every, STS_KEEP_EVERY=every, STS_LOG_EVERY="1")
    namespace = globals()
    for index, cell in enumerate(code_cells()):
        exec(compile(cell, f"notebook_cell_{index}", "exec"), namespace)


if __name__ == "__main__":
    main()
