"""Train one TTS_0.10 ablation arm without Jupyter (GPU servers, Windows, tmux/nohup/Slurm).

Executes every code cell of notebooks/vits-full-train.ipynb in this process, exactly as
"Run All" would, with the arm and paths given here. Example:

    python tools/train_tts010.py --arm G1 --data-dir /data/LJSpeech-1.1

The run folder is printed at the start and written under --run-root
(default output/full-train/). Resume a stopped run with --resume <run folder>.

Fine-tune: start a NEW run from a finished checkpoint of the same arm, here with chunk-aware
training (durations from tools/chunk_alignments.py); --steps counts the new run's steps
(one command):

    python tools/train_tts010.py --arm R0 --data-dir /data/LJSpeech-1.1 --tag ftchunk
        --init-from R0/checkpoints/vits_step_40000.pt --steps 5000
        --chunk-prob 0.5 --chunk-durations chunk_durations_R0_40k.json
"""

import argparse
import os
import sys
from pathlib import Path

# Before the notebook's pyplot import below: an interactive backend would open the final plot
# in a window and block the end of the run until it is closed.
os.environ["MPLBACKEND"] = "Agg"
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
    parser.add_argument("--steps", type=int, default=None, help="training steps of this run (default 40000)")
    parser.add_argument("--keep-every", type=int, default=None,
                        help="keep the checkpoints at multiples of this step (default 10000; the latest is always kept)")
    parser.add_argument("--tag", default="", help="added to the run folder name, e.g. ftchunk")
    parser.add_argument("--init-from", default=None,
                        help="checkpoint of a finished run of the same arm: start a NEW run from its weights")
    parser.add_argument("--fresh-optimizer", action="store_true",
                        help="with --init-from: new optimizer moments and fp16 scaler instead of the checkpoint's")
    parser.add_argument("--chunk-prob", type=float, default=0.0,
                        help="chunk-aware training: probability of cropping a clip to whole words (default 0 = off)")
    parser.add_argument("--chunk-durations", default=None, help="durations file from tools/chunk_alignments.py")
    parser.add_argument("--chunk-words", default="1-8", help="words per crop, MIN-MAX (default 1-8)")
    parser.add_argument("--chunk-prefix-share", type=float, default=0.5,
                        help="share of crops that start at the first word (default 0.5)")
    parser.add_argument("--chunk-boundaries", choices=("words", "punctuation"), default="words",
                        help="TTS_0.20 option: end crops after any word (default) or at punctuation")
    parser.add_argument("--phoneme-cache", default=None,
                        help="phoneme cache of tools/phonemize.py for arms with phoneme input "
                             "(default sts_eval/phonemes/ljspeech_g2p_en.tsv)")
    args = parser.parse_args()
    if args.resume and args.init_from:
        parser.error("--resume continues a run with its own initialisation; do not combine it with --init-from")
    if args.fresh_optimizer and not args.init_from:
        parser.error("--fresh-optimizer needs --init-from")
    if (args.steps or args.keep_every) and args.smoke_steps:
        parser.error("--smoke-steps sets its own step and checkpoint counts; do not add --steps or --keep-every")
    if args.chunk_prob > 0 and not args.chunk_durations:
        parser.error("--chunk-prob needs --chunk-durations")
    if args.chunk_boundaries != "words" and args.chunk_prob <= 0:
        parser.error("--chunk-boundaries needs --chunk-prob")

    for key in [key for key in os.environ if key.startswith("STS_")]:
        del os.environ[key]  # never inherit a stale arm, resume path or smoke-test setting
    os.environ.update(
        STS_ARM=args.arm, STS_DATA_DIR=str(Path(args.data_dir).resolve()), STS_REPO_ROOT=str(ROOT),
        STS_RUN_ROOT=str(Path(args.run_root).resolve()), STS_NUM_WORKERS=str(args.num_workers),
        STS_TIME_BUDGET_HOURS=str(args.time_budget_hours), MPLBACKEND="Agg",
    )
    if args.resume:
        os.environ["STS_RESUME_FROM"] = str(Path(args.resume).resolve())
    if args.init_from:
        os.environ.update(STS_INIT_FROM=str(Path(args.init_from).resolve()),
                          STS_INIT_OPTIMIZER="fresh" if args.fresh_optimizer else "carry")
    if args.steps:
        os.environ["STS_TOTAL_STEPS"] = str(args.steps)
    if args.keep_every:
        os.environ["STS_KEEP_EVERY"] = str(args.keep_every)
    if args.tag:
        os.environ["STS_RUN_TAG"] = args.tag
    if args.chunk_prob > 0:
        os.environ.update(STS_CHUNK_PROB=str(args.chunk_prob), STS_CHUNK_WORDS=args.chunk_words,
                          STS_CHUNK_DURATIONS=str(Path(args.chunk_durations).resolve()),
                          STS_CHUNK_PREFIX_SHARE=str(args.chunk_prefix_share),
                          STS_CHUNK_BOUNDARIES=args.chunk_boundaries)
    if args.phoneme_cache:
        os.environ["STS_PHONEME_CACHE"] = str(Path(args.phoneme_cache).resolve())
    if args.smoke_steps:
        every = str(max(1, args.smoke_steps // 2))
        os.environ.update(STS_ALLOW_SUBSET="1", STS_TOTAL_STEPS=str(args.smoke_steps), STS_EVAL_EVERY=every,
                          STS_SAVE_EVERY=every, STS_KEEP_EVERY=every, STS_LOG_EVERY="1")
    namespace = globals()
    for index, cell in enumerate(code_cells()):
        exec(compile(cell, f"notebook_cell_{index}", "exec"), namespace)


if __name__ == "__main__":
    main()
