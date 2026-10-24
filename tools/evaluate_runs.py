"""Central evaluation of every finished TTS_0.10 run in a folder, then one comparison table.

Finds run folders (any depth, e.g. a Drive download `tts010-ablation/<arm>/<run>/`) whose
status.json says `completed` at --step and that hold `checkpoints/vits_step_<step>.pt`. Each one
not yet evaluated gets tools/eval_tts010.py and then tools/asr_wer.py (Whisper, run with
--asr-python). Finished work is skipped, so the command can simply be re-run after new runs
arrive or after an interruption. Extra reference checkpoints can be added with --extra.

The table (summary.md, summary.csv, summary.json in --eval-dir; summary_long.* for --split
ljspeech_long_100.txt) lists every evaluated label of that test set, its
difference from R0, and whether that difference exceeds the R0/R0b difference (the seed noise
floor). One seed per arm: "beyond noise floor" is a screening flag, not a significance test.

python tools/evaluate_runs.py --runs-dir output/tts010-turino/runs --eval-dir output/tts010-turino/eval \
    --data-dir data/LJSpeech-1.1 --asr-python C:/.../Python310/python.exe
"""

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.notebook_defs import literal_assignment  # noqa: E402

ORDER = ["R0", "R0b", "G1", "G2", "G3", "A1", "A2", "L1", "O1", "D1", "E0", "T011"]
# (key, header, lower is better)
METRICS = [
    ("full_log_mel_l1", "Held-out reconstruction log-mel L1", True),
    ("full_spectral_convergence", "Held-out reconstruction STFT SC", True),
    ("wer", "Whisper WER", True),
    ("cer", "Whisper CER", True),
    ("delta_wer", "ΔWER vs natural", True),
    ("empty_outputs", "Empty ASR outputs", True),
    ("duration_ratio", "Duration ratio (1 = natural)", None),
    ("rms_ratio", "Level ratio (1 = natural)", None),
]


def find_runs(runs_dir, step):
    runs = []
    for config_path in sorted(Path(runs_dir).glob("**/run_config.json")):
        run = config_path.parent
        status_path = run / "status.json"
        checkpoint = run / "checkpoints" / f"vits_step_{step}.pt"
        if not status_path.is_file():
            continue
        config = json.loads(config_path.read_text(encoding="utf-8"))
        status = json.loads(status_path.read_text(encoding="utf-8"))
        if config.get("model_version") != "TTS_0.10" or not checkpoint.is_file():
            continue
        if status.get("status") != "completed" or status.get("step") != step:
            continue
        runs.append({"arm": config["arm"], "run": run, "checkpoint": checkpoint})
    return runs


DEFAULT_SPLIT = "ljspeech_heldout_100.txt"


def split_suffix(split):
    """'' for the default held-out split, else e.g. '_long' for ljspeech_long_100.txt."""
    if split == DEFAULT_SPLIT:
        return ""
    return "_" + Path(split).stem.removeprefix("ljspeech_").removesuffix("_100")


def label_for(run, eval_dir, step, split=DEFAULT_SPLIT):
    """<arm>_<k>k[_<split>], or with the run's time stamp when another run of that arm already took it."""
    base = f"{run['arm']}_{step // 1000}k{split_suffix(split)}"
    existing = eval_dir / base / "metrics.json"
    if existing.is_file():
        recorded = json.loads(existing.read_text(encoding="utf-8")).get("checkpoint")
        if recorded and Path(recorded).resolve() != run["checkpoint"].resolve():
            return f"{base}_{run['run'].name.rsplit('_', 1)[-1]}"
    return base


def evaluate(label, checkpoint, args):
    out = Path(args.eval_dir) / label
    if not (out / "metrics.json").is_file():
        if out.exists():
            raise SystemExit(f"{out} exists without metrics.json (interrupted?): remove it and re-run.")
        print(f"[eval] {label}", flush=True)
        subprocess.run([sys.executable, str(ROOT / "tools" / "eval_tts010.py"), "--checkpoint", str(checkpoint),
                        "--label", label, "--data-dir", args.data_dir, "--out", args.eval_dir,
                        "--split", args.split], check=True)
    if not args.skip_asr and not (out / "asr.json").is_file():
        print(f"[asr]  {label}", flush=True)
        subprocess.run([args.asr_python, str(ROOT / "tools" / "asr_wer.py"), str(out), "--model", args.asr_model],
                       check=True)


def read_label(directory):
    metrics = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
    row = {"label": directory.name, "step": metrics.get("checkpoint_step"),
           "full_log_mel_l1": metrics["heldout_full"]["log_mel_l1"],
           "full_spectral_convergence": metrics["heldout_full"]["stft_spectral_convergence"],
           "duration_ratio": metrics["heldout_tts"]["duration_ratio_mean"],
           "rms_ratio": metrics["heldout_tts"]["rms_ratio_mean"],
           "heldout_count": metrics["heldout_full"]["count"], "split": metrics.get("split", DEFAULT_SPLIT)}
    switches = metrics.get("switches") or {}
    arm_config = Path(metrics["checkpoint"]).parents[1] / "run_config.json"
    if arm_config.is_file():
        row["arm"] = json.loads(arm_config.read_text(encoding="utf-8")).get("arm")
    else:  # run folder moved since evaluation: fall back to the label, e.g. "G1_40k"
        prefix = directory.name.split("_", 1)[0]
        row["arm"] = prefix if prefix in literal_assignment("ARMS") and switches else None
    row["switches"] = switches
    asr_path = directory / "asr.json"
    if asr_path.is_file():
        asr = json.loads(asr_path.read_text(encoding="utf-8"))
        summary = asr["summary"]
        row.update(wer=summary["synthesized"]["wer"]["rate"], cer=summary["synthesized"]["cer"]["rate"],
                   delta_wer=summary.get("delta_wer"), empty_outputs=summary["synthesized"]["failures"],
                   natural_wer=summary["natural"]["wer"]["rate"], asr_model=asr["asr"]["model"])
    return row


def build_table(eval_dir, split=DEFAULT_SPLIT):
    """Rows of one test set only, so R0 and R0b are compared on the same sentences."""
    arms = literal_assignment("ARMS")
    rows = [read_label(d) for d in sorted(Path(eval_dir).iterdir()) if (d / "metrics.json").is_file()]
    rows = [row for row in rows if row["split"] == split]
    rank = {arm: i for i, arm in enumerate(ORDER)}
    rows.sort(key=lambda r: (rank.get(r["arm"], len(ORDER)), r["label"]))
    by_arm = {r["arm"]: r for r in rows if r["arm"]}
    reference, seed = by_arm.get("R0"), by_arm.get("R0b")
    for row in rows:
        spec = arms.get(row["arm"]) if row["arm"] else None
        row["description"] = spec["title"] if spec else "reference checkpoint (not a TTS_0.10 run)"
        for key, _, _ in METRICS:
            value = row.get(key)
            if reference is None or value is None or reference.get(key) is None:
                continue
            row[f"{key}_vs_R0"] = value - reference[key]
            if seed is not None and seed.get(key) is not None and row["arm"] not in ("R0", "R0b"):
                floor = abs(seed[key] - reference[key])
                row[f"{key}_beyond_noise"] = abs(value - reference[key]) > floor
    return rows, reference is not None, seed is not None


def fmt(value, key):
    if value is None:
        return "–"
    if key in ("wer", "cer"):
        return f"{100 * value:.1f}%"
    if key == "delta_wer":
        return f"{100 * value:+.1f} pt"
    if key == "empty_outputs":
        return str(value)
    return f"{value:.3f}"


def write_outputs(rows, has_r0, has_r0b, eval_dir, split=DEFAULT_SPLIT):
    eval_dir = Path(eval_dir)
    stem = "summary" + split_suffix(split)
    lines = ["# TTS_0.10 central evaluation", "",
             f"Generated {datetime.now(timezone.utc).isoformat(timespec='seconds')} from {eval_dir}. "
             f"Held-out set: sts_eval/splits/{split}; reference checkpoints are not TTS_0.10 runs.", ""]
    header = ["Label", "Arm", "What it is"] + [h for _, h, _ in METRICS]
    lines += ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for row in rows:
        cells = [row["label"], row["arm"] or "–", row["description"]] + [fmt(row.get(k), k) for k, _, _ in METRICS]
        lines.append("| " + " | ".join(cells) + " |")
    if has_r0:
        lines += ["", "## Difference from R0", "",
                  "Positive = higher than R0. `*` = larger than the R0/R0b difference (seed noise floor)."
                  if has_r0b else "Positive = higher than R0. The noise floor needs R0b, which is not evaluated yet.", ""]
        header = ["Label", "Arm"] + [h for _, h, _ in METRICS]
        lines += ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
        for row in rows:
            if row["arm"] == "R0":
                continue
            cells = [row["label"], row["arm"] or "–"]
            for key, _, _ in METRICS:
                delta = row.get(f"{key}_vs_R0")
                text = "–" if delta is None else (f"{100 * delta:+.1f} pt" if key in ("wer", "cer", "delta_wer")
                                                 else f"{delta:+d}" if key == "empty_outputs" else f"{delta:+.3f}")
                cells.append(text + ("*" if row.get(f"{key}_beyond_noise") else ""))
            lines.append("| " + " | ".join(cells) + " |")
    else:
        lines += ["", "R0 is not evaluated yet, so no differences are shown."]
    (eval_dir / f"{stem}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    keys = ["label", "arm", "description", "step", "heldout_count", "asr_model", "natural_wer"] + \
           [k for k, _, _ in METRICS] + [f"{k}_vs_R0" for k, _, _ in METRICS] + [f"{k}_beyond_noise" for k, _, _ in METRICS]
    with open(eval_dir / f"{stem}.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    (eval_dir / f"{stem}.json").write_text(json.dumps(rows, indent=2, default=str) + "\n", encoding="utf-8")
    return eval_dir / f"{stem}.md"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs-dir", help="folder searched (recursively) for run folders")
    parser.add_argument("--eval-dir", required=True)
    parser.add_argument("--data-dir", help="complete LJSpeech-1.1 (needed unless --table-only)")
    parser.add_argument("--step", type=int, default=40000)
    parser.add_argument("--split", default=DEFAULT_SPLIT,
                        help="test set in sts_eval/splits/; labels get a suffix, e.g. G1_40k_long")
    parser.add_argument("--extra", action="append", default=[], metavar="LABEL=CHECKPOINT",
                        help="reference checkpoint to evaluate as well (repeatable)")
    parser.add_argument("--asr-python", default=sys.executable, help="Python with openai-whisper installed")
    parser.add_argument("--asr-model", default="medium")
    parser.add_argument("--skip-asr", action="store_true")
    parser.add_argument("--table-only", action="store_true", help="only rebuild the summary table")
    parser.add_argument("--dry-run", action="store_true", help="list what would be evaluated")
    args = parser.parse_args()
    eval_dir = Path(args.eval_dir)
    eval_dir.mkdir(parents=True, exist_ok=True)

    if not args.table_only:
        if not args.data_dir:
            parser.error("--data-dir is required unless --table-only")
        jobs = []
        for run in find_runs(args.runs_dir, args.step) if args.runs_dir else []:
            jobs.append((label_for(run, eval_dir, args.step, args.split), run["checkpoint"]))
        for item in args.extra:
            label, _, checkpoint = item.partition("=")
            jobs.append((label, Path(checkpoint)))
        for label, checkpoint in jobs:
            out = eval_dir / label
            done = (out / "metrics.json").is_file() and (args.skip_asr or (out / "asr.json").is_file())
            print(f"{'done ' if done else 'todo '} {label:24s} {checkpoint}", flush=True)
            if not done and not args.dry_run:
                evaluate(label, checkpoint, args)
        if args.dry_run:
            return
    rows, has_r0, has_r0b = build_table(eval_dir, args.split)
    print(write_outputs(rows, has_r0, has_r0b, eval_dir, args.split))


if __name__ == "__main__":
    main()
