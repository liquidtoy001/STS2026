"""LibriTTS-R -> a speaker-manifest corpus for the notebook (docs/SPEAKER_CONDITIONING.md).

Reads the extracted LibriTTS-R subsets (<root>/<subset>/<speaker>/<chapter>/<id>.wav, 24 kHz,
with <id>.normalized.txt) and writes into --out:

  wavs/<id>.wav            mono 22,050 Hz 16-bit PCM (resampled with torchaudio; shared by every
                           manifest written into the same folder, existing files are checked and reused)
  speakers_<name>.csv      file_name|normalized_transcription|speaker_name (speaker_name = LibriTTS reader ID)
  prepare_<name>.json      what was kept and dropped, and why; resampling; hashes

Utterances are dropped when their WAV has no .normalized.txt, when they are on the LibriTTS-R list of failed speech restorations
(recommended by the corpus authors), when their text has characters outside the notebook's
character table after the replacements in TEXT_REPLACEMENTS, or when they are shorter than
--min-seconds or longer than --max-seconds.

--heldout FILE: test utterances the notebook keeps out of training. A missing FILE is created by
drawing one utterance from each of --heldout-count speakers of train-clean-100 (seeded), so the
same test set fits every manifest that contains train-clean-100; an existing FILE is checked
against the manifest.

python tools/prepare_libritts_r.py --root /scratch/libritts_r/LibriTTS_R --out /scratch/libritts_r/corpus \
    --subsets train-clean-100 --name train-clean-100 \
    --failed-lists /scratch/libritts_r/libritts_r_failed_speech_restoration_examples \
    --heldout sts_eval/splits/libritts_r_heldout_100.txt
"""

import argparse
import hashlib
import json
import multiprocessing
import random
import re
import sys
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

SOURCE_RATE = 24_000
TARGET_RATE = 22_050
# The notebook's character table (token_table("chars") without the pad symbol).
VOCAB = set(";:,.!?¡¿—…\"«»“”ü ()-'[]" + "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")
# Typographic variants of characters the table has; anything else outside the table drops the utterance.
TEXT_REPLACEMENTS = {"’": "'", "‘": "'", "–": "—", " ": " "}
FORMAT = "sts2026-libritts-r-corpus-v1"


def clean_text(text):
    """(cleaned text, characters outside the table): NFC, TEXT_REPLACEMENTS, single spaces."""
    text = unicodedata.normalize("NFC", text)
    for old, new in TEXT_REPLACEMENTS.items():
        text = text.replace(old, new)
    text = " ".join(text.split())
    return text, sorted({c for c in text if c not in VOCAB})


def read_failed_lists(folder, subsets):
    """Utterance IDs on the LibriTTS-R failed-restoration lists of these subsets."""
    failed = set()
    for subset in subsets:
        path = Path(folder) / f"{subset}_bad_sample_list.txt"
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                failed.add(Path(line.strip()).stem)
    return failed


def scan(root, subsets):
    """([(subset, speaker, utterance_id, wav_path, normalized text)] sorted by ID, IDs of WAVs without text)."""
    items, no_text = [], []
    for subset in subsets:
        folder = Path(root) / subset
        if not folder.is_dir():
            raise FileNotFoundError(f"{folder} is missing; extract the {subset} tarball first.")
        for wav in folder.glob("*/*/*.wav"):
            text_path = wav.with_suffix(".normalized.txt")
            if not text_path.is_file():
                no_text.append(wav.stem)
                continue
            items.append((subset, wav.parent.parent.name, wav.stem, wav,
                          text_path.read_text(encoding="utf-8")))
    return sorted(items, key=lambda item: item[2]), sorted(no_text)


def convert(job):
    """Resample one recording to TARGET_RATE (reusing a valid existing output); returns its record."""
    import numpy as np
    import soundfile
    import torch
    import torchaudio
    source, target = Path(job[0]), Path(job[1])
    info = soundfile.info(str(source))
    if info.samplerate != SOURCE_RATE or info.channels != 1:
        raise ValueError(f"{source} is {info.samplerate} Hz, {info.channels} channels; expected {SOURCE_RATE} Hz mono.")
    expected = -(-info.frames * TARGET_RATE // SOURCE_RATE)
    if target.is_file():
        existing = soundfile.info(str(target))
        if existing.samplerate == TARGET_RATE and existing.frames == expected and existing.subtype == "PCM_16":
            return {"id": source.stem, "seconds": expected / TARGET_RATE, "reused": True, "clipped": 0}
    audio, _ = soundfile.read(str(source), dtype="float32")
    resampled = torchaudio.functional.resample(torch.from_numpy(audio), SOURCE_RATE, TARGET_RATE).numpy()
    if resampled.shape[0] != expected:
        raise AssertionError(f"{source}: {resampled.shape[0]} samples after resampling, expected {expected}.")
    limit = 32767 / 32768
    clipped = int(np.count_nonzero(np.abs(resampled) > limit))
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(".partial.wav")
    soundfile.write(str(partial), np.clip(resampled, -1.0, limit), TARGET_RATE, subtype="PCM_16")
    partial.replace(target)
    return {"id": source.stem, "seconds": expected / TARGET_RATE, "reused": False, "clipped": clipped}


def draw_heldout(kept, count, seed):
    """One utterance from each of `count` train-clean-100 speakers, drawn with a seeded RNG; sorted IDs.

    kept: [(subset, speaker, utterance_id, seconds, words)]. Candidates have 2-10 s and >= 5 words,
    so every test utterance is a full sentence of ordinary length.
    """
    by_speaker = {}
    for subset, speaker, utterance, seconds, words in kept:
        if subset == "train-clean-100" and 2.0 <= seconds <= 10.0 and words >= 5:
            by_speaker.setdefault(speaker, []).append(utterance)
    if len(by_speaker) < count:
        raise ValueError(f"Only {len(by_speaker)} train-clean-100 speakers have test candidates; {count} requested.")
    rng = random.Random(seed)
    speakers = rng.sample(sorted(by_speaker, key=int), count)
    return sorted(rng.choice(sorted(by_speaker[speaker])) for speaker in speakers)


def percentiles(values, points=(0, 5, 25, 50, 75, 95, 100)):
    ordered = sorted(values)
    return {f"p{p}": round(ordered[min(len(ordered) - 1, round(p / 100 * (len(ordered) - 1)))], 3)
            for p in points} if ordered else {}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", required=True, help="extracted LibriTTS_R folder containing the subset folders")
    parser.add_argument("--out", required=True, help="corpus folder (wavs/ and the manifests)")
    parser.add_argument("--subsets", nargs="+", required=True, help="e.g. train-clean-100 train-clean-360")
    parser.add_argument("--name", required=True, help="manifest name, e.g. train-clean-100")
    parser.add_argument("--failed-lists", required=True,
                        help="folder of <subset>_bad_sample_list.txt (libritts_r_failed_speech_restoration_examples)")
    parser.add_argument("--min-seconds", type=float, default=1.0)
    parser.add_argument("--max-seconds", type=float, default=15.0)
    parser.add_argument("--heldout", default=None, help="test-set ID file to create or check (see above)")
    parser.add_argument("--heldout-count", type=int, default=100)
    parser.add_argument("--heldout-seed", type=int, default=2026)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    import numpy
    import soundfile
    import torch
    import torchaudio

    out = Path(args.out)
    (out / "wavs").mkdir(parents=True, exist_ok=True)
    failed = read_failed_lists(args.failed_lists, args.subsets)
    items, no_text = scan(args.root, args.subsets)
    print(f"{len(items)} utterances in {args.subsets}; {len(failed)} on the failed-restoration lists", flush=True)

    dropped = {"failed_restoration": [], "characters": {}, "too_short": [], "too_long": []}
    kept, rows, jobs = [], [], []
    for subset, speaker, utterance, wav, raw_text in items:
        if utterance in failed:
            dropped["failed_restoration"].append(utterance)
            continue
        text, outside = clean_text(raw_text)
        if outside or not text:
            for char in outside or ["<empty>"]:
                dropped["characters"].setdefault(char, []).append(utterance)
            continue
        seconds = -(-soundfile.info(str(wav)).frames * TARGET_RATE // SOURCE_RATE) / TARGET_RATE
        if seconds < args.min_seconds:
            dropped["too_short"].append(utterance)
        elif seconds > args.max_seconds:
            dropped["too_long"].append(utterance)
        else:
            kept.append((subset, speaker, utterance, seconds, len(text.split())))
            rows.append(f"{utterance}|{text}|{speaker}")
            jobs.append((str(wav), str(out / "wavs" / f"{utterance}.wav")))
    rejected_by_text = {u for ids in dropped["characters"].values() for u in ids}
    print(f"{len(kept)} utterances kept; converting their recordings", flush=True)

    records = {}
    with multiprocessing.Pool(args.workers) as pool:
        for index, record in enumerate(pool.imap_unordered(convert, jobs, chunksize=64), 1):
            records[record["id"]] = record
            if index % 5000 == 0 or index == len(jobs):
                print(f"{index}/{len(jobs)} recordings", flush=True)

    # wavs/ may hold recordings of other manifests written into the same folder; the manifest decides.
    manifest = out / f"speakers_{args.name}.csv"
    manifest.write_text("file_name|normalized_transcription|speaker_name\n" + "\n".join(rows) + "\n",
                        encoding="utf-8", newline="\n")

    heldout = None
    if args.heldout:
        heldout_path = Path(args.heldout)
        kept_ids = {utterance for _, _, utterance, _, _ in kept}
        if heldout_path.is_file():
            ids = [line.strip() for line in heldout_path.read_text(encoding="utf-8").splitlines()
                   if line.strip() and not line.startswith("#")]
            missing = [name for name in ids if name not in kept_ids]
            if missing:
                raise ValueError(f"{len(missing)} test utterances of {heldout_path} are not in the manifest: {missing[:3]}")
            action = "checked"
        else:
            ids = draw_heldout(kept, args.heldout_count, args.heldout_seed)
            heldout_path.write_text(
                f"# LibriTTS-R test utterances: one from each of {args.heldout_count} train-clean-100 speakers\n"
                f"# (2-10 s, >= 5 words), drawn by tools/prepare_libritts_r.py with seed {args.heldout_seed}.\n"
                + "\n".join(ids) + "\n", encoding="utf-8", newline="\n")
            action = "created"
        heldout = {"path": str(heldout_path), "action": action, "utterances": len(ids),
                   "speakers": len({u.split("_")[0] for u in ids}),
                   "sha256": hashlib.sha256(heldout_path.read_bytes()).hexdigest()}
        print(f"test set {action}: {heldout_path} ({len(ids)} utterances)", flush=True)

    speakers = {}
    for _, speaker, _, seconds, _ in kept:
        speakers.setdefault(speaker, []).append(seconds)
    seconds = [s for _, _, _, s, _ in kept]
    summary = {
        "format": FORMAT,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "command": sys.argv,
        "subsets": args.subsets,
        "manifest": {"path": str(manifest), "sha256": hashlib.sha256(manifest.read_bytes()).hexdigest()},
        "heldout": heldout,
        "audio": {"source_rate": SOURCE_RATE, "target_rate": TARGET_RATE, "subtype": "PCM_16",
                  "resampler": "torchaudio.functional.resample defaults (sinc_interp_hann, width 6, rolloff 0.99)",
                  "recordings_converted": sum(not r["reused"] for r in records.values()),
                  "recordings_reused": sum(r["reused"] for r in records.values()),
                  "clipped_samples": sum(r["clipped"] for r in records.values()),
                  "recordings_with_clipping": sum(r["clipped"] > 0 for r in records.values())},
        "text": {"replacements": TEXT_REPLACEMENTS, "vocabulary": "notebook token_table('chars')"},
        "filters": {"min_seconds": args.min_seconds, "max_seconds": args.max_seconds},
        "counts": {"scanned": len(items) + len(no_text), "no_text": len(no_text), "failed_restoration": len(dropped["failed_restoration"]),
                   "characters": len(rejected_by_text), "too_short": len(dropped["too_short"]),
                   "too_long": len(dropped["too_long"]), "kept": len(kept)},
        "kept": {"hours": round(sum(seconds) / 3600, 2), "speakers": len(speakers),
                 "seconds": percentiles(seconds),
                 "utterances_per_speaker": percentiles([len(v) for v in speakers.values()]),
                 "minutes_per_speaker": percentiles([sum(v) / 60 for v in speakers.values()]),
                 "by_subset": {subset: sum(1 for k in kept if k[0] == subset) for subset in args.subsets}},
        "dropped_characters": {char: {"utterances": len(ids), "examples": ids[:5]}
                               for char, ids in sorted(dropped["characters"].items(), key=lambda kv: -len(kv[1]))},
        "dropped_examples": {"no_text": no_text[:20],
                             **{key: dropped[key][:20] for key in ("failed_restoration", "too_short", "too_long")}},
        "versions": {"python": sys.version.split()[0], "torch": torch.__version__,
                     "torchaudio": torchaudio.__version__, "soundfile": soundfile.__version__,
                     "numpy": numpy.__version__},
    }
    report = out / f"prepare_{args.name}.json"
    report.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("counts", "kept")}, indent=2))
    print(f"{manifest}\n{report}")


if __name__ == "__main__":
    sys.exit(main())
