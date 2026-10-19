"""Per-character durations of every LJSpeech clip from a trained TTS_0.10 checkpoint's MAS.

Chunk-aware training (tools/train_tts010.py --chunk-prob) crops clips at word boundaries; the
audio cut points come from this file. For each clip the alignment of the training forward pass
is recomputed without sampling: text-encoder prior, posterior MEAN through the flow, the
Gaussian alignment score and the notebook's monotonic alignment search. durations[i] is the
number of spectrogram frames of encoded token i; the durations of a clip sum to its frame count.
Held-out clips (sts_eval/splits/ljspeech_heldout_100.txt) are skipped.

python tools/chunk_alignments.py --checkpoint R0/checkpoints/vits_step_40000.pt \
    --data-dir /data/LJSpeech-1.1 --out chunk_durations_R0_40k.json
"""

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.eval_tts010 import load_model  # noqa: E402
from tools.notebook_defs import literal_assignment, load_definitions  # noqa: E402

FORMAT = "sts2026-chunk-durations-v1"  # checked by load_chunk_durations in the notebook
ALIGNMENT = "MAS on the posterior mean (no sampling), eval mode, fp32"


@torch.no_grad()
def alignment_durations(ns, model, batch):
    """[(file-order) list of per-token duration lists] for one collated batch."""
    text, text_lengths, spectrogram, spectrogram_lengths = batch[:4]
    _, prior_mean, prior_log_scale, text_mask = model.text_encoder(text, text_lengths)
    _, posterior_mean, _, spectrogram_mask = model.posterior_encoder(spectrogram, spectrogram_lengths)
    z_prior = model.flow(posterior_mean * spectrogram_mask, spectrogram_mask)
    score = ns["gaussian_alignment_score"](z_prior.float(), prior_mean.float(), prior_log_scale.float())
    mask = spectrogram_mask.float().transpose(1, 2) * text_mask.float()
    durations = ns["maximum_path"](score, mask).sum(dim=1).long().cpu()
    return [durations[i, :int(text_lengths[i])].tolist() for i in range(durations.size(0))]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    ns = load_definitions()
    ns["PAD_ID"] = 0
    ns["symbol_to_id"] = {s: i for i, s in enumerate(
        ["_"] + list(literal_assignment("PUNCTUATION")) + list(literal_assignment("LETTERS")))}
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model, checkpoint, _ = load_model(ns, args.checkpoint, device)
    data_dir = Path(args.data_dir)
    metadata = data_dir / "metadata.csv"
    corpus, _ = ns["load_training_metadata"](metadata)
    heldout = set(ns["load_id_list"](ROOT / "sts_eval" / "splits" / "ljspeech_heldout_100.txt"))
    frame = corpus[~corpus["file_name"].isin(heldout)]
    missing = [name for name in frame["file_name"] if not (data_dir / "wavs" / f"{name}.wav").is_file()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} recordings are missing, e.g. {missing[:3]}.")
    # Similar lengths in one batch keep padding (and MAS time) small.
    frame = frame.assign(length=frame["normalized_transcription"].str.len()).sort_values(
        ["length", "file_name"]).reset_index(drop=True)
    dataset = ns["LJSpeechDataset"](frame, data_dir / "wavs")
    names = frame["file_name"].tolist()
    durations, started = {}, time.perf_counter()
    for start in range(0, len(names), args.batch_size):
        indices = range(start, min(start + args.batch_size, len(names)))
        batch = ns["collate_tts_batch"]([dataset[i] for i in indices])
        batch = tuple(item.to(device) for item in batch)
        for index, values in zip(indices, alignment_durations(ns, model, batch)):
            durations[names[index]] = values
        if (start // args.batch_size) % 100 == 0:
            print(f"{len(durations)}/{len(names)} clips, {time.perf_counter() - started:.0f} s", flush=True)

    checkpoint_path = Path(args.checkpoint).resolve()
    record = {
        "format": FORMAT, "alignment": ALIGNMENT, "hop_length": 256,
        "source_checkpoint": str(checkpoint_path),
        "source_checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
        "source_step": checkpoint.get("step"), "switches": checkpoint.get("switches"),
        "metadata_sha256": hashlib.sha256(metadata.read_bytes()).hexdigest(),
        "heldout_skipped": len(heldout & set(corpus["file_name"])),
        "device": str(device), "torch": torch.__version__,
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "durations": {name: durations[name] for name in sorted(durations)},
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, separators=(",", ":")) + "\n", encoding="utf-8", newline="\n")
    print(f"{len(durations)} clips in {time.perf_counter() - started:.0f} s ->", out,
          "sha256", hashlib.sha256(out.read_bytes()).hexdigest())


if __name__ == "__main__":
    main()
