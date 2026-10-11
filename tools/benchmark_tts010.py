"""Measure training-step time and GPU memory of TTS_0.10 arms on this machine.

Uses the notebook's own model, discriminator, loss and trainer for each arm, a real DataLoader,
and synthetic audio with LJSpeech-like lengths: real normalized texts from metadata.csv (after
the arm's length filter and held-out exclusion), durations of chars / 15.4 s x U(0.85, 1.15).
Compute depends on lengths, not content. It does not train a usable model.

python tools/benchmark_tts010.py --metadata /data/LJSpeech-1.1/metadata.csv --arms R0 E0
"""

import argparse
import json
import math
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.notebook_defs import literal_assignment, load_definitions  # noqa: E402

# Spawned DataLoader workers import this module; the notebook classes must exist here.
load_definitions(globals())
CHARS_PER_SECOND = 15.4


class SyntheticSpeech(torch.utils.data.Dataset):
    def __init__(self, texts, table, seed=0):
        rng = np.random.default_rng(seed)
        self.texts = list(texts)
        self.table = dict(table)
        self.samples = [int(max(1.0, len(t) / CHARS_PER_SECOND * rng.uniform(0.85, 1.15)) * 22050)
                        for t in self.texts]

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        generator = torch.Generator().manual_seed(index)
        audio = (torch.randn(1, self.samples[index], generator=generator) * 0.1).clamp(-1, 1)
        return encode_text(self.texts[index], self.table), waveform_to_spectrogram(audio), audio  # noqa: F821


def arm_settings(arm):
    arms = literal_assignment("ARMS")
    new, old = literal_assignment("TTS010_SWITCHES"), literal_assignment("TTS002_SWITCHES")
    switches = {name: (old if name in arms[arm]["revert"] else new)[name] for name in new}
    return switches, literal_assignment("OPTIM_PRESETS")[switches["optim"]]


def benchmark(arm, texts_all, heldout, table, steps, warmup, workers, device):
    switches, preset = arm_settings(arm)
    texts = [t for name, t in texts_all if name not in heldout
             and (not switches["max_text_chars"] or 1 <= len(t) <= switches["max_text_chars"])]
    loader = torch.utils.data.DataLoader(
        SyntheticSpeech(texts, table), batch_size=preset["batch_size"], shuffle=True,
        collate_fn=collate_tts_batch_partial, num_workers=workers, pin_memory=device.type == "cuda",
        persistent_workers=workers > 0)
    torch.manual_seed(0)
    model = VITSModel(vocab_size=len(table), spec_channels=513, padding_id=0,  # noqa: F821
                      text_encoder=switches["text_encoder"], posterior_encoder=switches["posterior_encoder"],
                      decoder=switches["decoder"])
    discriminator = (MultiPeriodDiscriminator() if switches["discriminator"] == "tts010"  # noqa: F821
                     else LegacyDiscriminator(base_channels=32))  # noqa: F821
    loss = VITSLoss(recipe=switches["recon_loss"], feature_matching=switches["feature_matching"])  # noqa: F821
    trainer = VITSTrainer(model, discriminator, loss, loader, learning_rate=preset["learning_rate"],  # noqa: F821
                          use_amp=preset["amp"], max_grad_norm=preset["max_grad_norm"], device=device,
                          switches=switches, checkpoint_dir=ROOT / "output" / "benchmark-tmp",
                          sample_dir=ROOT / "output" / "benchmark-tmp")
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    times, iterator = [], iter(loader)
    for step in range(warmup + steps):
        start = time.perf_counter()
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        trainer._training_step(trainer._move_batch_to_device(batch))
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        if step >= warmup:
            times.append(time.perf_counter() - start)
    mean = statistics.mean(times)
    result = {
        "arm": arm, "batch_size": preset["batch_size"], "autocast_fp16": trainer.use_amp,
        "training_clips": len(texts), "steps_timed": len(times),
        "sec_per_step_mean": round(mean, 4), "sec_per_step_median": round(statistics.median(times), 4),
        "hours_for_40k_steps": round(mean * 40000 / 3600, 2),
        "peak_allocated_GiB": round(torch.cuda.max_memory_allocated(device) / 2**30, 2) if device.type == "cuda" else None,
        "generator_parameters": sum(p.numel() for p in model.parameters()),
        "discriminator_parameters": sum(p.numel() for p in discriminator.parameters()),
    }
    del trainer, model, discriminator, loss, loader
    return result


def collate_tts_batch_partial(batch):
    return collate_tts_batch(batch, pad_id=0)  # noqa: F821


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--metadata", required=True, help="LJSpeech-1.1 metadata.csv")
    parser.add_argument("--arms", nargs="+", default=sorted(set(literal_assignment("ARMS")) - {"R0b"}))
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--output", default=str(ROOT / "output" / "tts010-benchmark.json"))
    parser.add_argument("--deterministic", action="store_true",
                        help="use deterministic cuDNN kernels (the notebook does not)")
    args = parser.parse_args()
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = args.deterministic
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    frame, _ = load_training_metadata(args.metadata)  # noqa: F821
    texts = list(zip(frame["file_name"], frame["normalized_transcription"]))
    heldout = set(load_id_list(ROOT / "sts_eval" / "splits" / "ljspeech_heldout_100.txt"))  # noqa: F821
    table = {s: i for i, s in enumerate(
        ["_"] + list(literal_assignment("PUNCTUATION")) + list(literal_assignment("LETTERS")))}
    record = {"gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
              "torch": torch.__version__, "scope": "synthetic audio with LJSpeech-like lengths; timing only",
              "cudnn_deterministic": args.deterministic,
              "results": []}
    for arm in args.arms:
        result = benchmark(arm, texts, heldout, table, args.steps, args.warmup, args.num_workers, device)
        record["results"].append(result)
        print(json.dumps(result), flush=True)
    record["completed_at_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
