"""Chunked (streaming-text) synthesis study: how intelligible is a checkpoint when it speaks a few words at a time?

A voice agent can start speaking before the language model has finished a sentence by handing
the TTS a few words at a time. For each held-out sentence (same seed per sentence) this writes:

  full        the whole sentence in one call (reference)
  naive_N     every N-word chunk synthesized on its own, the audio concatenated
  ctx_N_kK    the text encoder sees all previous words + the chunk + K following words, but only
              the chunk's characters receive frames (the durations of the others are set to 0)

for N in 1, 2, 4, 8 and K in 0, 2, 4, plus asr_manifest.json (channels = conditions and, unless
--no-natural, the LJSpeech recording) for tools/asr_wer.py and timing.json (synthesis time per
call and duration ratio per condition). Analyse with tools/chunk_analysis.py.

python tools/chunk_study.py --checkpoint R0/checkpoints/vits_step_40000.pt \
    --data-dir /data/LJSpeech-1.1 --out output/chunk-study/R0_40k
python tools/asr_wer.py output/chunk-study/R0_40k --model medium
python tools/chunk_analysis.py analyze output/chunk-study/R0_40k
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
import torchaudio

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.eval_tts010 import load_model  # noqa: E402
from tools.text_input import model_words  # noqa: E402
from tools.notebook_defs import literal_assignment, load_definitions  # noqa: E402

NOISE_SCALE = 0.667
SEED = 1234
CONDITIONS = ([("full", None, False, 0)] + [(f"naive_{n}", n, False, 0) for n in (1, 2, 4, 8)]
              + [(f"ctx_{n}_k{k}", n, True, k) for n in (1, 2, 4, 8) for k in (0, 2, 4)])


@torch.inference_mode()
def synth(ns, model, text, target_span, seed, device, noise=NOISE_SCALE, lookahead=None):
    """Synthesize only characters [start, end) of text, using the whole text as encoder context.

    text is in the model's input (phoneme strings for phoneme models, tools/text_input.py). A
    windowed encoder (TTS_0.20 option) gets the chunk's lookahead as its own.
    """
    table = getattr(model, "input_table", None) or ns["symbol_to_id"]
    encoder = model.text_encoder
    if getattr(encoder, "window", None) is not None:
        encoder.lookahead_words = encoder.window[1] if lookahead is None else lookahead
    keep = [(i, c) for i, c in enumerate(text) if c in table]
    tokens = torch.tensor([[table[c] for _, c in keep]], device=device)
    target = torch.tensor([[[1.0 if target_span[0] <= i < target_span[1] else 0.0 for i, _ in keep]]], device=device)
    lengths = torch.tensor([tokens.size(1)], device=device)
    torch.manual_seed(seed)
    x, m_p, logs_p, x_mask = model.text_encoder(tokens, lengths)
    logw = model.predict_log_duration(x, x_mask).clamp(min=-5.0, max=math.log(50))
    w = torch.ceil(torch.exp(logw)) * x_mask * target
    y_len = w.sum(dim=(1, 2)).long().clamp(min=1)
    y_mask = ns["create_sequence_mask"](y_len, int(y_len.max())).unsqueeze(1).to(x.dtype)
    attn = ns["make_inference_path"](w, y_mask.unsqueeze(-1) * x_mask.unsqueeze(2))
    m = ns["expand_with_alignment"](m_p, attn)
    logs = ns["expand_with_alignment"](logs_p, attn)
    z_p = m + torch.randn_like(m) * torch.exp(logs) * noise
    z = model.flow(z_p, y_mask, reverse=True)
    audio = model.decoder(z * y_mask)[0, 0, :int(y_len) * 256]
    return audio.float().cpu()


def spans(words, a, b, left, look):
    """Text = left context + words[a:b] + look following words; return (text, char span of the chunk)."""
    lo = 0 if left else a
    text = " ".join(words[lo:b + look])
    start = len(" ".join(words[lo:a])) + (1 if a > lo else 0)
    end = start + len(" ".join(words[a:b])) + (1 if b < len(words) else 0)  # include the following space
    return text, (start, end)


def run_condition(ns, model, words, n, left, look, seed, device):
    """Audio of one condition and the synthesis time of every call."""
    if n is None:
        started = time.perf_counter()
        audio = synth(ns, model, " ".join(words), (0, 10**6), seed, device)
        return audio, [time.perf_counter() - started]
    parts, times = [], []
    for a in range(0, len(words), n):
        text, span = spans(words, a, min(a + n, len(words)), left, look)
        started = time.perf_counter()
        parts.append(synth(ns, model, text, span, seed + a, device, lookahead=look))
        times.append(time.perf_counter() - started)
    return torch.cat(parts), times


def notebook_namespace():
    ns = load_definitions()
    ns["PAD_ID"] = 0
    ns["symbol_to_id"] = {s: i for i, s in enumerate(
        ["_"] + list(literal_assignment("PUNCTUATION")) + list(literal_assignment("LETTERS")))}
    return ns


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out", required=True, help="new folder for the WAVs, manifest and timing")
    parser.add_argument("--sentences", type=int, default=100, help="first N held-out sentences (default all 100)")
    parser.add_argument("--no-natural", action="store_true",
                        help="leave the recordings out of the manifest (when they are already scored for another checkpoint)")
    parser.add_argument("--cpu-timing-sentences", type=int, default=10,
                        help="also time full/naive_2/ctx_2_k2 on the CPU for this many sentences (0 = skip)")
    args = parser.parse_args()

    out, data = Path(args.out), Path(args.data_dir)
    out.mkdir(parents=True, exist_ok=False)
    ns = notebook_namespace()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model, checkpoint, _ = load_model(ns, args.checkpoint, device)
    corpus, _ = ns["load_training_metadata"](data / "metadata.csv")
    texts = dict(zip(corpus["file_name"], corpus["normalized_transcription"]))
    ids = ns["load_id_list"](ROOT / "sts_eval/splits/ljspeech_heldout_100.txt")[:args.sentences]

    manifest, timing, durations = [], {}, {}
    synth(ns, model, " warm up sentence for the kernels.", (0, 10**6), 0, device)
    for idx, sid in enumerate(ids):
        words = model_words(texts[sid], getattr(model, "input_switches", None))
        recording = data / "wavs" / f"{sid}.wav"
        real = torchaudio.load(str(recording))[0]
        if not args.no_natural:
            manifest.append({"id": sid, "channel": "natural", "wav": str(recording), "reference": texts[sid]})
        for name, n, left, look in CONDITIONS:
            audio, times = run_condition(ns, model, words, n, left, look, SEED + 100 * idx, device)
            path = out / name / f"{sid}.wav"
            path.parent.mkdir(exist_ok=True)
            torchaudio.save(str(path), audio.clamp(-1, 1).unsqueeze(0), 22050, encoding="PCM_S", bits_per_sample=16)
            manifest.append({"id": sid, "channel": name, "wav": str(path), "reference": texts[sid]})
            timing.setdefault(name, []).extend(times)
            durations.setdefault(name, []).append(audio.numel() / real.size(-1))

    cpu = {}
    if args.cpu_timing_sentences:
        cpu_model, _, _ = load_model(ns, args.checkpoint, torch.device("cpu"))
        torch.set_num_threads(8)
        for name, n, left, look in [c for c in CONDITIONS if c[0] in ("full", "naive_2", "ctx_2_k2")]:
            for sid in ids[:args.cpu_timing_sentences]:
                _, times = run_condition(ns, cpu_model, model_words(texts[sid], getattr(cpu_model, "input_switches", None)),
                                         n, left, look, 1, torch.device("cpu"))
                cpu.setdefault(name, []).extend(times)
    summary = {name: {"gpu_ms_per_call_mean": 1000 * sum(v) / len(v), "calls": len(v),
                      "duration_ratio_mean": sum(durations[name]) / len(durations[name])} for name, v in timing.items()}
    for name, v in cpu.items():
        summary[name]["cpu_ms_per_call_mean"] = 1000 * sum(v) / len(v)
    mean_words = sum(len(texts[s].split()) for s in ids) / len(ids)
    record = {"checkpoint": str(Path(args.checkpoint).resolve()), "checkpoint_step": checkpoint.get("step"),
              "device": str(device), "torch": torch.__version__, "noise_scale": NOISE_SCALE, "seed": SEED,
              "sentences": ids, "mean_words": mean_words, "summary": summary}
    (out / "timing.json").write_text(json.dumps(record, indent=1), encoding="utf-8")
    (out / "asr_manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print(json.dumps({"mean_words": mean_words, **summary}, indent=1))
    print(out)


if __name__ == "__main__":
    main()
