"""Whisper word/character error rates for an evaluation folder written by tools/eval_tts010.py.

Locked protocol: openai-whisper, task=transcribe, language=en, temperature=0, beam size 5, no prompt,
no conditioning on previous text, mono 16 kHz input (scipy polyphase resampling). Reference and
hypothesis share one normalisation: Unicode NFKC, then whisper's EnglishTextNormalizer. WER and CER
are corpus rates, sum(S + D + I) / sum(N), not means of sentence rates.

Channels from asr_manifest.json:
  synthesized      text-only synthesis of each held-out sentence
  natural          the LJSpeech recording of the same sentence (ASR baseline)
  synthesized_new  synthesis of the fixed new sentences (no natural recording)
delta_wer = WER(synthesized) - WER(natural) over the paired held-out sentences.

Runs in any Python with openai-whisper, numpy and scipy (no torchaudio needed):

python tools/asr_wer.py output/tts010-eval/R0_40k --model medium
"""

import argparse
import hashlib
import json
import sys
import unicodedata
from datetime import datetime, timezone
from math import gcd
from pathlib import Path

import numpy as np

DECODE = {"task": "transcribe", "language": "en", "temperature": 0.0, "beam_size": 5,
          "condition_on_previous_text": False, "initial_prompt": None}


def align_counts(reference, hypothesis):
    """Minimum-edit alignment of token lists: (substitutions, deletions, insertions)."""
    rows, cols = len(reference) + 1, len(hypothesis) + 1
    cost = [[0] * cols for _ in range(rows)]
    for i in range(rows):
        cost[i][0] = i
    for j in range(cols):
        cost[0][j] = j
    for i in range(1, rows):
        for j in range(1, cols):
            cost[i][j] = min(cost[i - 1][j] + 1, cost[i][j - 1] + 1,
                             cost[i - 1][j - 1] + (reference[i - 1] != hypothesis[j - 1]))
    i, j, s, d, n_ins = len(reference), len(hypothesis), 0, 0, 0
    while i or j:
        if i and j and cost[i][j] == cost[i - 1][j - 1] + (reference[i - 1] != hypothesis[j - 1]):
            s += reference[i - 1] != hypothesis[j - 1]
            i, j = i - 1, j - 1
        elif i and cost[i][j] == cost[i - 1][j] + 1:
            d, i = d + 1, i - 1
        else:
            n_ins, j = n_ins + 1, j - 1
    return s, d, n_ins


def corpus_rates(rows, unit):
    n = sum(r[f"{unit}_n"] for r in rows)
    errors = {k: sum(r[f"{unit}_{k}"] for r in rows) for k in ("s", "d", "i")}
    rate = sum(errors.values()) / n if n else None
    return {"rate": rate, "tokens": n, **{f"{k}_rate": (v / n if n else None) for k, v in errors.items()}}


def read_audio_16k(path):
    from scipy.io import wavfile
    from scipy.signal import resample_poly
    rate, audio = wavfile.read(path)
    if audio.dtype.kind == "i":
        audio = audio.astype(np.float32) / np.iinfo(audio.dtype).max
    audio = audio.astype(np.float32)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    factor = gcd(16000, rate)
    return resample_poly(audio, 16000 // factor, rate // factor).astype(np.float32)


def score(reference, hypothesis, normalizer):
    ref = normalizer(unicodedata.normalize("NFKC", reference))
    hyp = normalizer(unicodedata.normalize("NFKC", hypothesis))
    row = {"ref_norm": ref, "hyp_norm": hyp}
    for unit, ref_tokens, hyp_tokens in (("word", ref.split(), hyp.split()),
                                         ("char", list(ref.replace(" ", "")), list(hyp.replace(" ", "")))):
        s, d, i = align_counts(ref_tokens, hyp_tokens)
        row.update({f"{unit}_n": len(ref_tokens), f"{unit}_s": s, f"{unit}_d": d, f"{unit}_i": i})
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("eval_dir", help="a folder written by tools/eval_tts010.py or tools/chunk_study.py")
    parser.add_argument("--model", default="medium", help="openai-whisper model name")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    # A console code page that cannot encode a hypothesis (e.g. GBK and U+FFFD) must not stop a long run.
    sys.stdout.reconfigure(errors="backslashreplace")
    import whisper
    from whisper.normalizers import EnglishTextNormalizer

    eval_dir = Path(args.eval_dir)
    manifest = json.loads((eval_dir / "asr_manifest.json").read_text(encoding="utf-8"))
    model = whisper.load_model(args.model, device=args.device)
    model_file = Path(whisper._MODELS[args.model]).name if hasattr(whisper, "_MODELS") else None
    cache = Path.home() / ".cache" / "whisper" / (model_file or "")
    normalizer = EnglishTextNormalizer()
    rows = []
    for item in manifest:
        row = {"id": item["id"], "channel": item["channel"], "wav": item["wav"], "reference": item["reference"]}
        try:
            audio = read_audio_16k(item["wav"])
            result = model.transcribe(audio, fp16=False, **DECODE)
            row["hypothesis"] = result["text"].strip()
            row["status"] = "ok" if row["hypothesis"] else "empty_output"
        except (OSError, ValueError) as error:
            row["hypothesis"], row["status"] = "", f"failed: {type(error).__name__}: {error}"
        row.update(score(row["reference"], row["hypothesis"], normalizer))
        rows.append(row)
        print(f"{row['channel']:16s} {row['id']:12s} {row['hypothesis'][:70]}", flush=True)

    summary = {}
    for channel in sorted({r["channel"] for r in rows}):
        selected = [r for r in rows if r["channel"] == channel]
        summary[channel] = {"utterances": len(selected),
                            "failures": sum(r["status"] != "ok" for r in selected),
                            "sentence_error_rate": sum(r["word_s"] + r["word_d"] + r["word_i"] > 0 for r in selected) / len(selected),
                            "wer": corpus_rates(selected, "word"), "cer": corpus_rates(selected, "char")}
    if "synthesized" in summary and "natural" in summary:
        summary["delta_wer"] = summary["synthesized"]["wer"]["rate"] - summary["natural"]["wer"]["rate"]
        summary["delta_cer"] = summary["synthesized"]["cer"]["rate"] - summary["natural"]["cer"]["rate"]
    record = {
        "protocol": "asr_back_transcription_v1", "decode": DECODE,
        "asr": {"package": "openai-whisper", "version": getattr(whisper, "__version__", None), "model": args.model,
                "model_sha256": hashlib.sha256(cache.read_bytes()).hexdigest() if cache.is_file() else None,
                "device": str(next(model.parameters()).device)},
        "normalisation": "NFKC + whisper.normalizers.EnglishTextNormalizer; CER over characters without spaces",
        "summary": summary, "utterances": rows,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (eval_dir / "asr.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
