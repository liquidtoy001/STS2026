"""Whisper word/character error rates for an evaluation folder written by tools/eval_tts010.py.

Locked protocol v1 (asr.json): openai-whisper, task=transcribe, language=en, temperature=0, beam size 5,
no prompt, no conditioning on previous text, mono 16 kHz input (scipy polyphase resampling).

Protocol v2 (--protocol v2, asr_v2.json): v1, plus whisper's standard temperature fallback for
repetition loops only: when a decode's compression ratio exceeds 2.4, decode again at temperature
0.2, 0.4, ... 1.0 (best of 5, seeded per utterance); the log-probability and no-speech fallbacks
stay off. A clip whose v1 hypothesis is not a loop gets exactly its v1 decode, so with an asr.json
of the same model present only the looping clips are decoded again (--no-reuse decodes all). Reference and
hypothesis share one normalisation: Unicode NFKC, then whisper's EnglishTextNormalizer. WER and CER
are corpus rates, sum(S + D + I) / sum(N), not means of sentence rates.

Channels from asr_manifest.json:
  synthesized      text-only synthesis of each held-out sentence
  natural          the LJSpeech recording of the same sentence (ASR baseline)
  synthesized_new  synthesis of the fixed new sentences (no natural recording)
delta_wer = WER(synthesized) - WER(natural) over the paired held-out sentences.

Runs in any Python with openai-whisper, numpy and scipy (no torchaudio needed):

python tools/asr_wer.py output/tts010-eval/R0_40k --model medium
python tools/asr_wer.py output/tts010-eval/R0_40k --model medium --protocol v2
"""

import argparse
import hashlib
import json
import sys
import unicodedata
import zlib
from datetime import datetime, timezone
from math import gcd
from pathlib import Path

import numpy as np

DECODE = {"task": "transcribe", "language": "en", "temperature": 0.0, "beam_size": 5,
          "condition_on_previous_text": False, "initial_prompt": None}
LOOP_COMPRESSION_RATIO = 2.4
DECODE_V2 = {**DECODE, "temperature": (0.0, 0.2, 0.4, 0.6, 0.8, 1.0), "best_of": 5,
             "compression_ratio_threshold": LOOP_COMPRESSION_RATIO, "logprob_threshold": None,
             "no_speech_threshold": None}
SEED_V2 = 1234
# protocol -> (record name, decode options, output file)
PROTOCOLS = {"v1": ("asr_back_transcription_v1", DECODE, "asr.json"),
             "v2": ("asr_back_transcription_v2", DECODE_V2, "asr_v2.json")}


def compression_ratio(text):
    """whisper's repetition measure: UTF-8 bytes over zlib-compressed bytes."""
    data = text.encode("utf-8")
    return len(data) / len(zlib.compress(data)) if data else 0.0


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
    parser.add_argument("--protocol", choices=sorted(PROTOCOLS), default="v1")
    parser.add_argument("--no-reuse", action="store_true", help="v2: decode every clip, ignoring asr.json")
    args = parser.parse_args()
    protocol_name, decode, out_name = PROTOCOLS[args.protocol]
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
    reuse = {}
    v1_path = eval_dir / PROTOCOLS["v1"][2]
    if args.protocol == "v2" and not args.no_reuse and v1_path.is_file():
        v1 = json.loads(v1_path.read_text(encoding="utf-8"))
        if v1.get("protocol") == PROTOCOLS["v1"][0] and v1["asr"]["model"] == args.model:
            reuse = {(r["id"], r["channel"], r["wav"]): r for r in v1["utterances"]
                     if r["status"] == "ok" and compression_ratio(r["hypothesis"]) <= LOOP_COMPRESSION_RATIO}
    rows = []
    for index, item in enumerate(manifest):
        row = {"id": item["id"], "channel": item["channel"], "wav": item["wav"], "reference": item["reference"]}
        previous = reuse.get((item["id"], item["channel"], item["wav"]))
        if previous is not None:
            row["hypothesis"], row["status"], row["source"] = previous["hypothesis"], previous["status"], "v1_decode"
            row.update(score(row["reference"], row["hypothesis"], normalizer))
            rows.append(row)
            continue
        try:
            audio = read_audio_16k(item["wav"])
            if args.protocol == "v2":
                import torch
                torch.manual_seed(SEED_V2 + index)
            result = model.transcribe(audio, fp16=False, **decode)
            row["hypothesis"] = result["text"].strip()
            row["status"] = "ok" if row["hypothesis"] else "empty_output"
            if args.protocol == "v2":
                row["source"] = "decoded"
                row["max_temperature"] = max((s.get("temperature", 0.0) for s in result.get("segments", [])),
                                             default=0.0)
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
        "protocol": protocol_name, "decode": decode,
        "asr": {"package": "openai-whisper", "version": getattr(whisper, "__version__", None), "model": args.model,
                "model_sha256": hashlib.sha256(cache.read_bytes()).hexdigest() if cache.is_file() else None,
                "device": str(next(model.parameters()).device)},
        "normalisation": "NFKC + whisper.normalizers.EnglishTextNormalizer; CER over characters without spaces",
        "summary": summary, "utterances": rows,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    if args.protocol == "v2":
        record["reused_v1_decodes"] = sum(r.get("source") == "v1_decode" for r in rows)
        record["fallback_decodes"] = sum(r.get("max_temperature", 0.0) > 0 for r in rows)
    (eval_dir / out_name).write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
