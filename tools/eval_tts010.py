"""Evaluate one checkpoint with the shared TTS_0.10 protocol (run centrally on one machine).

Measures, with fixed seeds:
  probe          sts_eval fixed crops of LJ001-0001...0008 (continuity with earlier reports)
  heldout_crops  sts_eval fixed crops of the first 8 held-out clips
  heldout_full   posterior-mean reconstruction of every held-out utterance, full length
  heldout_tts    text-only synthesis of every held-out sentence: duration and level vs the recording
  listening      WAV files for 4 fixed new sentences

All synthesized held-out sentences are written to audio/heldout_tts/ together with
asr_manifest.json (synthesized and natural recordings with their reference texts), so that
tools/asr_wer.py can add Whisper WER/CER and the natural-speech baseline in a separate step.

Accepts TTS_0.10 checkpoints (training runs or tools/convert_benlu_checkpoint.py output) and
TTS_0.01/0.02 checkpoints (all components tts002, fed the |X| spectrogram they were trained on).
Held-out metrics need the complete LJSpeech-1.1 recordings; missing recordings are counted,
never silently replaced.

python tools/eval_tts010.py --checkpoint run/checkpoints/vits_step_40000.pt --label G1_40k \
    --data-dir /data/LJSpeech-1.1
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
import torchaudio

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sts_eval import EvaluationConfig, ReconstructionEvaluator  # noqa: E402
from sts_eval.reconstruction import mel_transform, score_waveforms  # noqa: E402
from tools.notebook_defs import literal_assignment, load_definitions  # noqa: E402

NOISE_SCALE = 0.667
EVAL_SEED = 1234
PROBE_IDS = [f"LJ001-{i:04d}" for i in range(1, 9)]
NEW_SENTENCES = [
    "The quick brown fox jumps over the lazy dog.",
    "Our team trained this speech model from scratch for a university course project.",
    "Please call Stella and ask her to bring these things with her from the store.",
    "How much wood would a woodchuck chuck if a woodchuck could chuck wood?",
]


def legacy_spectrogram(waveform, n_fft=1024, hop_length=256, win_length=1024):
    """The TTS_0.01/0.02 magnitude spectrogram, |X|, for checkpoints trained on it."""
    padding = (n_fft - hop_length) // 2
    waveform = torch.nn.functional.pad(waveform, (padding, padding), mode="reflect")
    window = torch.hann_window(win_length, device=waveform.device)
    return torch.stft(waveform, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window,
                      center=False, normalized=False, onesided=True, return_complex=True).abs().squeeze(0)


def load_model(ns, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    switches = checkpoint.get("switches") or literal_assignment("TTS002_SWITCHES")
    speaker = checkpoint.get("speaker_config", {"n_speakers": 0, "gin_channels": 0})
    if speaker["n_speakers"]:
        raise ValueError("This protocol evaluates single-speaker LJSpeech checkpoints.")
    model = ns["VITSModel"](vocab_size=len(ns["symbol_to_id"]), spec_channels=513, padding_id=0,
                            text_encoder=switches["text_encoder"], posterior_encoder=switches["posterior_encoder"],
                            decoder=switches["decoder"])
    model.load_state_dict(checkpoint["model"], strict=True)
    return model.to(device).eval(), checkpoint, switches


def synthesize(model, ns, text, seed, device):
    tokens = ns["encode_text"](text)
    torch.manual_seed(seed)
    result = model.infer(tokens.unsqueeze(0).to(device), torch.tensor([tokens.numel()], device=device),
                         noise_scale=NOISE_SCALE)
    return result["generated_audio"][0, 0, :int(result["audio_lengths"][0])].float().cpu()


def mean(rows, key):
    values = [row[key] for row in rows if row.get(key) is not None]
    return sum(values) / len(values) if values else None


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--label", required=True, help="e.g. R0_40k")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out", default=str(ROOT / "output" / "tts010-eval"))
    args = parser.parse_args()

    ns = load_definitions()
    ns["PAD_ID"] = 0
    ns["symbol_to_id"] = {s: i for i, s in enumerate(
        ["_"] + list(literal_assignment("PUNCTUATION")) + list(literal_assignment("LETTERS")))}
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model, checkpoint, switches = load_model(ns, args.checkpoint, device)
    legacy = "switches" not in checkpoint
    if legacy:
        # TTS_0.01/0.02 were trained on |X|; TTS_0.10 uses sqrt(|X|^2 + 1e-6).
        ns["waveform_to_spectrogram"] = legacy_spectrogram
    data_dir = Path(args.data_dir)
    corpus, _ = ns["load_training_metadata"](data_dir / "metadata.csv")
    texts = dict(zip(corpus["file_name"], corpus["normalized_transcription"]))
    heldout_ids = ns["load_id_list"](ROOT / "sts_eval" / "splits" / "ljspeech_heldout_100.txt")
    out = Path(args.out) / args.label
    out.mkdir(parents=True, exist_ok=False)
    (out / "audio" / "heldout_tts").mkdir(parents=True)
    available = [i for i in heldout_ids if (data_dir / "wavs" / f"{i}.wav").is_file()]
    record = {
        "label": args.label, "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_step": checkpoint.get("step"), "model_version": checkpoint.get("model_version"),
        "switches": switches, "device": str(device), "torch": torch.__version__,
        "noise_scale": NOISE_SCALE, "eval_seed": EVAL_SEED,
        "spectrogram": "|X| (TTS_0.01/0.02 input)" if legacy else "sqrt(|X|^2 + 1e-6)",
        "heldout_recordings_found": len(available), "heldout_ids_total": len(heldout_ids),
        "scope": "LJSpeech single speaker; held-out = sts_eval/splits/ljspeech_heldout_100.txt",
    }

    def dataset(ids):
        frame = corpus.set_index("file_name", drop=False).loc[ids].reset_index(drop=True)
        return ns["LJSpeechDataset"](frame, data_dir / "wavs")

    config = EvaluationConfig(seed=EVAL_SEED)
    for name, ids, scope in (("probe", PROBE_IDS, "fixed_lj001_probe"),
                             ("heldout_crops", available[:8], "held_out_not_trained")):
        ids = [i for i in ids if (data_dir / "wavs" / f"{i}.wav").is_file()]
        if not ids:
            record[name] = None
            continue
        data = dataset(ids)
        batch = ns["collate_tts_batch"]([data[i] for i in range(len(data))])
        evaluator = ReconstructionEvaluator(batch, ids, out / name, config=config, scope=scope)
        record[name] = evaluator.evaluate(model, int(checkpoint.get("step") or 0))

    mel = mel_transform(config).to(device)  # the sts_eval metric definitions, on full utterances
    full_rows, tts_rows, manifest = [], [], []
    data = dataset(available) if available else []
    with torch.inference_mode():
        for index, sample_id in enumerate(available):
            _, spectrogram, waveform = data[index]
            spec = spectrogram.unsqueeze(0).to(device)
            lengths = torch.tensor([spec.size(-1)], device=device)
            _, posterior_mean, _, mask = model.posterior_encoder(spec, lengths)
            reconstruction = model.decoder(posterior_mean * mask)[0, 0]
            target = waveform[0].to(device)
            n = min(reconstruction.numel(), target.numel())
            full_rows.append({"sample_id": sample_id, **score_waveforms(
                reconstruction[:n].float(), target[:n].float(), config, mel)})

            audio = synthesize(model, ns, texts[sample_id], EVAL_SEED + index, device)
            path = out / "audio" / "heldout_tts" / f"{sample_id}.wav"
            torchaudio.save(str(path), audio.clamp(-1, 1).unsqueeze(0), 22050, encoding="PCM_S", bits_per_sample=16)
            tts_rows.append({"sample_id": sample_id, "seconds": audio.numel() / 22050,
                             "duration_ratio": audio.numel() / waveform.size(-1),
                             "rms_ratio": float(audio.pow(2).mean().sqrt() / waveform.pow(2).mean().sqrt().clamp_min(1e-8))})
            manifest.append({"id": sample_id, "channel": "synthesized", "wav": str(path), "reference": texts[sample_id]})
            manifest.append({"id": sample_id, "channel": "natural", "wav": str(data_dir / "wavs" / f"{sample_id}.wav"),
                             "reference": texts[sample_id]})
        for index, sentence in enumerate(NEW_SENTENCES):
            audio = synthesize(model, ns, sentence, EVAL_SEED + 1000 + index, device)
            path = out / "audio" / f"new_{index + 1}.wav"
            torchaudio.save(str(path), audio.clamp(-1, 1).unsqueeze(0), 22050, encoding="PCM_S", bits_per_sample=16)
            manifest.append({"id": f"new_{index + 1}", "channel": "synthesized_new", "wav": str(path),
                             "reference": sentence})

    record["heldout_full"] = {"count": len(full_rows), **{key: mean(full_rows, key) for key in (
        "log_mel_l1", "stft_spectral_convergence", "stft_log_magnitude")}}
    record["heldout_tts"] = {"count": len(tts_rows), "duration_ratio_mean": mean(tts_rows, "duration_ratio"),
                             "rms_ratio_mean": mean(tts_rows, "rms_ratio")}
    record["listening_sentences"] = NEW_SENTENCES
    record["completed_at_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    (out / "per_utterance.json").write_text(json.dumps({"heldout_full": full_rows, "heldout_tts": tts_rows},
                                                        indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (out / "asr_manifest.json").write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    (out / "metrics.json").write_text(json.dumps(record, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    summary = {k: record[k] for k in ("heldout_full", "heldout_tts")}
    summary["probe"] = [{k: round(v, 4) if isinstance(v, float) else v for k, v in row.items()}
                        for row in (record["probe"] or [])]
    print(json.dumps(summary, indent=1))
    print(out)


if __name__ == "__main__":
    main()
