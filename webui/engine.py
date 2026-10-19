"""Checkpoint registry and text-only synthesis behind the web UI.

Models are listed in a JSON config (webui/config.json, or the committed config.example.json).
Each checkpoint is rebuilt from the switches and speaker configuration stored in it, with
the notebook's own class definitions, so a run trained on another dataset or with speaker
conditioning loads without code changes as long as it uses the notebook's symbol table.
"""

import base64
import glob
import io
import json
import random
import threading
import time
import wave
from pathlib import Path

import torch

from tools.notebook_defs import literal_assignment, load_definitions
from webui import text_en

ROOT = Path(__file__).resolve().parents[1]
NORMALIZERS = {"english": text_en.normalize, "none": None}
# TTS_0.10 arms other than D1 were trained on transcriptions of at most 100 characters and
# the longest LJSpeech transcription has about 190; longer segments are split at clause breaks.
MAX_SEGMENT_CHARS = 200
MAX_INPUT_CHARS = 100_000
# Fixed log-mel display range (natural log of the magnitude mel), shared by every model so
# spectrograms of different checkpoints are comparable.
MEL_RANGE = (-7.0, 5.0)


def resolve_checkpoint(pattern, base):
    """Return the newest file matching a path or glob pattern relative to `base`, or None."""
    path = Path(pattern)
    if not path.is_absolute():
        path = base / path
    matches = [Path(p) for p in glob.glob(str(path), recursive=True)]
    matches = [p for p in matches if p.is_file()]
    return max(matches, key=lambda p: p.stat().st_mtime) if matches else None


def wav_bytes(audio, sample_rate):
    pcm = (audio.clamp(-1, 1) * 32767).round().to(torch.int16).cpu().numpy()
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm.tobytes())
    return buffer.getvalue()


class Engine:
    def __init__(self, config_path, device=None):
        self.config_path = Path(config_path)
        self.config = json.loads(self.config_path.read_text(encoding="utf-8"))
        self.checkpoint_root = (self.config_path.parent / self.config.get("checkpoint_root", "..")).resolve()
        self.device = torch.device(device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
        self.ns = load_definitions()
        pad = literal_assignment("PAD_SYMBOL")
        self.symbols = [pad] + list(literal_assignment("PUNCTUATION")) + list(literal_assignment("LETTERS"))
        self.symbol_to_id = {symbol: index for index, symbol in enumerate(self.symbols)}
        self.text_symbols = frozenset(self.symbols[1:])
        self.ns["PAD_ID"] = self.symbol_to_id[pad]
        self.ns["symbol_to_id"] = self.symbol_to_id
        self.audio = {name.lower(): literal_assignment(name) for name in ("SAMPLE_RATE", "N_FFT", "HOP_LENGTH",
                                                                          "WIN_LENGTH", "N_MELS")}
        self.entries = {}
        for entry in self.config["models"]:
            if entry["id"] in self.entries:
                raise ValueError(f"Duplicate model id {entry['id']!r} in {self.config_path}")
            if entry.get("normalizer", "english") not in NORMALIZERS:
                raise ValueError(f"Unknown normalizer {entry['normalizer']!r}; choose one of {sorted(NORMALIZERS)}")
            self.entries[entry["id"]] = entry
        self.loaded = {}
        self.lock = threading.Lock()
        self.mel_transforms = {}

    # -- registry -----------------------------------------------------------------------------

    def model_list(self):
        rows = []
        for entry in self.entries.values():
            path = resolve_checkpoint(entry["checkpoint"], self.checkpoint_root)
            row = {key: entry.get(key) for key in ("id", "label", "description", "wer", "notes")}
            row.update({"pattern": entry["checkpoint"], "path": str(path) if path else None,
                        "available": path is not None, "loaded": entry["id"] in self.loaded})
            if entry["id"] in self.loaded:
                row.update(self.loaded[entry["id"]]["info"])
            rows.append(row)
        return rows

    def status(self):
        return {
            "device": str(self.device),
            "gpu": torch.cuda.get_device_name(self.device) if self.device.type == "cuda" else None,
            "torch": torch.__version__,
            "config": str(self.config_path),
            "checkpoint_root": str(self.checkpoint_root),
            "default_model": self.config.get("default_model"),
            "evaluation": self.config.get("evaluation"),
            "limitations": self.config.get("limitations", []),
            "examples": self.config.get("examples", []),
            "max_segment_chars": MAX_SEGMENT_CHARS,
            "mel_range": MEL_RANGE,
            "models": self.model_list(),
        }

    def load(self, model_id):
        with self.lock:
            if model_id not in self.loaded:
                self.loaded[model_id] = self._load(model_id)
            return self.loaded[model_id]["info"]

    def _load(self, model_id):
        entry = self.entries.get(model_id)
        if entry is None:
            raise KeyError(f"Unknown model {model_id!r}")
        path = resolve_checkpoint(entry["checkpoint"], self.checkpoint_root)
        if path is None:
            raise FileNotFoundError(f"No checkpoint matches {entry['checkpoint']} under {self.checkpoint_root}")
        started = time.perf_counter()
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        state = checkpoint["model"]
        vocab = state["text_encoder.token_embedding.weight"].shape[0]
        if vocab != len(self.symbols):
            raise ValueError(f"{path.name} has {vocab} text symbols; the notebook defines {len(self.symbols)}.")
        # TTS_0.01/0.02 checkpoints carry no switches: every component is the tts002 version.
        switches = checkpoint.get("switches") or literal_assignment("TTS002_SWITCHES")
        speaker = checkpoint.get("speaker_config") or {"n_speakers": 0, "gin_channels": 0, "speaker_to_id": {}}
        model = self.ns["VITSModel"](
            vocab_size=vocab, spec_channels=self.audio["n_fft"] // 2 + 1, padding_id=self.ns["PAD_ID"],
            hop_length=self.audio["hop_length"], n_speakers=speaker["n_speakers"],
            gin_channels=speaker["gin_channels"], text_encoder=switches["text_encoder"],
            posterior_encoder=switches["posterior_encoder"], decoder=switches["decoder"])
        model.load_state_dict(state, strict=True)
        model = model.to(self.device).eval()
        positions = state.get("text_encoder.position_embedding.weight")
        speakers = sorted(speaker.get("speaker_to_id", {}), key=speaker.get("speaker_to_id", {}).get)
        info = {
            "path": str(path), "step": checkpoint.get("step"), "model_version": checkpoint.get("model_version"),
            "switches": switches, "speakers": speakers,
            "sample_rate": int(entry.get("sample_rate", self.audio["sample_rate"])),
            "hop_length": model.hop_length, "max_tokens": int(positions.shape[0]) if positions is not None else None,
            "parameters": sum(p.numel() for p in model.parameters()),
            "normalizer": entry.get("normalizer", "english"),
        }
        loaded = {"model": model, "info": info, "speaker_to_id": speaker.get("speaker_to_id", {})}
        # Warm-up (CUDA context, cuDNN kernels) so the first request is timed like the rest.
        self._infer(loaded, "The quick brown fox jumps over the lazy dog.", 0.667, 1.0, 0,
                    speakers[0] if speakers else None)
        info["load_seconds"] = round(time.perf_counter() - started, 2)
        return loaded

    # -- text ---------------------------------------------------------------------------------

    def normalize(self, text, normalizer="english"):
        function = NORMALIZERS[normalizer]
        if function is None:
            kept = "".join(c for c in text if c in self.text_symbols)
            return kept, sorted({c for c in text if c not in self.text_symbols and not c.isspace()})
        return function(text, self.text_symbols)

    def prepare(self, text, normalizer="english"):
        if len(text) > MAX_INPUT_CHARS:
            raise ValueError(f"Text is longer than {MAX_INPUT_CHARS} characters.")

        def measure(piece):
            return len(self.normalize(piece, normalizer)[0])

        segments = []
        for sentence in text_en.split_sentences(text):
            for piece in text_en.chunk(sentence, measure, MAX_SEGMENT_CHARS):
                normalized, dropped = self.normalize(piece, normalizer)
                segments.append({"index": len(segments), "original": piece, "normalized": normalized,
                                 "dropped": dropped, "skip": not any(c.isalpha() for c in normalized)})
        return segments

    # -- synthesis ----------------------------------------------------------------------------

    def synthesize(self, model_id, text, noise_scale=0.667, length_scale=1.0, seed=None, speaker=None):
        self.load(model_id)
        if seed is None:
            seed = random.randrange(2 ** 31)
        with self.lock:
            return self._infer(self.loaded[model_id], text, float(noise_scale), float(length_scale), int(seed),
                               speaker)

    def _infer(self, loaded, text, noise_scale, length_scale, seed, speaker):
        model, info = loaded["model"], loaded["info"]
        if not 0 <= noise_scale <= 2 or not 0.25 <= length_scale <= 4:
            raise ValueError("noise_scale must be in [0, 2] and length_scale in [0.25, 4].")
        tokens = self.ns["encode_text"](text, self.symbol_to_id)
        if tokens.numel() == 0 or (tokens == self.ns["PAD_ID"]).any():
            raise ValueError("Nothing to synthesize: the text has no characters from the symbol table.")
        if info["max_tokens"] and tokens.numel() > info["max_tokens"]:
            raise ValueError(f"{tokens.numel()} characters exceed this model's limit of {info['max_tokens']}.")
        speaker_ids = None
        if loaded["speaker_to_id"]:
            if speaker not in loaded["speaker_to_id"]:
                raise ValueError(f"Choose a speaker: {sorted(loaded['speaker_to_id'])}")
            speaker_ids = torch.tensor([loaded["speaker_to_id"][speaker]], device=self.device)
        elif speaker:
            raise ValueError("This checkpoint has no speaker conditioning.")

        sync = torch.cuda.synchronize if self.device.type == "cuda" else (lambda: None)
        torch.manual_seed(seed)
        sync()
        started = time.perf_counter()
        result = model.infer(tokens.unsqueeze(0).to(self.device), torch.tensor([tokens.numel()], device=self.device),
                             noise_scale=noise_scale, length_scale=length_scale, speaker_ids=speaker_ids)
        sync()
        infer_seconds = time.perf_counter() - started
        audio = result["generated_audio"][0, 0, :int(result["audio_lengths"][0])].float()
        attention = result["attention"][0, 0]  # [output frames, text tokens], a hard monotonic path
        durations = attention.sum(dim=0).round().long().tolist()

        sample_rate = info["sample_rate"]
        if sample_rate not in self.mel_transforms:
            from sts_eval import EvaluationConfig
            from sts_eval.reconstruction import mel_transform
            config = EvaluationConfig(sample_rate=sample_rate, n_fft=self.audio["n_fft"],
                                      hop_length=self.audio["hop_length"], win_length=self.audio["win_length"],
                                      n_mels=self.audio["n_mels"])
            self.mel_transforms[sample_rate] = mel_transform(config).to(self.device)
        log_mel = self.mel_transforms[sample_rate](audio).clamp_min(1e-5).log()
        low, high = MEL_RANGE
        quantized = ((log_mel - low) / (high - low)).clamp(0, 1).mul(255).round().to(torch.uint8).cpu()
        seconds = audio.numel() / sample_rate
        return {
            "wav": base64.b64encode(wav_bytes(audio, sample_rate)).decode("ascii"),
            "sample_rate": sample_rate, "seconds": seconds, "seed": seed,
            "infer_seconds": infer_seconds, "rtf": infer_seconds / seconds if seconds else None,
            "device": str(self.device),
            "mel": {"data": base64.b64encode(quantized.numpy().tobytes()).decode("ascii"),
                    "n_mels": int(quantized.shape[0]), "frames": int(quantized.shape[1]),
                    "hop_length": self.audio["hop_length"], "range": MEL_RANGE,
                    "peak": float(log_mel.max())},
            "alignment": {"tokens": [self.symbols[i] for i in tokens.tolist()], "durations": durations,
                          "frames": int(attention.shape[0]), "hop_length": info["hop_length"],
                          "hard": bool(((attention == 0) | (attention == 1)).all())},
        }
