"""Checkpoint registry and text-only synthesis behind the web UI.

Models are listed in a JSON config (webui/config.json, or the committed config.example.json).
Each checkpoint is rebuilt from the switches and speaker configuration stored in it, with
the notebook's own class definitions, so a run trained on another dataset or with speaker
conditioning loads without code changes as long as it uses the notebook's symbol table.

Chunked synthesis for the voice-agent demo follows tools/chunk_study.py (synth and spans): the
text encoder reads the chunk with its context, and only the chunk's characters receive frames.
"""

import base64
import glob
import io
import json
import math
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


def chunk_span(words, start, end, lookahead, context):
    """Encoder text and the voiced character span of words[start:end], as in tools/chunk_study.spans.

    Context mode reads all earlier words, the chunk and up to `lookahead` following words; the
    span includes the space after the chunk unless the chunk ends the text. Independent mode
    reads the chunk alone.
    """
    if not 0 <= start < end <= len(words):
        raise ValueError(f"Chunk words [{start}, {end}) outside the {len(words)} words received.")
    if lookahead < 0 or (lookahead and not context):
        raise ValueError("lookahead must be >= 0, and 0 for independent chunks.")
    low = 0 if context else start
    text = " ".join(words[low:end + lookahead])
    begin = len(" ".join(words[low:start])) + (1 if start > low else 0)
    finish = begin + len(" ".join(words[start:end])) + (1 if end < len(words) else 0)
    return text, (begin, finish)


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
        self.studies = {}
        for study in self.config.get("chunk_studies", []):
            path = self.find_record(study["analysis"])
            if path is None:
                continue  # e.g. a study whose Whisper scoring has not finished
            analysis = json.loads(path.read_text(encoding="utf-8"))
            timing_path = self.find_record(study.get("timing", ""))
            timing = json.loads(timing_path.read_text(encoding="utf-8")) if timing_path else {}
            self.studies[study["model"]] = {
                "label": study.get("label"), "source": study.get("source"), "analysis": str(path),
                "sentences": analysis.get("sentences"), "natural_wer": analysis.get("natural_wer"),
                "conditions": analysis["conditions"],
                "mean_words": timing.get("mean_words"),
                "gpu_ms_per_call": {name: row.get("gpu_ms_per_call_mean")
                                    for name, row in timing.get("summary", {}).items()},
            }

    def find_record(self, pattern):
        """A study file (path or glob) in the repository, e.g. docs/records, or under the checkpoint root."""
        if not pattern:
            return None
        for base in (ROOT, self.checkpoint_root):
            path = resolve_checkpoint(pattern, base)
            if path is not None:
                return path
        return None

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
            "agent": self.config.get("agent", {}),
            "chunk_studies": self.studies,
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

    def _check(self, loaded, text, noise_scale, length_scale, speaker):
        """Validate settings; return the encoded tokens and the speaker IDs (or None)."""
        if not 0 <= noise_scale <= 2 or not 0.25 <= length_scale <= 4:
            raise ValueError("noise_scale must be in [0, 2] and length_scale in [0.25, 4].")
        tokens = self.ns["encode_text"](text, self.symbol_to_id)
        if tokens.numel() == 0 or (tokens == self.ns["PAD_ID"]).any():
            raise ValueError("Nothing to synthesize: the text has no characters from the symbol table.")
        limit = loaded["info"]["max_tokens"]
        if limit and tokens.numel() > limit:
            raise ValueError(f"{tokens.numel()} characters exceed this model's limit of {limit}.")
        if loaded["speaker_to_id"]:
            if speaker not in loaded["speaker_to_id"]:
                raise ValueError(f"Choose a speaker: {sorted(loaded['speaker_to_id'])}")
            return tokens, torch.tensor([loaded["speaker_to_id"][speaker]], device=self.device)
        if speaker:
            raise ValueError("This checkpoint has no speaker conditioning.")
        return tokens, None

    def _sync(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def _infer(self, loaded, text, noise_scale, length_scale, seed, speaker):
        model, info = loaded["model"], loaded["info"]
        tokens, speaker_ids = self._check(loaded, text, noise_scale, length_scale, speaker)
        sync = self._sync
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

    # -- chunked synthesis (voice-agent demo) -------------------------------------------------

    def synthesize_chunk(self, model_id, text, start, end, lookahead=0, mode="context", seed=None,
                         noise_scale=0.667, length_scale=1.0, speaker=None):
        """Audio of words[start:end] of `text` (the words received so far, already normalised).

        mode "context": the encoder reads all earlier words, the chunk and `lookahead` following
        words (as many as have arrived); "independent": the chunk alone. As in the chunk study,
        the chunk's seed is seed + start, so seed 1234 reproduces the study's audio.
        """
        if mode not in ("context", "independent"):
            raise ValueError("mode must be 'context' or 'independent'.")
        words = text.split()
        encoder_text, span = chunk_span(words, int(start), int(end), int(lookahead), mode == "context")
        self.load(model_id)
        if seed is None:
            seed = random.randrange(2 ** 31 - 10_000)
        chunk_seed = int(seed) + int(start)
        with self.lock:
            audio, infer_seconds = self._infer_span(self.loaded[model_id], encoder_text, span, float(noise_scale),
                                                    float(length_scale), chunk_seed, speaker)
        sample_rate = self.loaded[model_id]["info"]["sample_rate"]
        return {
            "wav": base64.b64encode(wav_bytes(audio, sample_rate)).decode("ascii"),
            "sample_rate": sample_rate, "seconds": audio.numel() / sample_rate, "seed": chunk_seed,
            "infer_seconds": infer_seconds, "device": str(self.device),
            "encoder_text": encoder_text, "voiced": encoder_text[span[0]:span[1]].strip(),
        }

    def _infer_span(self, loaded, text, span, noise_scale, length_scale, seed, speaker):
        """tools/chunk_study.synth with speaker conditioning and length_scale; returns (audio, seconds)."""
        model = loaded["model"]
        tokens, speaker_ids = self._check(loaded, text, noise_scale, length_scale, speaker)
        kept = [i for i, c in enumerate(text) if c in self.symbol_to_id]
        target = torch.tensor([[[1.0 if span[0] <= i < span[1] else 0.0 for i in kept]]], device=self.device)
        tokens = tokens.unsqueeze(0).to(self.device)
        lengths = torch.tensor([tokens.size(1)], device=self.device)
        self._sync()
        started = time.perf_counter()
        with torch.inference_mode():
            g = model.speaker_condition(speaker_ids, 1)
            torch.manual_seed(seed)
            x, m_p, logs_p, x_mask = model.text_encoder(tokens, lengths)
            logw = model.duration_predictor(x, x_mask, g=g).clamp(min=-5.0, max=math.log(50))
            w = torch.ceil(torch.exp(logw) * length_scale) * x_mask * target
            y_len = w.sum(dim=(1, 2)).long().clamp(min=1)
            y_mask = self.ns["create_sequence_mask"](y_len, int(y_len.max())).unsqueeze(1).to(x.dtype)
            attn = self.ns["make_inference_path"](w, y_mask.unsqueeze(-1) * x_mask.unsqueeze(2))
            m = self.ns["expand_with_alignment"](m_p, attn)
            logs = self.ns["expand_with_alignment"](logs_p, attn)
            z_p = m + torch.randn_like(m) * torch.exp(logs) * noise_scale
            z = model.flow(z_p, y_mask, reverse=True, g=g)
            audio = model.decoder(z * y_mask, g=g)[0, 0, :int(y_len) * model.hop_length].float()
        self._sync()
        return audio, time.perf_counter() - started
