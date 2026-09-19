"""Fixed-segment posterior evaluation; this does not measure text-to-speech quality.

The model adapter contract is deliberately small: posterior_encoder, decoder,
hop_length, segment_frames, and segment_samples. No training forward pass or MAS
is needed. Inputs use the original notebook's six-tensor batch format.
"""

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import platform
import subprocess

import torch
import torchaudio


@dataclass(frozen=True)
class EvaluationConfig:
    sample_rate: int = 22050
    n_fft: int = 1024
    hop_length: int = 256
    win_length: int = 1024
    n_mels: int = 80
    segment_frames: int = 32
    seed: int = 1234
    # Match the baseline loss resolutions, with explicit transform semantics.
    resolutions: tuple = ((512, 128, 512), (1024, 256, 1024), (2048, 512, 2048))


def _fingerprint(batch):
    digest = hashlib.sha256()
    for tensor in batch:
        value = tensor.detach().cpu().contiguous()
        digest.update(str((tuple(value.shape), str(value.dtype))).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


class ReconstructionEvaluator:
    """Own a frozen batch and crop plan, then evaluate any compatible model.

    Each metric is calculated on one valid (unpadded) waveform at a time.
    Summary values are utterance means, not duration-weighted means.
    Both posterior-mean and seeded posterior-sample reconstructions are saved.
    """

    def __init__(self, batch, sample_ids, output_dir, *, config=None, provenance=None):
        self.config = config or EvaluationConfig()
        c = self.config
        if min(c.sample_rate, c.n_fft, c.hop_length, c.win_length,
               c.n_mels, c.segment_frames) <= 0 or c.win_length > c.n_fft:
            raise ValueError("Invalid evaluation transform configuration.")
        if not c.resolutions or any(not 0 < w <= n or h <= 0 for n, h, w in c.resolutions):
            raise ValueError("Invalid STFT resolutions.")
        if len(batch) != 6:
            raise ValueError("Expected the notebook's six-tensor batch.")
        self.batch = tuple(t.detach().cpu().clone() for t in batch)
        self.sample_ids = list(sample_ids)
        size = self.batch[2].shape[0]
        if size == 0 or len(self.sample_ids) != size or len(set(self.sample_ids)) != size:
            raise ValueError("Provide one unique sample ID per batch item.")
        if any(t.shape[0] != size for t in self.batch):
            raise ValueError("Batch dimensions differ.")
        if self.batch[2].ndim != 3 or self.batch[4].ndim != 3 or self.batch[4].shape[1] != 1:
            raise ValueError("Expected spectrogram [B,F,T] and mono audio [B,1,T].")
        if any(not torch.isfinite(t).all() for t in self.batch):
            raise ValueError("Batch contains non-finite values.")
        spec_lengths, wav_lengths = self.batch[3], self.batch[5]
        if (spec_lengths <= 0).any() or (spec_lengths > self.batch[2].shape[-1]).any():
            raise ValueError("Invalid spectrogram lengths.")
        if (wav_lengths <= 0).any() or (wav_lengths > self.batch[4].shape[-1]).any():
            raise ValueError("Invalid waveform lengths.")
        # Use only frames which also map to real waveform samples.
        usable_frames = torch.minimum(spec_lengths, wav_lengths // c.hop_length)
        self.starts = ((usable_frames - c.segment_frames).clamp(min=0) // 2).long()
        self.lengths = torch.minimum(
            wav_lengths - self.starts * c.hop_length,
            torch.minimum(spec_lengths - self.starts, torch.full_like(spec_lengths, c.segment_frames))
            * c.hop_length,
        ).long()
        # Reflect-padded STFT requires input longer than half the largest FFT.
        self.minimum_samples = max(c.n_fft, *(r[0] for r in c.resolutions)) // 2 + 1
        if (self.lengths < self.minimum_samples).any():
            raise ValueError(f"Evaluation needs at least {self.minimum_samples} valid samples per crop.")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=False)
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=c.sample_rate, n_fft=c.n_fft, hop_length=c.hop_length,
            win_length=c.win_length, n_mels=c.n_mels, power=1.0,
            center=True, pad_mode="reflect", norm=None, mel_scale="htk",
        )
        self.history = []
        self._last_step = -1
        self.manifest = {
            "protocol": "posterior_fixed_batch_reconstruction_v1",
            "scope": "training_batch_overfit_not_held_out",
            "config": asdict(c), "batch_sha256": _fingerprint(self.batch),
            "sample_ids": self.sample_ids, "start_frames": self.starts.tolist(),
            "valid_samples": self.lengths.tolist(),
            "aggregation": "arithmetic mean over utterances, separately by mode",
            "environment": {"python": platform.python_version(), "torch": torch.__version__,
                            "torchaudio": torchaudio.__version__},
            "provenance": provenance or {},
        }
        _write_json(self.output_dir / "manifest.json", self.manifest)

    def score(self, generated, target):
        """Score equal-length 1-D waveforms. Call after removing padding."""
        if generated.ndim != 1 or generated.shape != target.shape:
            raise ValueError("Metrics require aligned, equal-length 1-D audio.")
        if generated.numel() < self.minimum_samples:
            raise ValueError("Audio too short for configured STFT.")
        if not torch.isfinite(generated).all() or not torch.isfinite(target).all():
            raise FloatingPointError("Non-finite evaluation audio.")
        generated, target = generated.float(), target.float()
        self.mel.to(generated.device)
        mel_l1 = (self.mel(generated).clamp_min(1e-5).log()
                  - self.mel(target).clamp_min(1e-5).log()).abs().mean()
        convergence, log_magnitude = [], []
        for n_fft, hop, win in self.config.resolutions:
            window = torch.hann_window(win, device=generated.device)
            spectra = [torch.stft(x, n_fft, hop, win, window, center=True,
                                 pad_mode="reflect", return_complex=True).abs().clamp_min(1e-7)
                       for x in (generated, target)]
            pred, ref = spectra
            convergence.append(torch.linalg.vector_norm(pred - ref)
                               / torch.linalg.vector_norm(ref).clamp_min(1e-7))
            log_magnitude.append((pred.log() - ref.log()).abs().mean())
        result = {"log_mel_l1": mel_l1.item(),
                  "stft_spectral_convergence": torch.stack(convergence).mean().item(),
                  "stft_log_magnitude": torch.stack(log_magnitude).mean().item()}
        if not all(torch.isfinite(torch.tensor(v)) for v in result.values()):
            raise FloatingPointError("Non-finite evaluation metric.")
        return result

    @torch.inference_mode()
    def evaluate(self, model, step, *, save_audio=False):
        """Recompute outputs without perturbing RNG or any module's train/eval flag."""
        if step <= self._last_step:
            raise ValueError("Evaluation steps must increase; use a new directory for a new run.")
        model = model.module if isinstance(model, torch.nn.DataParallel) else model
        c = self.config
        for name, expected in (("hop_length", c.hop_length), ("segment_frames", c.segment_frames),
                               ("segment_samples", c.hop_length * c.segment_frames)):
            if getattr(model, name) != expected:
                raise ValueError(f"Model {name} differs from the evaluation protocol.")
        device = next(model.parameters()).device
        if device.type not in ("cpu", "cuda"):
            raise ValueError("This evaluator currently supports CPU and CUDA.")
        flags = [(module, module.training) for module in model.modules()]
        # torch.manual_seed affects all CUDA generators: fork all visible devices.
        devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
        records, audio = [], {}
        try:
            model.eval()
            with torch.random.fork_rng(devices=devices), torch.autocast(device.type, enabled=False):
                torch.manual_seed(c.seed)
                spec = self.batch[2].to(device)
                lengths = self.batch[3].to(device)
                _, mean, log_scale, mask = model.posterior_encoder(spec, lengths)
                if mean.shape != log_scale.shape or mean.shape[0] != len(self.sample_ids):
                    raise ValueError("Posterior statistics have incompatible shapes.")
                # Generate evaluation noise independently of encoder internals.
                generator = torch.Generator(device=device).manual_seed(c.seed)
                noise = torch.randn(mean.shape, device=device, dtype=mean.dtype, generator=generator)
                for mode in ("mean", "sample"):
                    latent = mean if mode == "mean" else mean + noise * log_scale.exp()
                    latent = latent * mask
                    segments = []
                    for i, start in enumerate(self.starts.tolist()):
                        crop = latent[i:i + 1, :, start:start + c.segment_frames]
                        segments.append(torch.nn.functional.pad(crop, (0, c.segment_frames - crop.shape[-1])))
                    generated = model.decoder(torch.cat(segments))
                    expected_shape = (len(self.sample_ids), 1, c.segment_frames * c.hop_length)
                    if tuple(generated.shape) != expected_shape:
                        raise ValueError(f"Decoder returned {tuple(generated.shape)}, expected {expected_shape}.")
                    if not torch.isfinite(generated).all():
                        raise FloatingPointError("Decoder produced non-finite audio.")
                    for i, sample_id in enumerate(self.sample_ids):
                        start = int(self.starts[i]) * c.hop_length
                        length = int(self.lengths[i])
                        target = self.batch[4][i, 0, start:start + length].to(device)
                        pred = generated[i, 0, :length]
                        records.append({"step": step, "sample_id": sample_id, "mode": mode,
                                        "start_frame": int(self.starts[i]), "valid_samples": length,
                                        **self.score(pred, target)})
                        if save_audio:
                            audio[f"{i:03d}_{mode}"] = pred.cpu()
                            audio[f"{i:03d}_target"] = target.cpu()
        finally:
            for module, flag in flags:
                module.training = flag
        summaries = []
        for mode in ("mean", "sample"):
            rows = [r for r in records if r["mode"] == mode]
            summaries.append({"step": step, "mode": mode, "count": len(rows),
                              **{key: sum(r[key] for r in rows) / len(rows)
                                 for key in ("log_mel_l1", "stft_spectral_convergence", "stft_log_magnitude")}})
        step_dir = self.output_dir / f"step_{step:06d}"
        step_dir.mkdir(exist_ok=False)
        _write_json(step_dir / "per_sample.json", records)
        _write_json(step_dir / "summary.json", summaries)
        # Float WAV preserves raw amplitude; clipping is never applied before scoring.
        for name, waveform in audio.items():
            torchaudio.save(str(step_dir / f"{name}.wav"), waveform.unsqueeze(0), c.sample_rate,
                            encoding="PCM_F", bits_per_sample=32)
        self.history.extend(summaries)
        _write_json(self.output_dir / "evaluation_history.json", self.history)
        self._last_step = step
        return summaries


def repository_provenance(root):
    """Capture local revision and dirty state; never claim an uncommitted run is clean."""
    def git(*args):
        result = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True)
        return result.stdout.strip() if result.returncode == 0 else None
    return {"commit": git("rev-parse", "HEAD"), "working_tree": git("status", "--short")}


def plot_history(evaluation_history, training_history, output_path):
    """Keep training objectives and evaluation modes visibly separate."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    if training_history:
        for key in ("total", "mel", "stft", "wave"):
            axes[0].plot([r["step"] for r in training_history],
                         [r[key] for r in training_history], label=key)
        axes[0].legend()
    axes[0].set_title("Training losses (random crops)")
    for axis, metric in zip(axes[1:], ("log_mel_l1", "stft_spectral_convergence")):
        for mode in ("mean", "sample"):
            rows = [r for r in evaluation_history if r["mode"] == mode]
            axis.plot([r["step"] for r in rows], [r[metric] for r in rows], marker=".", label=mode)
        axis.set_title(f"Fixed-crop evaluation: {metric}")
        axis.legend()
    for axis in axes:
        axis.set_xlabel("Training iteration (AMP may skip updates)")
        axis.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    return fig
