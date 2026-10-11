# TTS_0.10 — benlu's model in the project harness, with switchable components

Integration: Turino. Model and training recipe: benlu (`final-vits`). Version date: 2026-10-12
(Australia/Brisbane). Status: implemented and integration-tested; released as the research
pre-release [`tts-v0.10`](https://github.com/liquidtoy001/STS2026/releases/tag/tts-v0.10) and merged into `main` on 2026-10-12. The ablation runs
([ABLATION_TTS_0.10.md](ABLATION_TTS_0.10.md)) have not started.
Source: `notebooks/vits-full-train.ipynb`, implementation commit [`c06d618`](https://github.com/liquidtoy001/STS2026/commit/c06d618).

## What changed from TTS_0.02

TTS_0.10 adopts benlu's generator, discriminator and training recipe. Each difference is a
switch whose TTS_0.02 setting can be restored for the ablation.

| Switch | TTS_0.02 | TTS_0.10 |
|---|---|---|
| `text_encoder` | `nn.TransformerEncoder`, GELU linear feed-forward | post-norm attention + kernel-3 convolutional feed-forward; embeddings scaled by √192 |
| `posterior_encoder` | 1×1 projection + 4 dilated convolutions | 513→256→192 stem, 8 gated WaveNet blocks with skip sum, log-σ clamp [−7, 2] |
| `decoder` | one kernel-3 residual block per upsampling stage | two residual blocks (kernels 3 and 7) per stage, averaged (HiFi-GAN MRF) |
| `discriminator` | one 1-D convolution stack (7.1 M parameters) | scale + period (2, 3, 5, 7, 11) discriminators, weight norm (46.7 M) |
| `feature_matching` | off | on, weight 2 |
| `recon_loss` | HTK log-mel 1 + multi-resolution STFT 0.5 + waveform L1 10 | Slaney log-mel L1 × 45 |
| `optim` | batch 8, learning rate 1e-4, fp32, gradient clipping 1.0 | batch 16, learning rate 2e-4, fp16 autocast, no clipping |
| `max_text_chars` | all clips | normalized text of 1–100 characters (6,208 training clips) |

Unchanged: flow, duration predictor, duration and KL losses, LSGAN adversarial loss (weight 1),
AdamW (betas 0.8/0.99, eps 1e-9, weight decay 0.01), character vocabulary, speaker-ID
conditioning (now added inside every posterior-encoder block). Generator parameters:
21.1 M (TTS_0.02: 12.1 M).

Intended benefit: the WaveNet posterior encoder, MRF decoder and multi-period discriminator are the
standard VITS/HiFi-GAN components for modelling waveform detail and periodicity; the mel × 45 and
feature-matching objective is the VITS training recipe.

## Implementation work added during integration

| Change | Evidence |
|---|---|
| Batch-vectorised monotonic alignment search | identical paths to benlu's loop on 40 test cases with ties; 20 ms instead of 123 ms per batch of 16; one training step 0.238 s instead of 0.377 s |
| cuDNN non-deterministic kernels (as benlu's trainer) | deterministic kernels made an R0 step 3.3× slower (0.876 s vs 0.262 s); runs are seeded, not bitwise reproducible |
| `metadata.csv` read with `csv.QUOTE_NONE` | keeps the 16 transcriptions that were lost in the TTS_0.01 runs |
| Fixed held-out split `sts_eval/splits/ljspeech_heldout_100.txt` | 100 clips (≤100 characters, not LJ001-0001…0008) removed from training in every arm |
| Checkpoints with both optimizers, the fp16 scaler, switches and speaker mapping; resume | benlu's checkpoints held weights only |
| Spectral and adversarial losses in fp32 under fp16 autocast | benlu computed the mel loss inside the fp16 region |
| Non-finite steps skipped and counted; gradient norms logged | stop after 20 consecutive non-finite steps |
| One Kaggle export per arm; `tools/train_tts010.py` runs an arm without Jupyter, including Windows DataLoader workers | |
| `tools/benchmark_tts010.py`, `tools/eval_tts010.py`, `tools/asr_wer.py`, `tools/convert_benlu_checkpoint.py` | cost measurement, central held-out evaluation, Whisper WER/CER with a natural-speech baseline, benlu checkpoint conversion |

## Verification

- With benlu's weights loaded, the TTS_0.10 training forward pass (audio, alignment, segment
  start, prior and posterior statistics) and text-only inference are **identical** to his
  archived `AudioSyn` (fixed seeds, CPU). Discriminator outputs and adversarial losses match too.
- With every switch at TTS_0.02, initial weights, training forward pass, inference, discriminator
  and loss terms are identical to the `tts-v0.02` release.
- Every arm completes training steps on synthetic tones; checkpoints round-trip; mismatched
  switches are refused on resume. 30 automated tests pass.
- The complete notebook ran on an RTX 4070 Ti SUPER for arms R0 (fp16) and E0 (fp32), stopped
  at the time budget and resumed (`tools/check_speaker_notebook.py`). On the complete LJSpeech-1.1,
  a 20-step R0 smoke run found all 100 held-out recordings, trained on 6,208 clips and ran both
  probes. These are wiring checks, not quality results (`docs/records/tts-0.10/validation.json`).

## Measured so far

All measurements below were made on 2026-10-12 on one RTX 4070 Ti SUPER (PyTorch 2.7.1+cu118).
Records: `docs/records/tts-0.10/`.

### benlu's 40,000-step checkpoint vs the TTS_0.01 runs (fixed-crop probe)

The `sts_eval` probe (LJ001-0001…0008, central 0.37 s crops, posterior mean, lower is better),
run by `tools/eval_tts010.py`. TTS_0.01 checkpoints are fed the `|X|` spectrogram they were
trained on; their mean-mode values reproduce the recorded 2026-09-29 results to four decimals.

| Checkpoint | Steps | Log-mel L1 | STFT spectral convergence | STFT log-magnitude |
|---|---:|---:|---:|---:|
| benlu TTS_0.10 | 40,000 | **0.562** | **0.548** | **0.979** |
| TTS_0.01, no GAN | 50,000 | 1.218 | 0.941 | 1.754 |
| TTS_0.01, GAN | 50,000 | 1.344 | 0.948 | 1.719 |

The TTS_0.01 runs trained on all eight probe clips; benlu's run trained on LJ001-0002/0004/0006/0008
only (the others exceed 100 characters). His error on the four clips it never saw (0.550) is no
higher than on the four it trained on (0.574). Posterior-sample values are in the records; they
depend on the GPU's random-number stream, so sample mode is compared only within one machine.

### Text-only synthesis (informal)

Same seed and noise scale 0.667 for all three checkpoints, on two LJSpeech sentences and two new
sentences ([spectrograms](records/tts-0.10/informal-synthesis/spectrograms.jpg)). benlu's
outputs show harmonics and formant structure; the TTS_0.01 outputs show a broadband buzz without
harmonics. Output RMS level: 0.043–0.058 (benlu) vs 0.013–0.020 (TTS_0.01); a reference LJSpeech
recording measures 0.083. This is a visual and level check, not a listening test or an
intelligibility measurement.

### Intelligibility of new sentences (informal, Whisper medium)

`tools/asr_wer.py` (openai-whisper 20240930, `medium`, locked decoding) on the four fixed new
sentences of `tools/eval_tts010.py`, one synthesis each. Eight real LJSpeech recordings
(LJ001-0001…0008) give the natural-speech level. Four sentences are far too few for a reliable
rate; the held-out evaluation uses 100.

| Audio | Words wrong (WER) | Characters wrong (CER) |
|---|---:|---:|
| benlu TTS_0.10, 40,000 steps | 32% | 23% |
| TTS_0.01, no GAN, 50,000 steps | 98% | 84% |
| TTS_0.01, GAN, 50,000 steps | 100% | 94% |
| Real LJSpeech recordings | 0.8% | 0.2% |

Whisper returned either nothing or "thanks for watching" for every TTS_0.01 sentence, its usual
output for non-speech; for benlu's checkpoint most errors are substitutions of similar-sounding
words. Records: `docs/records/tts-0.10/asr-informal/`.

### Training cost (`tools/benchmark_tts010.py`)

Synthetic audio with LJSpeech-like lengths, batch per arm, 40 timed steps after 10 warm-up steps.
The GPU was shared with desktop applications; repeated R0 measurements gave 0.228 and 0.262 s.

| Arm | Batch | Seconds per step | Hours for 40,000 steps | Peak memory (GiB) |
|---|---:|---:|---:|---:|
| R0 (TTS_0.10) | 16 | 0.228 | 2.5 | 3.25 |
| G1 text encoder | 16 | 0.224 | 2.5 | 3.17 |
| G2 posterior encoder | 16 | 0.217 | 2.4 | 3.12 |
| G3 decoder | 16 | 0.213 | 2.4 | 2.91 |
| A1 discriminator | 16 | 0.154 | 1.7 | 1.93 |
| A2 no feature matching | 16 | 0.235 | 2.6 | 3.19 |
| L1 reconstruction loss | 16 | 0.219 | 2.4 | 3.28 |
| O1 optimisation (fp32) | 8 | 0.197 | 2.2 | 2.95 |
| D1 no length filter | 16 | 0.267 | 3.0 | 3.85 |
| E0 (all TTS_0.02) | 8 | 0.139 | 1.6 | 1.41 |

With deterministic cuDNN kernels R0 took 0.876 s per step (3.3× slower); TTS_0.10 does not use them.

## Limitations

- No TTS_0.10 run has used this harness yet; the only trained TTS_0.10 weights are benlu's
  40,000-step checkpoint, trained with his notebook on all ≤100-character clips, so the
  held-out clips were part of its training data.
- Whisper word error rate (`tools/asr_wer.py`) has not been run on held-out sentences: the complete
  LJSpeech recordings are not yet on the evaluation machine.
- Single speaker (LJSpeech). Speaker-ID conditioning is wired into the new modules but untrained.

## Provenance

benlu's inputs (received 2026-10-11): `final-vits.ipynb` (sha256 `633790cb…`), module archive
`Archive.zip` (`b8434a41…`), checkpoint `weight_tts_epoch5.zip` (`61827f4f…`, step 40,000; the
file name counts saves every 8,000 steps). The notebook and modules are archived unchanged in
`docs/records/tts-0.10/benlu-source/`. benlu trained the 40,000 steps in about 10 hours on Kaggle
with the T4 ×2 accelerator (reported by benlu; his trainer uses one device).
