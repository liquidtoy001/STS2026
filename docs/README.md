# Documentation

Start with the [repository README](../README.md): what the project is, the main results and how
to reproduce them. The pages below record each study in full: the protocol, the measured
numbers, what was not tested, and the commands that produced them. Dates are in
Australia/Brisbane time.

## Voice-agent TTS: the main study

| Page | Contents |
|---|---|
| [TTS for voice agents](VOICE_AGENT_TTS.md) | The problem, the chunk study (17 chunking conditions on 100 held-out sentences), the results for TTS_0.10 R0 and the first 5,000-step chunk-aware fine-tune |
| [TTS_0.11](TTS_0.11.md) | The final model: results on both test sets, the chunk study on TTS_0.11, chunk-aware fine-tuning with two seeds, synthesis times, ASR protocol v2 |
| [Chunk-aware fine-tuning](CHUNK_FINETUNE.md) | The training options (`--init-from`, `--chunk-prob`, `--chunk-durations`, ...) and the local 5,000-step check |
| [Web UI](WEBUI.md) | The local TTS page and its Agent mode, a simulated voice-agent stream |

## Model versions

| Page | Contents |
|---|---|
| [TTS_0.10](TTS_0.10.md) | benlu's `final-vits` model and recipe in the project harness, each change switchable |
| [TTS_0.10 ablation](ABLATION_TTS_0.10.md) | Leave-one-out ablation of TTS_0.10's eight changes: 11 runs, seed noise floor, results |
| [TTS_0.20 candidates](TTS_0.20.md) | Phoneme input, windowed text encoder, stochastic durations and LibriTTS-R pretraining; none beat TTS_0.11 |
| [TTS_0.02: speaker-ID conditioning](SPEAKER_CONDITIONING.md) | Optional speaker IDs, speaker manifests, folding a speaker into a single-speaker model |
| [TTS_0.01: full training, GAN vs no GAN](FULL_TRAINING.md) | The first full-corpus LJSpeech runs, with and without GAN loss |

## Early reconstruction experiments

Before the TTS work, the project measured posterior reconstruction (spectrogram → posterior
encoder → decoder → waveform) on eight fixed LJSpeech training clips. These are training-batch
measurements, not held-out evaluation.

| Page | Contents |
|---|---|
| [Reconstruction evaluation](EVALUATION.md) | The shared evaluator and its metrics |
| [Local GPU run](LOCAL_EVALUATION.md) and [its result](LOCAL_BASELINE_RESULT.md) | The first measured baseline |
| [Evaluator validation](EVALUATION_VALIDATION.md) | Implementation checks of the evaluator |
| [Original vs new encoder/decoder](JACQUES_EVAL.md) | A second reconstruction study |

## Setup and background

- [Environment and dataset setup for the reconstruction notebook](SETUP.md). For the TTS models,
  see [Reproduce](../README.md#reproduce).
- [ASR and VITS evaluation guide](guides/ASR_VITS_Evaluation_Guide.pdf): study notes written
  while planning the evaluation (September 2026), not a report of measured results.

## Records

[`records/`](records/) holds small, versioned evidence. Datasets, checkpoints and audio are not in
Git. One user name in a data path (`tts-0.10-ablation/R0b/run_config.json`) is replaced with
`<user>`; the records are otherwise unchanged.

| Folder | Contents |
|---|---|
| [`tts-0.10-ablation/`](records/tts-0.10-ablation/) | Per-arm run configurations, metrics and Whisper scores; [summary](records/tts-0.10-ablation/summary.md); bootstrap comparisons with R0 |
| [`chunk-study/R0_40k/`](records/chunk-study/R0_40k/) | Chunk study of R0: analysis, Whisper counts, timing, figure |
| [`chunk-finetune/`](records/chunk-finetune/) | The local 5,000-step fine-tunes: run configurations, evaluations, comparisons |
| [`tts-0.10/`](records/tts-0.10/) | TTS_0.10 validation, step-time benchmarks, informal synthesis and ASR checks |
| [`tts-0.20/libritts-r/`](records/tts-0.20/libritts-r/) | LibriTTS-R preparation records |
| [`tts-0.02/`](records/tts-0.02/) | TTS_0.02 validation |
| [`full-train-gan/`](records/full-train-gan/) | TTS_0.01 GAN vs no-GAN runs |
| [`jacques-enc-dec/`](records/jacques-enc-dec/) | The encoder/decoder reconstruction study |

The TTS_0.11 numbers are recorded in [TTS_0.11](TTS_0.11.md); the raw evaluation files of those
runs were lost with a temporary GPU disk.

## Project site

[`site/index.html`](site/index.html) is a static site with the versions, architecture, results
and timeline. GitHub shows its source; clone the repository and open the file in a browser.
