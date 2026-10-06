# STS2026

## Project overview

STS2026 explores a VITS-based path from posterior-conditioned speech reconstruction to text-to-speech (TTS) and, ultimately, speech-to-speech voice conversion. The long-term goal is to take speech from a source speaker and generate it in a target speaker's voice while preserving the words and prosody. Reconstruction is the first stage; the full-training notebook now also implements text-only inference and optional learned speaker-ID conditioning.

The project studies the trade-offs among content preservation, target-speaker similarity and audio quality under different training objectives. Potential applications include assistive communication, accessible reading, audiobooks, virtual assistants, dubbing, personalized speech and speaker anonymization.

The current work includes a measured reconstruction baseline on eight utterances and full-LJSpeech training runs both with and without GAN loss. These use different protocols and should be interpreted separately. Held-out TTS and voice-conversion performance have not yet been established. Planned evaluation includes reconstruction measures such as SI-SDR and STOI, and voice-conversion measures such as ASR word error rate, F0 and speaker-embedding similarity.

## Latest model source: TTS_0.02

**2026-10-07 · Turino · [Release `tts-v0.02`](https://github.com/liquidtoy001/STS2026/releases/tag/tts-v0.02).**
This research pre-release adds **closed-set speaker-ID conditioning** to the
full-training notebook. It does not include trained multi-speaker weights.

- A learned embedding conditions the posterior encoder, forward/inverse flow,
  duration predictor and waveform decoder during training and text-only inference.
- A speaker manifest supplies named identities; the batch carries their numeric
  IDs. Checkpoints and run records retain the exact name-to-ID mapping and
  conditioning dimensions. Resume rejects incompatible mappings.
- Reconstruction evaluation and generated listening samples use the same IDs.
  Both standalone Kaggle exports include the updated model and evaluator.
- Default LJSpeech mode remains unconditional. Regression checks against
  `tts-v0.01` found identical state-dictionary keys, initialized weights, training
  forward outputs and inference outputs with fixed CPU inputs and RNG seeds.

**Validation:** 19 automated tests passed. The complete notebook also executed on
an RTX 4070 Ti SUPER with two synthetic tones, saved after one optimizer update,
resumed for the second update, and wrote conditioned evaluation/audio artifacts.
These are implementation checks, **not speech-quality or speaker-similarity results**.
Real multi-speaker training, held-out TTS, reference-audio conditioning, zero-shot
cloning and voice conversion remain unverified or unimplemented.

See the [setup, architecture and checkpoint contract](docs/SPEAKER_CONDITIONING.md)
and [HTML version report](docs/site/index.html#version-speaker-conditioning).

## Notebook timeline

Dates below are Brisbane-local dates from the project history. The repository currently tracks two notebook source files; earlier stages are preserved in Git history, while Kaggle-specific GAN variants are generated from the full-training source.

| Date | Notebook stage | Purpose |
| --- | --- | --- |
| 2026-09-17 (`451c635`) | Initial posterior-reconstruction prototype (historically named `vits-light_v1.ipynb` at the repository root) | First model prototype, before the shared evaluation function. It is not the measured baseline. |
| 2026-09-20 (`c9f258c`) | First evaluated baseline (`notebooks/vits-light_v1.ipynb`) | Adds the shared reconstruction evaluator. The original comparison overfits eight fixed LJSpeech training utterances for 2,000 iterations with GAN disabled. This is the project's earliest evaluated baseline. |
| 2026-09-23–24 (`07e663d`) | GAN-enabled revision of `notebooks/vits-light_v1.ipynb` | Extends the same notebook line with a discriminator and adversarial training. The recorded 2026-09-20 baseline remains the earlier GAN-disabled experiment; identify results by their configuration. |
| 2026-09-30 (`3a5724b`) | Full-training notebook (`notebooks/vits-full-train.ipynb`) | Extends the model to full-corpus LJSpeech training with the complete training objective and evaluation/checkpoint workflow. The builder creates separate GAN-off and GAN-on Kaggle exports, differing only in `GAN_WEIGHT`; the exports are generated files, not additional tracked source notebooks. |
| 2026-10-07 (`tts-v0.02`) | TTS_0.02 speaker-ID conditioning (`notebooks/vits-full-train.ipynb`) | Adds opt-in speaker conditions throughout training, inference, evaluation and checkpoint resume. Source/integration validated; no trained multi-speaker checkpoint or quality claim. |

The full-training runs use 50,000 iterations and a different objective and data protocol from the eight-utterance baseline, so their metrics are not a direct baseline comparison. Both protocols evaluate fixed training utterances; neither establishes held-out quality.

The `vits-light_v1.ipynb` path continued to evolve after the baseline. Use commit `c9f258c` in Git history when you need the exact evaluated-baseline notebook; the current file contains later changes.

## Repository structure

```text
STS2026/
  notebooks/
    vits-light_v1.ipynb       # Reconstruction baseline and later GAN-enabled revision
    vits-full-train.ipynb     # Full-corpus training source
  sts_eval/                   # Shared reconstruction metrics and runner
  tools/                      # Kaggle export, report and data-preparation utilities
  tests/                      # Behavioral and artifact checks
  requirements/
    notebook.txt              # Unpinned notebook dependency inventory
    docs.txt                  # Documentation build and QA dependencies
  docs/
    SETUP.md                  # Environment and dataset details
    EVALUATION.md             # Evaluation workflow
    FULL_TRAINING.md          # Full-training and Kaggle workflow
    records/                  # Small, versioned experiment evidence
    site/                     # Static project-site source
  output/                     # Generated notebooks, reports and run artifacts (local)
  data/                       # Dataset files (local; do not commit)
  checkpoints/                # Model weights (local; do not commit)
```

## Collaboration agreements

- Coordinate one primary editor before changing a shared notebook. Keep changes focused and review them through a pull request before they become the shared baseline.
- Write experiment logs and measured-result reports in English. Record a completion timestamp in ISO 8601 UTC and name its source. Include the code revision, configuration, seed, dataset and split, environment, metrics, artifact location, and anything that was not tested.
- Keep overfit-training results distinct from held-out evaluation. State the protocol alongside every metric so results from different objectives or datasets are not presented as directly comparable.
- Keep raw datasets, checkpoints, credentials and bulk generated audio out of Git. Commit only small, reviewed records and artifacts needed to explain or reproduce a result.
- Clear notebook outputs before committing unless a small output is necessary to review the change. Report checks that were actually run; syntax or import checks do not establish successful training or audio quality.

## Quick start

This repository provides research notebooks rather than a deployed inference service. The following steps prepare a local environment for the reconstruction notebook.

```sh
git clone https://github.com/liquidtoy001/STS2026.git
cd STS2026
```

Create and activate a virtual environment. On Windows PowerShell:

```powershell
py -3 -m venv .venv
.venv\Scripts\Activate.ps1
```

On macOS or Linux:

```sh
python3 -m venv .venv
source .venv/bin/activate
```

Install a matching PyTorch and TorchAudio pair for your Python version, operating system and accelerator using the [official PyTorch installation selector](https://pytorch.org/get-started/locally/). Then install the remaining notebook packages and launch JupyterLab:

```sh
python -m pip install --upgrade pip
python -m pip install -r requirements/notebook.txt
python -m pip check
python -m jupyterlab --notebook-dir=notebooks
```

`requirements/notebook.txt` is an unpinned import inventory, not a tested lockfile; install PyTorch and TorchAudio separately before using it. The notebook records Python 3.12.13 as environment provenance, not as a verified supported-version range. The recorded local CUDA baseline used PyTorch/TorchAudio 2.7.1+cu118 on an RTX 4070 Ti SUPER. Full training was run on Kaggle with a Tesla T4; other platforms, including Apple MPS, are not verified.

For documentation build and QA utilities, install the separate dependencies with `python -m pip install -r requirements/docs.txt`.

For `vits-light_v1.ipynb`, obtain LJSpeech-1.1 through the team's approved source. The expected layout is `LJSpeech-1.1/metadata.csv` and `LJSpeech-1.1/wavs/<file_id>.wav`. Update `DATA_DIR` in the notebook to that directory; its checked-in default is a Kaggle path. The notebook uses 22,050 Hz audio and runs a 2,000-iteration overfit experiment on eight fixed training utterances.

To prepare the full-training Kaggle notebooks, run this from the repository root:

```sh
python tools/build_kaggle_full_train.py
```

The command writes GAN-off and GAN-on exports under `output/kaggle/`. Import each export into Kaggle, attach LJSpeech-1.1, enable a GPU, and run all cells. The full run is compute-intensive; inspect its configuration and time budget before starting it. Generated notebooks, datasets, checkpoints and audio are experiment artifacts and should not be committed.
