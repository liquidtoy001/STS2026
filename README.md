# STS2026

## Project overview

STS2026 explores a VITS-based path from posterior-conditioned speech reconstruction to text-to-speech (TTS) and, ultimately, speech-to-speech voice conversion. The long-term goal is to take speech from a source speaker and generate it in a target speaker's voice while preserving the words and prosody. Reconstruction is the first stage; the full-training notebook now also implements text-only inference and optional learned speaker-ID conditioning.

The project studies the trade-offs among content preservation, target-speaker similarity and audio quality under different training objectives. Potential applications include assistive communication, accessible reading, audiobooks, virtual assistants, dubbing, personalized speech and speaker anonymization.

The current work includes a measured reconstruction baseline on eight utterances and full-LJSpeech training runs both with and without GAN loss. These use different protocols and should be interpreted separately. Held-out TTS and voice-conversion performance have not yet been established. Planned evaluation includes reconstruction measures such as SI-SDR and STOI, and voice-conversion measures such as ASR word error rate, F0 and speaker-embedding similarity.

## Latest model source: TTS_0.10

**2026-10-12 · integration: Turino · model and recipe: benlu · pre-release [`tts-v0.10`](https://github.com/liquidtoy001/STS2026/releases/tag/tts-v0.10), merged to `main`.**
TTS_0.10 adopts benlu's
`final-vits` model and training recipe (WaveNet posterior encoder, multi-receptive-field decoder,
multi-period + scale discriminator, mel × 45 with feature matching, batch 16 in fp16, text of at
most 100 characters) inside the project's training harness, and keeps TTS_0.02 speaker-ID
conditioning. Every change is a switch that can revert one component to TTS_0.02.

- With benlu's 40,000-step weights loaded, its training forward pass and inference are identical
  to his original code; with every switch at TTS_0.02 it is identical to the `tts-v0.02` release.
- 100 LJSpeech clips are held out from all training (`sts_eval/splits/ljspeech_heldout_100.txt`).
- benlu's 40,000-step checkpoint scores log-mel L1 0.562 on the fixed LJ001 probe, against
  1.218–1.344 for the 50,000-step TTS_0.01 runs, and its text-only output shows the harmonic
  structure that TTS_0.01 lacks. On four new sentences Whisper misrecognises 32% of its words,
  against 98–100% for TTS_0.01 (real recordings: 0.8%); held-out evaluation has not been run.
- Engineering: batch-vectorised alignment search and non-deterministic cuDNN kernels; a TTS_0.10
  training step takes about 0.23 s on an RTX 4070 Ti SUPER (40,000 steps ≈ 2.5 h).
- No run of this harness has been trained yet. The team's leave-one-out ablation (11 runs, five
  trainers) measures which of the changes matter.

See [TTS_0.10](docs/TTS_0.10.md) and the [ablation plan](docs/ABLATION_TTS_0.10.md).
The previous release, [`tts-v0.02`](https://github.com/liquidtoy001/STS2026/releases/tag/tts-v0.02),
added [speaker-ID conditioning](docs/SPEAKER_CONDITIONING.md).

## Notebook timeline

Dates below are Brisbane-local dates from the project history. The repository currently tracks two notebook source files; earlier stages are preserved in Git history, while Kaggle exports (one per TTS_0.10 ablation arm) are generated from the full-training source.

| Date | Notebook stage | Purpose |
| --- | --- | --- |
| 2026-09-17 (`451c635`) | Initial posterior-reconstruction prototype (historically named `vits-light_v1.ipynb` at the repository root) | First model prototype, before the shared evaluation function. It is not the measured baseline. |
| 2026-09-20 (`c9f258c`) | First evaluated baseline (`notebooks/vits-light_v1.ipynb`) | Adds the shared reconstruction evaluator. The original comparison overfits eight fixed LJSpeech training utterances for 2,000 iterations with GAN disabled. This is the project's earliest evaluated baseline. |
| 2026-09-23–24 (`07e663d`) | GAN-enabled revision of `notebooks/vits-light_v1.ipynb` | Extends the same notebook line with a discriminator and adversarial training. The recorded 2026-09-20 baseline remains the earlier GAN-disabled experiment; identify results by their configuration. |
| 2026-09-30 (`3a5724b`) | Full-training notebook (`notebooks/vits-full-train.ipynb`) | Extends the model to full-corpus LJSpeech training with the complete training objective and evaluation/checkpoint workflow. The builder creates separate GAN-off and GAN-on Kaggle exports, differing only in `GAN_WEIGHT`; the exports are generated files, not additional tracked source notebooks. |
| 2026-10-07 (`tts-v0.02`) | TTS_0.02 speaker-ID conditioning (`notebooks/vits-full-train.ipynb`) | Adds opt-in speaker conditions throughout training, inference, evaluation and checkpoint resume. Source/integration validated; no trained multi-speaker checkpoint or quality claim. |
| 2026-10-12 | TTS_0.10 (`notebooks/vits-full-train.ipynb`) | benlu's model and recipe in the project harness, with switches that restore each TTS_0.02 component, a 100-clip held-out split and one export per ablation arm. Integration verified; ablation training not yet run. |

The full-training runs use 50,000 iterations and a different objective and data protocol from the eight-utterance baseline, so their metrics are not a direct baseline comparison. Both protocols evaluate fixed training utterances; neither establishes held-out quality.

The `vits-light_v1.ipynb` path continued to evolve after the baseline. Use commit `c9f258c` in Git history when you need the exact evaluated-baseline notebook; the current file contains later changes.

## Repository structure

```text
STS2026/
  notebooks/
    vits-light_v1.ipynb       # Reconstruction baseline and later GAN-enabled revision
    vits-full-train.ipynb     # Full-corpus training source
  sts_eval/                   # Shared reconstruction metrics and runner
    splits/                   # Fixed held-out split (TTS_0.10)
  tools/                      # Kaggle export, report and data-preparation utilities
  webui/                      # Local TTS visualiser: python webui/server.py
  tests/                      # Behavioral and artifact checks
  requirements/
    notebook.txt              # Unpinned notebook dependency inventory
    docs.txt                  # Documentation build and QA dependencies
  docs/
    SETUP.md                  # Environment and dataset details
    EVALUATION.md             # Evaluation workflow
    FULL_TRAINING.md          # Full-training and Kaggle workflow
    TTS_0.10.md               # TTS_0.10 changes, verification and provenance
    ABLATION_TTS_0.10.md      # Ablation arms, owners and run instructions
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

To train a TTS_0.10 arm (see [the ablation plan](docs/ABLATION_TTS_0.10.md)) on a GPU machine with the complete LJSpeech-1.1:

```sh
python tools/train_tts010.py --arm R0 --data-dir /path/LJSpeech-1.1
```

To fine-tune a finished checkpoint, optionally on word chunks for chunked synthesis, see [chunk-aware fine-tuning](docs/CHUNK_FINETUNE.md) and its [run manual](docs/CHUNK_FINETUNE_RUNBOOK.md). Why chunked synthesis matters for voice agents, and how R0 performs when it speaks a few words at a time: [TTS for voice agents](docs/VOICE_AGENT_TTS.md).

For Kaggle, `python tools/build_kaggle_full_train.py` writes one export per arm under `output/kaggle/`. Import it, attach LJSpeech-1.1, enable a GPU, and run all cells. Generated notebooks, datasets, checkpoints and audio are experiment artifacts and should not be committed.

## Local TTS visualiser

`webui/` is a local web page for live text-to-speech with trained checkpoints. It needs only the notebook environment (PyTorch, TorchAudio, NumPy) and Python's standard-library HTTP server. With the project environment active, run from the repository root:

```sh
python webui/server.py
```

The server loads the default checkpoint, serves `http://127.0.0.1:8000/` and opens it in the browser. The page has two tabs.

**Agent mode** demonstrates TTS for a voice agent ([docs/VOICE_AGENT_TTS.md](docs/VOICE_AGENT_TTS.md)). A simulated language model streams a reply word by word, after a set time to first token and at a set speed; no real LLM runs, so a run is offline and repeatable. Up to three strategies speak the same reply at the same time with real TTS calls: the whole sentence, independent N-word chunks, or context-aware N-word chunks with lookahead K (the conditions of the chunk study; `/api/synthesize_chunk` uses the same synthesis as `tools/chunk_study.py`). Chunks are queued back to back with Web Audio; one strategy is audible during the run, and afterwards each can be replayed, alone or as an A/B pair. The timeline shows when words arrive, each TTS call and the playback, with the time to first audio and stalls (the audio ran out before the next chunk was ready). The result cards show those live measurements next to each strategy's Whisper WER from the chunk study (`docs/records/chunk-study/R0_40k/analysis.json`), which is never computed live; two charts show the study's WER against waiting and how the waiting saved depends on the LLM speed and the sentence length. The three strategies share one GPU, so their calls queue behind each other.

**Reader** reads any English text: the text is split into sentences, each sentence is synthesised and played, and the current sentence and word are highlighted. A second checkpoint can be selected for A/B comparison. For the current sentence the page shows the mel spectrogram, the text-to-frame alignment returned by `model.infer(...)["attention"]` and the inference time; it also shows each checkpoint's measured WER and downloads the read text as one WAV file.

The model list is `webui/config.example.json`. Paths are relative to the repository root and may be glob patterns (the newest match is used); a checkpoint that is not found is listed as unavailable. To change the list, copy the file to `webui/config.json` (git-ignored) and edit the labels, paths, descriptions and WER values. A new checkpoint, such as a chunk-aware fine-tune, is one more entry in `models`; its chunk-study results appear in Agent mode once an entry in `chunk_studies` points to its `analysis.json` (entries whose file does not exist are skipped). `agent` sets the preset replies, the default strategies and the default stream settings. Each checkpoint is rebuilt from the switches and speaker configuration stored in it, so a speaker-conditioned checkpoint gets a speaker selector; it must use the notebook's symbol table. Text preparation (`webui/text_en.py`) writes out numbers, money, times, percentages and common abbreviations in the LJSpeech style; it does not handle acronyms, roman numerals, units, dates with slashes or URLs.

Options: `--port`, `--device cpu`, `--config PATH`, `--no-browser`, and `--host` (default `127.0.0.1`; the API has no authentication).
