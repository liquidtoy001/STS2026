# Chunk-aware fine-tuning of TTS_0.10

Integration: Turino. Date: 2026-10-19 (Australia/Brisbane). Status: implemented and tested on
branch `Turino`; not merged or released. Model: TTS_0.10 (benlu's recipe), arm R0.

A voice agent can start speaking before the language model has finished a sentence by
synthesizing the text a few words at a time. Synthesized that way, R0 at 40,000 steps is much
harder to understand than when it synthesizes the whole sentence
([VOICE_AGENT_TTS.md](VOICE_AGENT_TTS.md)). This page describes a way to fine-tune a finished
checkpoint on word chunks and the local check of that idea, and links the manual for the
full-size runs.

## What was added

| Option (`tools/train_tts010.py`) | Notebook variable | Meaning |
|---|---|---|
| `--init-from <checkpoint>` | `STS_INIT_FROM` | start a **new** run from a finished checkpoint of the same arm (switches and speaker mapping must match); this is not a resume |
| `--fresh-optimizer` | `STS_INIT_OPTIMIZER=fresh` | with `--init-from`: new AdamW moments and fp16 scaler; by default both optimizers' moments and the scaler are carried over, so the fine-tune continues the original optimisation |
| `--steps N` | `STS_TOTAL_STEPS` | steps of this run; a run started with `--init-from` counts from 0 |
| `--tag NAME` | `STS_RUN_TAG` | run folder `tts010_<arm>_<tag>_<UTC time>` |
| `--keep-every N` | `STS_KEEP_EVERY` | keep the checkpoints at multiples of N (saved every 5,000 steps; default 10,000; the latest is always kept) |
| `--chunk-prob P` | `STS_CHUNK_PROB` | chunk-aware training: crop each clip with probability P (default 0 = off) |
| `--chunk-durations FILE` | `STS_CHUNK_DURATIONS` | per-character durations from `tools/chunk_alignments.py` |
| `--chunk-words MIN-MAX` | `STS_CHUNK_WORDS` | words per crop (default `1-8`) |
| `--chunk-prefix-share S` | `STS_CHUNK_PREFIX_SHARE` | share of crops that start at the first word (default 0.5) |

**Chunk-aware training.** With probability P a training clip is cut to whole words together with
the matching audio: a prefix of the sentence (share S) or a span that starts at a random word,
MIN–MAX words long, extended by whole words until it lasts at least one decoder segment
(32 frames, 0.37 s). Spaces around the kept words are dropped. Prefixes resemble the text a
voice agent has so far; spans resemble a chunk synthesized on its own. The cut points come from
the alignment of a trained model, computed once: `tools/chunk_alignments.py` runs the R0 40k
model's monotonic alignment search on the posterior mean (no sampling) for every clip except the
100 held-out ones and stores the number of spectrogram frames of every character. During
training the alignment search still aligns each cropped clip itself.

**Records.** `run_config.json` stores `init_from` (checkpoint path, SHA-256, step, optimizer
carried or fresh) and `chunk_training` (all settings plus the durations file's SHA-256 and source
checkpoint); every checkpoint of a run started with `--init-from` stores `init_from` too. A
stopped fine-tune is resumed with its original command, `--resume <run folder>` replacing
`--init-from`; the resumed run keeps the original `init_from`. A resume with different steps or
chunk settings, or with both `--resume` and `--init-from`, is refused.

**Unchanged when off.** With `--chunk-prob 0` (the default) the dataset draws no random numbers
and returns exactly what the `tts-v0.10` release returns; one training step gives identical losses
and weights to the release. The equivalence tests against benlu's `AudioSyn` and
multi-period discriminator and against the `tts-v0.02` release still pass (39 tests in total,
8 of them new: word crops, crop audio, off-path equivalence, alignment durations equal the
training alignment, `--init-from`, chunk text spans, waiting words, chunk analysis).

**Chunk study in the repository.** `tools/chunk_study.py` (synthesis of the 17 chunk conditions)
and `tools/chunk_analysis.py` (`analyze` one study, `compare` checkpoints sentence by sentence)
replace the scratch scripts of the R0 study: they reproduce its WAVs bit for bit (3 sentences ×
17 conditions checked) and its `analysis.json` exactly.

**Fixed on the way.** `tools/train_tts010.py` imported pyplot before selecting the headless
backend, so on a desktop (Windows) the final plot opened a window and the process did not exit
until it was closed. The backend is now chosen before the import. `tools/asr_wer.py` stopped
after 355 of 500 utterances when a Whisper hypothesis contained a character the Windows console
code page (GBK) cannot encode; its console output now escapes such characters (scoring is
unchanged).

## Local check, 2026-10-19

Two runs on one RTX 4070 Ti SUPER, both started from R0 at 40,000 steps
(`tts010_R0_20261014T070647303632Z`, optimizer state carried) for 5,000 more steps with the R0
seed, so both saw the same clips in the same order:

| Run | Chunk training | Wall time | Seconds per step (median) | Skipped fp16 steps | Non-finite steps |
|---|---|---:|---:|---:|---:|
| chunk (`tts010_R0_ftchunk5k_20261019T034438400533Z`) | p = 0.5, 1–8 words, prefix share 0.5 | 1,215 s | 0.240 | 3 | 0 |
| control (`tts010_R0_ftctrl5k_20261019T040523871800Z`) | off | 1,275 s | 0.251 | 3 | 0 |

Both runs shared the CPU with a Whisper job, which slows data loading and the CPU-side alignment
search; the step times are therefore upper bounds for this GPU. Records:
[`docs/records/chunk-finetune/`](records/chunk-finetune/).

**Whole sentences** (`tools/eval_tts010.py` + `tools/asr_wer.py`, Whisper medium, the 100
held-out sentences, one synthesis each, same seeds for all three checkpoints):

| Held-out, 100 sentences | R0 40k | control +5k | chunk +5k |
|---|---:|---:|---:|
| Whisper WER (%) | 16.9 | 17.2 | 17.4 |
| Duration ratio (1 = natural) | 0.957 | 0.936 | 0.916 |
| Level ratio (1 = natural) | 0.898 | 0.884 | 0.844 |
| Reconstruction log-mel L1 | 0.533 | 0.540 | 0.518 |

WER difference, points [95% CI]: control − R0 +0.3 [-3.1, +3.3]; chunk − R0 +0.5 [-3.3, +4.5]; chunk − control +0.3 [-2.9, +3.5]. The LJSpeech recordings of the same sentences: 3.3%.

One chunk-model output (LJ040-0096, "Pic said, quote,", 1.0 s) was transcribed as nothing and
counts as three deletions.

**Chunked synthesis** (the chunk study: the same 100 held-out sentences, conditions and seeds as
the R0 study; Whisper medium; corpus WER; 95% intervals from 2,000 paired bootstrap resamples of
the sentences). *Independent* chunks: every N-word chunk synthesized on its own and the audio
concatenated. *Context-aware* chunks: the text encoder sees all previous words, the chunk and K
following words, and only the chunk's characters receive frames.

Duration of the synthesized speech relative to the recording (1 = the recording's length; all
17 conditions in the records):

| Condition | R0 40k | control +5k | chunk +5k |
|---|---:|---:|---:|
| whole sentence | 0.96 | 0.94 | 0.92 |
| independent 1-word chunks | 1.31 | 1.28 | 0.81 |
| independent 2-word chunks | 1.13 | 1.11 | 0.87 |
| independent 4-word chunks | 1.02 | 1.00 | 0.89 |
| independent 8-word chunks | 0.98 | 0.96 | 0.91 |
| context-aware 2-word chunks, no lookahead | 1.11 | 1.08 | 0.87 |
| context-aware 2-word chunks, lookahead 2 | 0.96 | 0.93 | 0.92 |

The Whisper WER of these conditions for the two fine-tunes was still being scored on 2026-10-19
(Whisper medium on the CPU, about 4 hours); R0's WER for every condition is in
[VOICE_AGENT_TTS.md](VOICE_AGENT_TTS.md).

So far: chunk-aware training did not detectably change whole-sentence intelligibility (chunk −
control +0.3 points, interval −2.9 to +3.5). It removed the stretching of short independent
chunks (2-word chunks 0.87× instead of 1.11×) but overshot: short chunks and whole sentences
became shorter than the recordings (whole sentences 0.92× against 0.94× for the control). Its
held-out reconstruction error was the lowest of the three (0.518 against 0.540 and 0.533).

Limits: one seed per run, 5,000 steps, one speaker, Whisper WER as the intelligibility measure
(no listening test). Neither R0 nor the fine-tunes trained on the held-out sentences.

## Full-size runs

The step-by-step manual for the two 20,000-step runs (chunk and control) on an L40S or
RTX PRO 6000 server, Google Colab or Kaggle T4 ×2, with checks, expected time, what to upload
and a run card: [CHUNK_FINETUNE_RUNBOOK.md](CHUNK_FINETUNE_RUNBOOK.md). Why the line exists and
what R0 does with chunked text: [VOICE_AGENT_TTS.md](VOICE_AGENT_TTS.md).
