# Speak Before the Sentence Ends: a compact VITS TTS for voice agents

Code, experiment records and documentation for the STAT3007 Group G7 project report
*Speak Before the Sentence Ends*. The repository is named STS2026.

## What the project is

A voice agent answers in three stages: speech recognition turns the user's speech into text, a
language model writes the reply word by word, and a text-to-speech (TTS) model speaks it. A TTS
that needs the whole sentence can start only when the language model has finished that sentence.
This project asks how few words a TTS can wait for and still be as intelligible as when it
synthesizes the whole sentence.

The model is a compact VITS trained on LJSpeech (one speaker, 22,050 Hz). Two methods are studied:

- **Context-aware chunked synthesis.** Each call gives the text encoder all previous words, the
  current chunk and the next K words (the lookahead). The durations of every character outside
  the chunk are set to zero, so only the chunk receives audio frames.
- **Chunk-aware fine-tuning.** A trained checkpoint is fine-tuned on training clips cut at word
  boundaries, so it learns to speak short pieces without the rest of the sentence. A control
  fine-tune of the same length on whole clips separates the effect of the cutting from the effect
  of the extra steps.

## Main results

Intelligibility is the word error rate (WER) of Whisper medium on 100 held-out LJSpeech
sentences that no model trained on, with ASR protocol v2 (`tools/asr_wer.py --protocol v2`). The
LJSpeech recordings of the same sentences score 3.3%.

| Model | Whisper WER |
|---|---:|
| TTS_0.10 arm R0, 40,000 steps | 16.9% |
| **TTS_0.11, 100,000 steps** | **11.2%** |
| TTS_0.11 on 100 held-out sentences longer than 100 characters (recordings 2.4%) | 8.4% |

Chunked synthesis with TTS_0.11 after 20,000 steps of chunk-aware fine-tuning (seed 2026):

| Condition | Words the TTS waits for | Whisper WER |
|---|---:|---:|
| whole sentence | about 12 | 10.3% |
| **context-aware 1-word chunks, lookahead 2** | **3** | **10.9%** |
| independent 1-word chunks (control fine-tune → chunk-aware fine-tune) | 1 | 44.4% → 25.8% |
| independent 2-word chunks (control fine-tune → chunk-aware fine-tune) | 2 | 23.0% → 12.5% |

- Without context, chunk-aware fine-tuning roughly halves the errors of 1- and 2-word chunks; a
  second fine-tuning seed (2027) gives the same result. Whole-sentence WER did not change
  detectably.
- One chunk is fast to synthesize: the first 1-word chunk with lookahead 2 takes 14 ms on an
  RTX 4070 Ti SUPER and 38 ms on 8 CPU threads; a whole sentence takes 30 ms and 237 ms.
- The leave-one-out ablation of TTS_0.10's eight changes (11 runs, seed noise floor 3.3 points)
  found one change that improved WER: removing the 100-character text filter. TTS_0.11 is
  TTS_0.10 without that filter, trained for 100,000 steps.

Full tables, 95% paired-bootstrap intervals and the second seed: [TTS_0.11](docs/TTS_0.11.md).

**Limits.** Whisper WER measures intelligibility, not naturalness; no listening test or MOS
evaluation has been run. No real language model or speech recognizer has been in the loop, so
"words waited for" stands in for latency. One speaker, one seed for each base model, 100
sentences per test set. The raw evaluation files of the TTS_0.11 runs were lost with a temporary
GPU disk; their numbers were recorded before (see [TTS_0.11](docs/TTS_0.11.md)).

## Versions

| Version | Date | What changed |
|---|---|---|
| TTS_0.01 ([`tts-v0.01`](https://github.com/liquidtoy001/STS2026/releases/tag/tts-v0.01)) | 2026-09-30 | full-corpus LJSpeech training of the first VITS notebook, with and without GAN loss |
| TTS_0.02 ([`tts-v0.02`](https://github.com/liquidtoy001/STS2026/releases/tag/tts-v0.02)) | 2026-10-07 | optional speaker-ID conditioning |
| TTS_0.10 ([`tts-v0.10`](https://github.com/liquidtoy001/STS2026/releases/tag/tts-v0.10)) | 2026-10-12 | benlu's `final-vits` model and training recipe, each change switchable for the ablation |
| TTS_0.11 (arm `T011`) | 2026-10-25 | no text-length filter, 100,000 steps, 100 long sentences held out; chunk-aware fine-tuning |
| TTS_0.20 candidates (arms `V1`–`V5`) | 2026-10-25 | phoneme input, windowed text encoder, stochastic durations, LibriTTS-R pretraining; none beat TTS_0.11 |

benlu built the TTS_0.01 and TTS_0.02 prototypes and the TTS_0.10 model and recipe; Turino integrated
each version into this repository. Dates are in Australia/Brisbane time. Trained checkpoints are not
included in this repository.

## Reproduce

**Environment.** Python with a matching PyTorch and TorchAudio pair from the
[PyTorch installation selector](https://pytorch.org/get-started/locally/), then:

```sh
python -m pip install -r requirements/notebook.txt
```

WER needs [openai-whisper](https://github.com/openai/whisper) (the results used version 20240930,
model `medium`); `--asr-python` can point to a separate Python that has it. Phoneme-input arms
also need `g2p_en`. The TTS_0.11 runs used PyTorch 2.11.0+cu130 on one NVIDIA L40S.

**Data.** The complete [LJSpeech-1.1](https://keithito.com/LJ-Speech-Dataset/) with
`metadata.csv` and `wavs/`. The held-out sentence lists are in `sts_eval/splits/`.

**Train and evaluate TTS_0.11.** The model is defined in `notebooks/vits-full-train.ipynb`;
`tools/train_tts010.py` runs it outside Jupyter.

```sh
python tools/train_tts010.py --arm T011 --data-dir /path/LJSpeech-1.1 --run-root runs \
    --steps 100000 --keep-every 10000
python tools/evaluate_runs.py --runs-dir runs --eval-dir eval --data-dir /path/LJSpeech-1.1 \
    --step 100000 --asr-protocol v2 --asr-python <python with openai-whisper>
```

Add `--split ljspeech_long_100.txt` to `evaluate_runs.py` for the long-sentence test set.

**Chunk-aware fine-tuning.** Compute the per-character durations of the trained checkpoint once,
then fine-tune it with and without chunk-aware training:

```sh
python tools/chunk_alignments.py --checkpoint <run>/checkpoints/vits_step_100000.pt \
    --data-dir /path/LJSpeech-1.1 --out durations.json
python tools/train_tts010.py --arm T011 --data-dir /path/LJSpeech-1.1 --run-root runs --tag ftchunk \
    --init-from <run>/checkpoints/vits_step_100000.pt --steps 20000 \
    --chunk-prob 0.5 --chunk-durations durations.json
python tools/train_tts010.py --arm T011 --data-dir /path/LJSpeech-1.1 --run-root runs --tag ftctrl \
    --init-from <run>/checkpoints/vits_step_100000.pt --steps 20000
```

`--seed 2027` gives the second seed. Options and defaults: [chunk-aware fine-tuning](docs/CHUNK_FINETUNE.md).

**Chunk study.** Synthesize the 100 held-out sentences under 17 chunking conditions, transcribe
them and compute WER with paired-bootstrap intervals:

```sh
python tools/chunk_study.py --checkpoint <checkpoint.pt> --data-dir /path/LJSpeech-1.1 --out study/model
python tools/asr_wer.py study/model --model medium --protocol v2
python tools/chunk_analysis.py analyze study/model --asr-protocol v2
python tools/chunk_analysis.py compare --study control=study/ftctrl chunk=study/ftchunk \
    --reference control --asr-protocol v2 --out comparison.json
```

Whisper takes several seconds per utterance on a CPU, so one study (1,800 utterances) takes
hours. Method and conditions: [TTS for voice agents](docs/VOICE_AGENT_TTS.md).

**Tests.**

```sh
python -m unittest discover -s tests -v
```

The tests run on synthetic inputs; they check wiring, equivalences and tools, not speech quality.

**Web UI.** `python webui/server.py` serves a local page at `http://127.0.0.1:8000/` with live
synthesis and an Agent mode that streams a simulated language-model reply into up to three
chunking strategies. It needs trained checkpoints; see [the web UI guide](docs/WEBUI.md).

## Where the records are

- [`docs/`](docs/README.md): one page per version and study, with the measured numbers and the
  commands that produced them.
- [`docs/records/`](docs/records/): small, versioned evidence: run configurations, metrics,
  Whisper transcripts and scores, bootstrap comparisons and figures. Large files (datasets,
  checkpoints, audio) are not in Git.
- [`docs/site/index.html`](docs/site/index.html): a static project site with the versions,
  architecture, results and timeline. GitHub shows its source; clone the repository and open the
  file in a browser.

## Repository layout

```text
notebooks/vits-full-train.ipynb   model, training and evaluation (TTS_0.01 to TTS_0.20)
notebooks/vits-light_v1.ipynb     the first posterior-reconstruction notebook (historical)
tools/                            training, evaluation, chunk study, data preparation
sts_eval/                         reconstruction metrics, held-out splits, phoneme table
webui/                            local TTS page with the voice-agent demo
tests/                            unit and equivalence tests
docs/                             documentation, records and the project site
```

## License

No open-source license is granted. The repository is shared for the STAT3007 course.
