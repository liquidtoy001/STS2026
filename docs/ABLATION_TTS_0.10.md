# TTS_0.10 ablation: which changes made the difference

Status 2026-10-21: ten of eleven arms trained and evaluated centrally (R0, G1–G3, A1, A2, L1,
O1, D1, E0; [results](#results-2026-10-21)); R0b trained, re-upload pending, so the noise floor
is not yet known. Plan agreed by the team on 2026-10-12 (Australia/Brisbane).
Model: [TTS_0.10](TTS_0.10.md), pre-release [`tts-v0.10`](https://github.com/liquidtoy001/STS2026/releases/tag/tts-v0.10). Source: `notebooks/vits-full-train.ipynb`.

## Question

benlu's 40,000-step TTS_0.10 checkpoint produces speech-like spectrograms (harmonics and
formants) where the 50,000-step TTS_0.01 runs do not, but TTS_0.10 changes eight things at once. This study asks how much each change contributes. A full factorial design
would need 2⁸ = 256 runs; we use a **leave-one-out** design instead: start from the complete
TTS_0.10 and revert exactly one change to its TTS_0.02 form. The loss in quality when a change is
removed is that change's contribution in the presence of all the others.

Three reference runs make the result interpretable:

- **R0** — complete TTS_0.10. Every other arm is compared with it.
- **R0b** — R0 with a different seed. The R0/R0b difference is the **noise floor**: an arm whose
  difference from R0 is smaller than this cannot be distinguished from chance.
- **E0** — every switch at TTS_0.02. E0 → R0 is the total improvement under identical training
  conditions (steps, data split, harness).

## Arms and owners

All arms: 40,000 steps, LJSpeech-1.1 minus the 100 held-out clips, the same code commit.

| Arm | Reverts to TTS_0.02 | What it tests | Owner | Hardware |
|---|---|---|---|---|
| R0 | nothing (seed 2026) | reference | assigned by the team | L40S 48 GB |
| R0b | nothing (seed 2027) | noise floor | assigned by the team | same GPU type as R0 |
| G1 | `text_encoder` | conv-FFN Transformer text encoder | assigned by the team | L40S 48 GB |
| G2 | `posterior_encoder` | gated WaveNet posterior encoder | assigned by the team | L40S 48 GB |
| G3 | `decoder` | multi-receptive-field decoder | assigned by the team | L40S 48 GB |
| A1 | `discriminator` | multi-period + scale discriminator | assigned by the team | L40S 48 GB |
| A2 | `feature_matching` | feature-matching loss | assigned by the team | L40S 48 GB |
| L1 | `recon_loss` | mel ×45 instead of mel + STFT + waveform | assigned by the team | L40S 48 GB |
| O1 | `optim` | batch 16, lr 2e-4, fp16 vs batch 8, lr 1e-4, fp32 | assigned by the team | L40S 48 GB |
| E0 | all eight | total improvement | Turino | RTX 4070 Ti SUPER 16 GB |
| D1 | `max_text_chars` | the ≤100-character filter | Turino | RTX 4070 Ti SUPER 16 GB |

Measure your own GPU with `tools/benchmark_tts010.py` before starting. The team assigns the remaining arms and the central evaluation; Turino runs E0 and D1.
The report chapters follow the arm groups: reference and noise floor (R0, R0b), generator
architecture (G1–G3), adversarial training (A1, A2), loss and optimisation (L1, O1), and the
TTS_0.02 baseline and data filter (E0, D1).

## Before you start (every trainer)

1. Get the code: `git fetch origin --tags && git checkout tts-v0.10` (the release every arm uses;
   later commits only change documentation). Check with `git rev-parse --short HEAD`; every arm
   must use the same commit (it is also written to
   `run_config.json`).
2. Environment: Python 3.10+ with PyTorch and TorchAudio (CUDA build), then
   `python -m pip install -r requirements/notebook.txt`. librosa is not needed.
3. Data: the complete LJSpeech-1.1 (`metadata.csv` and 13,100 files in `wavs/`). Check
   `sha256sum metadata.csv` = `852bce5c79e5184ba5bf2ea98aad4bcf49a5d787def5fd1f04fa2ab8aa6dc535`.
   The notebook refuses to train if any recording is missing.
4. Five-minute speed check on your GPU (writes `output/tts010-benchmark.json`):

   ```sh
   python tools/benchmark_tts010.py --metadata /path/LJSpeech-1.1/metadata.csv --arms G1 G2 G3
   ```

   It prints seconds per step and the expected hours for 40,000 steps. Send the JSON to Turino;
   it becomes the cost column for your arms.

## Run an arm

From the repository root, one command per arm (Linux GPU server, inside `tmux` or `screen`):

```sh
python tools/train_tts010.py --arm G1 --data-dir /path/LJSpeech-1.1 --run-root runs 2>&1 | tee runs/G1.log
```

The run folder is printed at the start: `runs/tts010_G1_<UTC time>/`. Do not pass any other
setting; the arm fixes everything else. On a 48 GB GPU several arms may run at the same time
(about 3.3 GB each; results are unaffected, and per-arm speed is measured separately with the benchmark).

**Metered GPU time:** add `--time-budget-hours <hours you can still spend minus 0.5>`; the run
saves a checkpoint and stops cleanly before that, and can be resumed. Shut the instance down as
soon as `status.json` says `completed` and the run folder is copied off.

**Kaggle instead:** `python tools/build_kaggle_full_train.py` writes `output/kaggle/tts010_<arm>.ipynb`.
Import it, add LJSpeech-1.1 as input, enable a GPU, Save & Run All (Commit).

**Smoke test (optional, 1 minute):** add `--smoke-steps 20 --run-root runs-smoke`. It trains 20 steps
and exercises evaluation, checkpointing and sampling. Never report a smoke run.

## While it runs

- Every 1,000 steps the log prints the losses, `s/it` and the ETA. Every 2,000 steps it prints
  the probe and held-out-probe log-mel L1.
- Expect the first few steps of fp16 arms to be skipped (`amp_backoffs`); that is normal.
- `nonfinite_skips` should stay at 0. Twenty consecutive non-finite steps stop the run with a
  `failed_or_interrupted` status — report it, do not restart silently.
- If the run stops at the time budget (11.3 h; only on slow GPUs) or the machine stops, resume:

  ```sh
  python tools/train_tts010.py --arm G1 --data-dir /path/LJSpeech-1.1 --run-root runs \
      --resume runs/tts010_G1_<UTC time>
  ```

  A resumed run continues from its latest checkpoint (every 5,000 steps) and records the resume.

## After the run

1. Check `status.json`: `"status": "completed"`, `"step": 40000`.
2. Upload the **whole run folder** to the team Google Drive under
   `tts010-ablation/<arm>/` (checkpoints at 10k/20k/30k/40k, samples, both evaluation folders,
   training history, `curves.png`, your log).
3. Post the run card below in the team chat (copy, fill, send), including the Drive path.
4. Listen to `samples/step_40000_sample_{1,2}.wav` and write one or two sentences of notes.

```text
Arm:                      (e.g. G1)
Executed by:              (name)
Hardware:                 (GPU model, memory; machine / Kaggle)
Commit:                   (git rev-parse --short HEAD)
Run folder:               (tts010_<arm>_<UTC time>)
Started / completed UTC:  (from run_config.json / status.json)
Wall time:                (status.json session_wall_seconds; sum over resumes)
Status:                   (completed / stopped / failed) at step N
Resumed:                  (no / yes: from step N)
Deviations:               (anything that differed from this document, or "none")
Listening notes:          (optional, English)
```

## Central evaluation

All checkpoints are evaluated on one machine with one script, so differences come from training,
not from evaluation:

```sh
python tools/eval_tts010.py --checkpoint <run>/checkpoints/vits_step_40000.pt \
    --label G1_40k --data-dir /path/LJSpeech-1.1
python tools/asr_wer.py output/tts010-eval/G1_40k --model medium
```

To evaluate every finished run in a folder (any depth, e.g. a Drive download) and rebuild the
comparison table in one command, re-runnable as runs arrive:

```sh
python tools/evaluate_runs.py --runs-dir <runs> --eval-dir output/tts010-eval \n    --data-dir /path/LJSpeech-1.1 --asr-python <python with openai-whisper>
```

It writes `summary.md`/`.csv`/`.json`: each arm's difference from R0, flagged when it exceeds the
R0/R0b difference.

`asr_wer.py` follows the team's ASR back-transcription protocol: a locked Whisper model
(`transcribe`, `language=en`, `temperature=0`, beam 5, no prompt), one shared text normalisation,
corpus WER/CER with substitution/deletion/insertion rates, and the LJSpeech recording of every
held-out sentence as the natural-speech baseline, reported as ΔWER = WER(synthesized) − WER(natural).
It needs only `openai-whisper`, NumPy and SciPy, so it can run in a separate Python environment.

| Measure | What it shows |
|---|---|
| Held-out full-utterance reconstruction: log-mel L1, STFT metrics | posterior encoder + decoder on unseen audio |
| Held-out text-to-speech: Whisper WER/CER and ΔWER vs the natural recordings | intelligibility of text-only synthesis |
| Held-out text-to-speech: duration ratio, level ratio | speaking rate and loudness vs the recording |
| Fixed-crop probes (LJ001, held-out) | continuity with the earlier reports |
| Listening set: 2 held-out + 4 new sentences | perceived quality (team listening notes) |
| Seconds per step, peak memory (benchmark tool, one GPU) | training cost |

Records go to `docs/records/tts-0.10-ablation/<arm>/`
(`run_config.json`, `status.json`, evaluation histories, log, metrics) and the report to the
documentation site.

## Results (2026-10-21)
<a id="results-2026-10-21"></a>

Central evaluation with `tools/evaluate_runs.py` on one machine (RTX 4070 Ti SUPER, Whisper
medium on CPU), 100 held-out sentences. The LJSpeech recordings of those sentences score 3.3%
WER. Every run: 40,000 steps, seed 2026, status `completed`, no non-finite steps, the same
notebook (sha256 `c366da6b…`, identical to `tts-v0.10`). Records:
[`docs/records/tts-0.10-ablation/`](records/tts-0.10-ablation/) (`summary.md` and one folder per
arm with `run_config.json`, `status.json`, `metrics.json`, `asr.json`).

| Arm | Reverted to TTS_0.02 | Whisper WER | WER − R0 | CER | Held-out log-mel L1 | Duration ratio | Empty ASR outputs | GPU |
|---|---|---:|---:|---:|---:|---:|---:|---|
| R0 | nothing | 16.9% | – | 8.2% | 0.533 | 0.957 | 0 | RTX PRO 6000 |
| G1 | text encoder | 23.7% | +6.8 pt | 12.6% | 0.517 | 0.912 | 0 | RTX PRO 6000 |
| G2 | posterior encoder | 24.7% | +7.8 pt | 14.8% | 0.572 | 0.965 | 0 | RTX PRO 6000 |
| G3 | decoder | 13.8% | −3.1 pt | 7.6% | 0.511 | 0.977 | 0 | RTX PRO 6000 |
| A1 | discriminator | 19.2% | +2.3 pt | 9.9% | 0.558 | 0.961 | 0 | RTX PRO 6000 |
| A2 | feature matching (off) | 15.7% | −1.2 pt | 8.2% | 0.509 | 0.978 | 0 | RTX PRO 6000 |
| L1 | reconstruction loss | 186.9% | +169.9 pt | 130.5% | 1.147 | 0.775 | 48 | RTX PRO 6000 |
| O1 | optimisation | 28.5% | +11.6 pt | 16.0% | 0.540 | 0.933 | 0 | RTX PRO 6000 |
| D1 | length filter (off) | 11.6% | −5.3 pt | 5.8% | 0.519 | 1.002 | 0 | RTX 4070 Ti SUPER |
| E0 | all eight | 100.2% | +83.2 pt | 87.3% | 1.518 | 0.725 | 17 | RTX 4070 Ti SUPER |

- Reverting the reconstruction loss (L1) is the only single change that breaks the model: WER
  above 100% (more inserted words than reference words), 48 of 100 sentences with no ASR output,
  and a held-out log-mel L1 of 1.147, closer to E0 (1.518) than to R0 (0.533).
- Reverting the optimisation (O1), the posterior encoder (G2), the text encoder (G1) or the
  discriminator (A1) raises WER by 2–12 points.
- Reverting the decoder (G3), switching feature matching off (A2) and removing the length filter
  (D1) give lower WER than R0.
- R0b is not evaluated, so differences of a few points cannot yet be separated from seed
  variation; each arm has one seed.

## Reading the results

- Compare every arm with R0. A difference smaller than |R0 − R0b| is "no detectable effect".
- Leave-one-out contributions are not additive and can interact: removing two changes may
  cost more or less than the sum of removing each.
- Steps are fixed, not epochs: O1 and E0 (batch 8) see half as many clips; D1 and E0 train on
  13,000 clips including long ones.
- One seed per arm except R0; one speaker; one dataset. The result explains this model on
  LJSpeech, not TTS in general.

## Report outline

1. TTS_0.02 does not produce speech-like audio; TTS_0.10 does (spectrograms, samples).
2. E0 → R0: total improvement under identical conditions.
3. Leave-one-out bars for each change, with the R0/R0b noise floor marked.
4. Cost: time and memory per arm.
5. Limits.
