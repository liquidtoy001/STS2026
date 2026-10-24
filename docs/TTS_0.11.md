# TTS_0.11: TTS_0.10 without the length filter, plus chunk-aware fine-tuning

Status 2026-10-25 (Australia/Brisbane): base run training on the UQ GPU Lab L40S; fine-tune and
control queued after it. No results yet.

## What changes

TTS_0.11 keeps the TTS_0.10 model and recipe and applies only changes with evidence behind them:

| Change | Evidence |
|---|---|
| No 100-character text filter: train on 12,900 clips instead of 6,208 | The only ablation arm beyond the seed noise floor that improved WER: D1 11.6% against R0 16.9%, −5.3 points [−8.9, −2.0] ([ablation](ABLATION_TTS_0.10.md#results-2026-10-21)) |
| 100,000 steps instead of 40,000 (milestones every 10,000 kept) | None yet; the 40k and 100k checkpoints are compared directly |
| Chunk-aware fine-tuning (20,000 steps) with a control fine-tune of the same length | Local check on R0: independent 2-word chunks 24.2% against 34.1% for the control ([CHUNK_FINETUNE.md](CHUNK_FINETUNE.md)) |
| Speaking-rate correction at inference (`length_scale`) | Fine-tuned chunks were 0.81–0.89× the recording's duration; no retraining needed |

## Arm T011

`T011` in the notebook's `ARMS`: the switches of D1 (`max_text_chars` reverted, seed 2026), and the
long-sentence test set `sts_eval/splits/ljspeech_long_100.txt` (100 clips of more than 100
characters, `tools/make_heldout_split.py --long`) is held out as well. D1 trained on those clips;
T011 does not, so long sentences can be evaluated fairly. Held out: 200 clips (the 100 TTS_0.10
held-out sentences, then the 100 long ones); training clips: 12,900.

```sh
python tools/train_tts010.py --arm T011 --data-dir /path/LJSpeech-1.1 --run-root runs \
    --steps 100000 --keep-every 10000
```

Evaluate either test set (labels get a `_long` suffix and a separate `summary_long.*`):

```sh
python tools/evaluate_runs.py --runs-dir runs --eval-dir eval --data-dir /path/LJSpeech-1.1 \
    --step 100000 --split ljspeech_long_100.txt --asr-python <python with openai-whisper>
```

## Known issue in the first T011 run

The base run started on 2026-10-24 (`tts010_T011_20261024T152445372698Z`) used code from before
a fix: its in-training held-out probe decoded the first eight held-out clips in corpus order,
which include one long clip, while `run_config.json` lists the TTS_0.10 probe IDs. Only the
in-training held-out probe curve is affected; the held-out evaluation after training
(`tools/eval_tts010.py`) is not. Later runs use the listed clips.

## Where it runs

UQ GPU Lab, 1× NVIDIA L40S, PyTorch 2.11.0+cu130 (the version of the A1–L1 arms).
