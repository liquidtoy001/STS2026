# TTS_0.11: TTS_0.10 without the length filter, plus chunk-aware fine-tuning

Status 2026-10-27 (Australia/Brisbane): trained and evaluated (results below). The seminar deck of
30/10 presents TTS_0.11 with chunk-aware fine-tuning.

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

## Results (25–26/10)

Whisper medium WER, ASR protocol v2 (`tools/asr_wer.py --protocol v2`, see below). Test sets:
`ljspeech_heldout_100.txt` (100 sentences, recordings 3.3%) and `ljspeech_long_100.txt` (100
sentences over 100 characters, recordings 2.4%). One seed for the base run.

| Checkpoint | 100 test sentences | 100 long sentences |
|---|---|---|
| TTS_0.10 R0, 40,000 steps | 16.9% | – |
| T011, 40,000 steps | 12.6% | 10.3% |
| T011, 60,000 steps | 12.8% | 9.9% |
| T011, 100,000 steps | 11.2% | 8.4% |
| + chunk-aware fine-tune 20k, seed 2026 / 2027 | 10.5% / 9.9% | 6.9% / 5.1% |
| + control fine-tune 20k, seed 2026 / 2027 | 9.8% / 9.4% | 6.3% / 7.3% |

Chunk study of T011 at 100,000 steps (100 held-out sentences): whole sentence 10.9%; independent
1-, 2-, 4-word chunks 52.7%, 25.8%, 17.8%; context-aware 2-word chunks without lookahead 21.3%;
context-aware 1-word chunks with lookahead 2 12.5% (vs whole +1.6 [−1.0, +4.1]); 2-word chunks with
lookahead 2 12.2% (+1.3 [−1.1, +3.5]). At about 3.9 words of waiting, context-aware 2-word chunks
with lookahead 2 beat independent 4-word chunks by 5.6 points [−8.7, −2.7].

Chunk-aware fine-tune against its control (chunk − control, 95% paired-bootstrap interval):

| Condition | Seed 2026 | Seed 2027 |
|---|---|---|
| independent 1-word chunks | 44.4% → 25.8%, −18.7 [−22.8, −14.5] | 46.5% → 22.1%, −24.4 [−28.8, −19.8] |
| independent 2-word chunks | 23.0% → 12.5%, −10.5 [−13.6, −7.5] | 24.3% → 13.7%, −10.6 [−14.5, −6.6] |
| independent 4-word chunks | 14.5% → 12.4%, −2.1 [−4.8, +0.5] | 14.1% → 10.0%, −4.1 [−6.8, −1.4] |
| context-aware 2-word chunks, no lookahead | 18.0% → 16.0%, −2.0 [−5.2, +1.1] | 18.4% → 12.2%, −6.2 [−9.2, −3.4] |
| context-aware 1-word chunks, lookahead 2 | 11.3% → 10.9%, [−2.7, +1.9] | 11.4% → 10.7%, [−2.9, +1.7] |
| whole sentence | 9.8% → 10.3%, [−1.7, +2.7] | 9.4% → 9.3%, [−2.0, +1.9] |

Independent 2-word chunks last 0.90× / 0.88× the recording after chunk-aware fine-tuning, 1.14× /
1.13× for the controls. The second seed uses `--seed 2027` (data order and crops) from the same
100k checkpoint.

Synthesis time, chunk-aware fine-tune (seed 2026), 30 held-out sentences, idle RTX 4070 Ti SUPER /
8 CPU threads: whole sentence 30 ms / 237 ms; first 1-word chunk with lookahead 2 14 ms / 38 ms;
first 2-word chunk 14 ms / 51 ms.

### ASR protocol v2

Protocol v1 decodes at temperature 0 without fallback, so a Whisper repetition loop (one
transcription repeating a phrase dozens of times) is never retried. v2 (`--protocol v2`, output
`asr_v2.json`; `tools/evaluate_runs.py` and `tools/chunk_analysis.py` take `--asr-protocol v2`)
adds whisper's fallback for loops only: when the compression ratio exceeds 2.4, decode again at
temperature 0.2…1.0 (best of 5, seeded). Other transcriptions equal the v1 ones, which v2 reuses.
22 of 33,696 transcriptions of the GPU Lab evaluations were loops. Changed scores: T011 100k long
21.7% → 8.4%, T011 60k long 15.8% → 9.9%, V2 29.1% → 19.9%, TTS_0.10 L1 186.9% → 98.7%; every
other TTS_0.10 arm and the R0 chunk study unchanged (R0 1-word chunks 56.1% → 55.7%).

The raw evaluation files of the GPU Lab runs (v2 scores, the seed-2027 runs) were on the instance's
temporary disk and were lost when it was released on 26/10; the numbers above were recorded before.

## Known issue in the first T011 run

The base run started on 2026-10-24 (`tts010_T011_20261024T152445372698Z`) used code from before
a fix: its in-training held-out probe decoded the first eight held-out clips in corpus order,
which include one long clip, while `run_config.json` lists the TTS_0.10 probe IDs. Only the
in-training held-out probe curve is affected; the held-out evaluation after training
(`tools/eval_tts010.py`) is not. Later runs use the listed clips.

## Where it runs

UQ GPU Lab, 1× NVIDIA L40S, PyTorch 2.11.0+cu130 (the version of the A1–L1 arms).
