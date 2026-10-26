# TTS_0.20 candidates: options for a voice-agent TTS

Status 2026-10-27 (Australia/Brisbane): phase 1 (V1–V5 on LJSpeech) and the first phase-2 run
(LibriTTS-R pretraining, then LJSpeech) are trained and evaluated. No option beat TTS_0.11, so none
was adopted; the larger LibriTTS-R manifest was prepared but not trained on.

## Options

Every option is off by default (`TTS020_OPTIONS` in `notebooks/vits-full-train.ipynb`). An arm
lists the options it changes; only those are added to its switches, so every TTS_0.10 arm and
T011 keep exactly their recorded switches and checkpoints.

| Option | Values | What it does |
|---|---|---|
| `text_input` | `chars` / `phonemes` | Phoneme input from g2p_en (CMUdict, homographs by part-of-speech tag, neural model for unknown words). Each whitespace word stays one word, so chunking works unchanged; each ARPAbet phoneme is one character `chr(0x100 + i)` (`tools/phonemize.py`). LJSpeech phonemes: `sts_eval/phonemes/ljspeech_g2p_en.tsv` (13,100 clips, checked against `metadata.csv` by hash). |
| `encoder_window` | `None` / `[left, K]` | Text-encoder attention limited to `left` words back and 0..K words ahead (K drawn per training batch). Only the first layer looks ahead; later layers look back only and the feed-forward convolutions are causal, so the lookahead never widens with depth. A token's encoder output is final once the text K words beyond its word has arrived: chunked synthesis with lookahead K gives exactly the whole-text output (test `test_windowed_encoder_is_exact_for_streaming`). |
| `duration_predictor` | `deterministic` / `stochastic` | The flow-based stochastic duration predictor of VITS (Kim et al., 2021), trained on the MAS durations by its variational bound; at inference it samples log durations (`noise_scale_w` 0.8). Runs in float32. |

Chunk-aware training gains `--chunk-boundaries punctuation` (`tools/train_tts010.py`): crops end
where a word ends in `, . ? ! ; :` or at the end of the text, instead of after any word.

## Arms

All: T011's switches and held-out sets (200 clips), seed 2026, 40,000 steps.

| Arm | Options |
|---|---|
| V1 | windowed encoder `[12, 4]` |
| V2 | phoneme input |
| V3 | phoneme input + windowed encoder `[12, 4]` |
| V4 | stochastic duration predictor |
| V5 | V3 fine-tuned 20,000 steps with chunk-aware training ending at punctuation (`--init-from` V3, `--chunk-boundaries punctuation`), with a control fine-tune |

```sh
python tools/train_tts010.py --arm V2 --data-dir /path/LJSpeech-1.1 --run-root runs
```

Phoneme models need `g2p_en` and its NLTK data (`averaged_perceptron_tagger_eng`, `cmudict`) at
inference and evaluation; training reads only the phoneme file. `tools/eval_tts010.py`,
`tools/chunk_study.py` and `tools/chunk_alignments.py` read the options from the checkpoint.

## Phase 1 results (25/10)

Whisper medium WER, ASR protocol v2; one seed per arm; 40,000 steps (V5: V3 + 20,000). TTS_0.10's
seed noise floor is 3.3 points.

| Arm | 100 test sentences | 100 long sentences | Chunk study: whole / 1-word chunks + lookahead 2 / independent 2-word chunks |
|---|---|---|---|
| T011 (reference, 40k) | 12.6% | 10.3% | – |
| V1 | 17.8% | 13.3% | 18.9% / 18.7% / 41.8% |
| V2 | 19.9% | 11.9% | 16.7% / 17.3% / 37.6% |
| V3 | 18.0% | 14.1% | 24.9% / 22.5% / 49.1% |
| V4 | 17.8% | 13.4% | 17.3% / 19.9% / 34.1% |
| V5 | 17.8% | 13.0% | 15.3% / 14.9% / 37.5% |
| V5 control (V3 + 20k on whole clips) | 14.7% | 11.4% | 15.9% / 17.8% / 42.8% |

All four options score at least 5 points worse than T011 at the same steps on the test sentences;
on long sentences the gap is 1.6–3.8 points.

## Phase 2: LibriTTS-R pretraining, then LJSpeech fine-tuning

Decided with the team lead 2026-10-25: the agent needs one voice, so LibriTTS-R serves only as
pretraining data. A model is pretrained on LibriTTS-R with speaker IDs (no single reader has
more than about 40 minutes), then fine-tuned on LJSpeech as a single-speaker model and evaluated
on the LJSpeech test sets like every other arm.

**Corpus** (`tools/prepare_libritts_r.py`, records in
[records/tts-0.20/libritts-r](records/tts-0.20/libritts-r/)): LibriTTS-R resampled from 24 kHz
to 22,050 Hz 16-bit mono, written as a speaker manifest (`docs/SPEAKER_CONDITIONING.md`).
Dropped: utterances on the corpus authors' list of failed speech restorations, texts with
characters outside the notebook's character table, and clips outside 1–15 s.

| Manifest | Subsets | Kept | Hours | Readers |
|---|---|---|---|---|
| `train-clean-100` | train-clean-100 | 29,270 of 33,232 | 43.7 | 246 |
| `train-clean-460` | train-clean-100 + train-clean-360 | 131,244 of 149,694 | 198.1 | 1,144 |

Test set `sts_eval/splits/libritts_r_heldout_100.txt`: one utterance (2–10 s, at least five
words) from each of 100 train-clean-100 readers, seed 2026; kept out of pretraining in both
manifests. Phoneme caches: `tools/phonemize.py --manifest`.

**Pretraining**: `--speaker-manifest` and `--speaker-heldout` (`STS_SPEAKER_MANIFEST`,
`STS_SPEAKER_HELDOUT`); 64-dimensional speaker vectors.

**Fine-tuning**: `--init-from <pretrain checkpoint> --fold-speaker mean` starts a new
single-speaker LJSpeech run (`VITSTrainer.init_from_speaker_checkpoint`). The mean speaker vector
is folded into the biases its projections are added to, so the result is an ordinary
single-speaker model that every evaluation tool loads unchanged. The fold reproduces the
conditioned posterior encoder, flow, decoder and stochastic duration predictor exactly; the
deterministic duration predictor differs only for the two tokens at either end of a text
(`tests/test_speaker_fold.py`). Optimizers start fresh; the discriminator is carried.

```sh
python tools/train_tts010.py --arm T011 --data-dir corpus --speaker-manifest corpus/speakers_train-clean-100.csv \
    --speaker-heldout sts_eval/splits/libritts_r_heldout_100.txt --tag pretrain-lttsr100
python tools/train_tts010.py --arm T011 --data-dir /path/LJSpeech-1.1 --tag ft-lttsr100 \
    --init-from runs/tts010_T011_pretrain-lttsr100_<run>/checkpoints/vits_step_40000.pt \
    --fold-speaker mean --steps 20000
```

Result (one seed): pretraining on `train-clean-100` for 40,000 steps, the mean speaker folded in,
then 20,000 steps on LJSpeech scores 15.2% on the 100 test sentences and 7.8% on the long ones,
against 12.8% and 9.9% for T011 at the same 60,000 steps on LJSpeech alone. Both differences are
within the seed noise floor and point in opposite directions: no clear gain. Its chunk study:
whole sentence 11.9%, independent 2-word chunks 31.6%, 1-word chunks with lookahead 2 13.0%.

Smoke runs on the L40S (60 steps each, T011 and V3, sharing the GPU with two training runs):
pretraining 0.47–0.49 s/step, peak 4.1 GiB; folded fine-tune 0.40–0.51 s/step.

## Measured cost (smoke runs, RTX 4070 Ti SUPER, no data-loader workers)

| | T011 | V1 | V2 | V3 | V4 |
|---|---|---|---|---|---|
| s/step | 0.22–0.25 | 0.27 | 0.29 | 0.33 | 0.54 |

## Sources

- VITS: Kim, Kong and Son (2021), stochastic duration predictor; spline flow after Durkan et al. (2019).
- Streaming with limited lookahead and bounded context: Liu et al. (2026), arXiv:2603.06444.
- g2p_en: Park and Kim, https://github.com/Kyubyong/g2p.
