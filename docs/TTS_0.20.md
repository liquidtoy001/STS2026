# TTS_0.20 candidates: options for a voice-agent TTS

Status 2026-10-25 (Australia/Brisbane): options implemented and tested; small-scale runs on
LJSpeech not started. Agreed with the team lead: screen the options on LJSpeech first (arms V1–V5,
compared with T011), then run the baseline and the best options on LibriTTS-R.

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

## Measured cost (smoke runs, RTX 4070 Ti SUPER, no data-loader workers)

| | T011 | V1 | V2 | V3 | V4 |
|---|---|---|---|---|---|
| s/step | 0.22–0.25 | 0.27 | 0.29 | 0.33 | 0.54 |

## Sources

- VITS: Kim, Kong and Son (2021), stochastic duration predictor; spline flow after Durkan et al. (2019).
- Streaming with limited lookahead and bounded context: Liu et al. (2026), arXiv:2603.06444.
- g2p_en: Park and Kim, https://github.com/Kyubyong/g2p.
