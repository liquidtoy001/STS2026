# TTS_0.10 central evaluation

Generated 2026-10-20T18:58:57+00:00 from E:\Documents\STS2026\output\tts010-turino\eval. Held-out set: sts_eval/splits/ljspeech_heldout_100.txt; reference checkpoints are not TTS_0.10 runs.

| Label | Arm | What it is | Held-out reconstruction log-mel L1 | Held-out reconstruction STFT SC | Whisper WER | Whisper CER | ΔWER vs natural | Empty ASR outputs | Duration ratio (1 = natural) | Level ratio (1 = natural) |
|---|---|---|---|---|---|---|---|---|---|---|
| R0_40k | R0 | complete TTS_0.10 | 0.533 | 0.542 | 16.9% | 8.2% | +13.6 pt | 0 | 0.957 | 0.898 |
| G1_40k | G1 | TTS_0.02 text encoder | 0.517 | 0.555 | 23.7% | 12.6% | +20.5 pt | 0 | 0.912 | 0.816 |
| G2_40k | G2 | TTS_0.02 posterior encoder | 0.572 | 0.673 | 24.7% | 14.8% | +21.5 pt | 0 | 0.965 | 0.958 |
| G3_40k | G3 | TTS_0.02 decoder | 0.511 | 0.520 | 13.8% | 7.6% | +10.5 pt | 0 | 0.977 | 0.828 |
| A1_40k | A1 | TTS_0.02 discriminator | 0.558 | 0.567 | 19.2% | 9.9% | +15.9 pt | 0 | 0.961 | 0.920 |
| A2_40k | A2 | no feature matching | 0.509 | 0.586 | 15.7% | 8.2% | +12.5 pt | 0 | 0.978 | 0.999 |
| L1_40k | L1 | TTS_0.02 reconstruction loss | 1.147 | 0.929 | 186.9% | 130.5% | +183.6 pt | 48 | 0.775 | 0.545 |
| O1_40k | O1 | TTS_0.02 optimisation | 0.540 | 0.576 | 28.5% | 16.0% | +25.3 pt | 0 | 0.933 | 0.782 |
| D1_40k | D1 | no text-length filter | 0.519 | 0.538 | 11.6% | 5.8% | +8.3 pt | 0 | 1.002 | 0.918 |
| E0_40k | E0 | every switch at TTS_0.02 | 1.518 | 0.962 | 100.2% | 87.3% | +96.9 pt | 17 | 0.725 | 0.242 |
| benlu_original_40k | – | reference checkpoint (not a TTS_0.10 run) | 0.529 | 0.572 | 15.7% | 8.3% | +12.4 pt | 0 | 0.976 | 0.877 |
| tts001_gan0_50k | – | reference checkpoint (not a TTS_0.10 run) | 1.278 | 0.941 | 99.7% | 88.3% | +96.4 pt | 21 | 0.732 | 0.216 |
| tts001_gan1_50k | – | reference checkpoint (not a TTS_0.10 run) | 1.340 | 0.944 | 100.3% | 90.8% | +97.1 pt | 35 | 0.737 | 0.279 |

## Difference from R0

Positive = higher than R0. The noise floor needs R0b, which is not evaluated yet.

| Label | Arm | Held-out reconstruction log-mel L1 | Held-out reconstruction STFT SC | Whisper WER | Whisper CER | ΔWER vs natural | Empty ASR outputs | Duration ratio (1 = natural) | Level ratio (1 = natural) |
|---|---|---|---|---|---|---|---|---|---|
| G1_40k | G1 | -0.016 | +0.013 | +6.8 pt | +4.4 pt | +6.8 pt | +0 | -0.046 | -0.082 |
| G2_40k | G2 | +0.040 | +0.132 | +7.8 pt | +6.7 pt | +7.8 pt | +0 | +0.007 | +0.060 |
| G3_40k | G3 | -0.022 | -0.022 | -3.1 pt | -0.5 pt | -3.1 pt | +0 | +0.019 | -0.071 |
| A1_40k | A1 | +0.025 | +0.026 | +2.3 pt | +1.7 pt | +2.3 pt | +0 | +0.003 | +0.021 |
| A2_40k | A2 | -0.023 | +0.044 | -1.2 pt | +0.0 pt | -1.2 pt | +0 | +0.020 | +0.101 |
| L1_40k | L1 | +0.614 | +0.388 | +169.9 pt | +122.3 pt | +169.9 pt | +48 | -0.183 | -0.353 |
| O1_40k | O1 | +0.007 | +0.034 | +11.6 pt | +7.8 pt | +11.6 pt | +0 | -0.025 | -0.117 |
| D1_40k | D1 | -0.013 | -0.004 | -5.3 pt | -2.4 pt | -5.3 pt | +0 | +0.044 | +0.019 |
| E0_40k | E0 | +0.985 | +0.420 | +83.2 pt | +79.2 pt | +83.2 pt | +17 | -0.233 | -0.656 |
| benlu_original_40k | – | -0.003 | +0.030 | -1.3 pt | +0.1 pt | -1.3 pt | +0 | +0.019 | -0.021 |
| tts001_gan0_50k | – | +0.745 | +0.399 | +82.7 pt | +80.1 pt | +82.7 pt | +21 | -0.225 | -0.682 |
| tts001_gan1_50k | – | +0.807 | +0.403 | +83.4 pt | +82.7 pt | +83.4 pt | +35 | -0.221 | -0.619 |
