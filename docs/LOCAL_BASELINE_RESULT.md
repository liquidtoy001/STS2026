# Original-model local baseline

- Training completed at (UTC): **approximately `2026-09-19T15:27:12+00:00`**.
- Brisbane local equivalent: approximately `2026-09-20T01:27:12+10:00`.
- Completion time source: the original training `status.json` last-write time.
  The old runner did not record a runtime completion timestamp; this is a
  reconstructed artifact timestamp, not a newly measured exact finish time.

The original notebook configuration was evaluated on eight fixed LJSpeech
training utterances with one training seed (2026). This is a training-batch
reconstruction baseline, not held-out or complete text-to-speech evaluation.

| Item | Measurement |
|---|---:|
| Parameters | 12,141,186 |
| GPU | RTX 4070 Ti SUPER |
| Training iterations | 2,000 |
| Actual optimizer updates | 1,988 |
| AMP-skipped updates | 12 |
| Training, evaluation and checkpoint-saving time | 640.92 seconds |
| Peak PyTorch allocated memory | 0.789 GiB |
| Peak PyTorch reserved memory | 0.830 GiB |

Memory figures cover PyTorch allocations, not total system GPU use. Model
initialization and dataset loading are outside the reported timing interval.

| Reconstruction mode | Metric | Iteration 0 | Iteration 2,000 |
|---|---|---:|---:|
| Mean | Log-mel L1 | 3.340485 | 0.560088 |
| Mean | STFT spectral convergence | 2.202136 | 0.862740 |
| Sample | Log-mel L1 | 1.961753 | 0.557840 |
| Sample | STFT spectral convergence | 2.196269 | 0.861846 |

Lower is better; these are not accuracy percentages. The measured reconstruction
error decreased, but this does not establish convergence, acceptable perceptual
quality or generalization to unseen utterances.

## Checkpoint rescoring

Completed at (UTC): **approximately `2026-09-19T15:27:56+00:00`**, reconstructed
from the original rescoring `status.json` last-write time.

The final checkpoint was strictly reloaded in a fresh Python process after
checking the data fingerprint and evaluation protocol. Across eight samples and
two modes, the maximum per-sample absolute difference was zero for all three
metrics. This verifies same-machine, same-environment checkpoint rescoring only;
cross-machine differences and variation between training runs were not measured.

Final checkpoint SHA256:
`4cfbd3c87bcb84d456038a30f99cf50fb64c7dc35ab0be023878c08b853022d4`.

## Evidence and iteration accounting

- Original local run: `output/local-eval/original_20260919T151630615757Z/`.
- Rescoring run: `output/local-eval/original_20260919T152754962878Z/`.
- English reports: `REPORT.md` inside each run directory.
- The measured source snapshot is under the original run's `source_snapshot/`;
  its hashes match `run_config.json`.
- The raw metrics and checkpoint were preserved when translating the report and
  adding explicitly approximate legacy completion metadata.

After observing AMP skips, the shared trainer gained an actual-update counter
and the plot axis was labeled as training iterations. These changes do not turn
the historical result into 2,000 effective updates. The recorded 1,988 updates
were counted independently by an optimizer hook in the original local runner.

New-model runs should report iteration counts, actual updates and AMP skips.
Additional seeds are needed to estimate training variation. No confidence
intervals or error percentages are provided for unperformed experiments.
See [local execution instructions](LOCAL_EVALUATION.md).
