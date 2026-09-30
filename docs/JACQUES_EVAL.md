# Jacques: original versus new encoder/decoder reconstruction study

This study belongs in the project report's **Experiments and results** section,
after the original-model baseline and before the full-training GAN comparison.
Its component identity is pending verification; interpret the supplied metrics
as two recorded runs until the executed component source is provided.

## Purpose and ownership

Jacques was asked to evaluate `original_eval` and a revised encoder/decoder pair.
The supplied folders are named `vits-eval-original` and `vits-eval-new-enc-dec`.
These names record the intended comparison. They do not independently establish
which component implementation ran. The numerical analysis below is derived
from Jacques's supplied records; the ZIP contains plots and screenshots but no
separate written analysis document or model source.

## Experimental protocol

Both runs use a Kaggle Tesla T4, PyTorch/TorchAudio 2.10.0+cu128, training seed
2026 and evaluation seed 1234. They train on the same fixed eight LJSpeech
utterances, LJ001-0001 through LJ001-0008, for 2,000 attempted iterations. Each
completed 1,990 optimizer updates with ten AMP skips. Evaluation occurs at
iterations 0, 500, 1,000, 1,500 and 2,000. Posterior mean and seeded posterior
sample are scored separately on identical central 32-frame crops: 8,192 samples
at 22,050 Hz, approximately 0.37 seconds.

The loss is `10 * mel + STFT + 10 * waveform`; duration and KL weights are zero
and logged GAN loss is zero. This evaluates posterior reconstruction of training
audio. It does not measure held-out generation, full-sentence TTS, or GAN effects.

Training executed by: **Jacques**. The team-designated completion date for this
study is **2026-09-20**. This is a display attribution supplied by the project
owner, not an alteration of the exported logs. The newer folder's runtime
timestamp conflicts with that date and needs reconciliation with Jacques.

The original run completed at `2026-09-20T07:42:35+00:00`; the folder labelled
new enc/dec completed at `2026-09-30T06:15:25+00:00`. These are supplied runtime
completion timestamps, distinct from report-import or editing times.

## Measured results

Lower is better. Values below are rounded for presentation; the archived JSON
and website data retain the supplied precision.

| Mode | Metric | Original | New enc/dec folder | Relative change |
|---|---|---:|---:|---:|
| Mean | Log-mel L1 | 0.570351 | 0.583125 | +2.24% |
| Mean | STFT spectral convergence | 0.829299 | 0.832924 | +0.44% |
| Mean | STFT log-magnitude | 1.135114 | 1.152971 | +1.57% |
| Sample | Log-mel L1 | 0.558221 | 0.583025 | +4.44% |
| Sample | STFT spectral convergence | 0.825226 | 0.828664 | +0.42% |
| Sample | STFT log-magnitude | 1.035579 | 1.064428 | +2.79% |

Both runs reduce reconstruction errors over training. At iteration 1,500, the
new enc/dec folder has lower errors for all six mode/metric combinations; at
iteration 2,000, all six are higher than the original run. The available evidence
therefore shows no consistent final-checkpoint improvement. It supplies no
independent training repeats or estimate of uncertainty. Checkpoints along one
trajectory are not independent experimental replicates.

## Component identity and limitations

The two `run_config.json` files are byte-identical, including `variant = original`,
the recorded model structure, optimizer settings and notebook SHA256
`ab8385ddd7711f0852b6a7870cd38f3d36841fafe2ccf099a871c16727778c29`.
Their evaluation manifests are also identical. This supports matching recorded
evaluation conditions, but cannot verify a changed encoder/decoder. It may
reflect stale provenance after editing; it does not prove that the runs were
duplicates or that the revised components were absent.

Before attributing the difference to architecture, obtain the actually executed
notebooks or source snapshots, the new component definitions, a checkpoint with
its configuration, and an explanation of the unchanged variant/hash. Keep the
supplied metadata intact and add any clarification as a separate dated record.
If encoder and decoder both changed, this comparison can only assess the pair;
separate ablations are needed to distinguish their individual effects.

No perceptual verdict is recorded here. The supplied WAVs are short reconstruction
crops. Team listening notes and full-utterance/held-out evaluation remain separate
work. These runs must not be combined with the differently configured 50,000-step
GAN comparison or substituted for the existing local baseline.

## Report-ready paragraph

Jacques evaluated two supplied model variants using the shared posterior
reconstruction protocol on eight fixed LJSpeech training utterances. Both runs
completed 2,000 iterations with 1,990 optimizer updates. At the final checkpoint,
the folder labelled new encoder/decoder showed slightly higher reconstruction
errors across all six mean/sample metrics; sample-mode log-mel L1 increased from
0.5582 to 0.5830. Although its intermediate checkpoint at iteration 1,500 had
lower errors, the study does not establish a consistent improvement. Architecture
attribution remains provisional because both runs carry identical model and
source provenance. The results concern training-batch reconstruction rather
than held-out synthesis or subjective speech quality.

## Storage and reproducibility

- [HTML report](site/report.html#jacques-enc-dec): paired metrics and training
  curves, interpretation and links to cloud artifacts; English notes.
- `docs/records/jacques-enc-dec/`: exact source JSON bytes, normalized paths,
  full per-sample/checkpoint records, ZIP checksum and file inventory.
- [Jacques Eval on Drive](https://drive.google.com/drive/folders/1kNvjpzigLiBSxK3i4NTdo-GwAKNVmYeV):
  each run contains `records/`, `evaluation/`, `figures/`, `screenshots/`, `_archive/`.
  Audio remains with its evaluation step. The three duplicate audio downloads
  per run are preserved in `_archive/`; no source files are deleted.
- Drive was subsequently reorganized by the project owner. Folder links retain
  their IDs, but the internal audio paths should be rechecked before listening.

Rebuild the data and imported records with:

```sh
python tools/import_jacques_eval.py "Jacques Eval-20260930T110856Z-1-001.zip"
```

The importer executes no notebook code. It validates completion, update counts,
paired protocol and per-sample arithmetic before writing report data. Raw audio,
screenshots, checkpoints and the original ZIP are not imported into Git records.
