# Posterior reconstruction baseline and handoff

This experiment evaluates `spectrogram -> posterior encoder -> decoder -> waveform`
on the **same eight utterances used for overfit training**. It is not held-out
evaluation, text-to-speech evaluation, or evidence of generalization. This page
records the September 2026 workflow; the TTS models from TTS_0.10 on are evaluated
with Whisper on held-out sentences ([central evaluation](ABLATION_TTS_0.10.md#central-evaluation)).

## Run on Kaggle

1. Build the uploadable notebook locally: `python tools/build_kaggle_eval.py`.
   Upload `output/kaggle/vits-light_v1_eval.ipynb` to Kaggle. It bundles exact
   shared evaluator sources; no separate Python package upload is required.
2. Attach LJSpeech-1.1, enable a GPU, and check `DATA_DIR` in the configuration
   cell against the actual mounted dataset. It must contain `metadata.csv` and
   `wavs/`. Use the runtime's compatible PyTorch/TorchAudio pair.
3. Execute definitions in order. Before the final training cell, inspect
   `FIXED_IDS`, `TRAIN_SEED`, `TOTAL_STEPS`, model, loss weights and optimizer.
   The shared setup selects LJ001-0001 through LJ001-0008 in that order.
4. Execute the training cell: 2,000 training iterations, training losses every 50 iterations,
   evaluation at steps 0/500/1000/1500/2000. This is a budget, not a convergence
   guarantee. Both architectures must use the same budget for this comparison.
   AMP can skip an optimizer update. Compare both iteration counts and actual
   update counts in the status/history/checkpoint; never silently equate them.
5. Run the display cell. Download the run directory beneath
   `/kaggle/working/sts2026-eval-runs/`. Inspect `status.json` and final step before
   interpreting results. Preserve failed/interrupted runs; do not count them as
   completed comparisons.

For a local checkout, run the original notebook from `notebooks/` or the root.
Change `DATA_DIR`. Outputs default to `output/eval/`. `STS_REPO_ROOT` and
`STS_RUN_ROOT` can override repository and artifact locations. Restart the kernel
for each comparison run and use a fresh model/optimizer; checkpoint resume is
deliberately rejected for these comparison runs.

For the local GPU command-line runner and downloading only the eight required
audio files, see [local instructions](LOCAL_EVALUATION.md). The runner can
also reload a checkpoint and compare its metrics with the saved reference,
without training again.

## Protocol

- Keep sample IDs/order, batch size, data preprocessing, sample rate, crop length,
  seeds, optimizer, loss weights, precision and training budget fixed.
- Encode the full spectrogram; decode a deterministic central 32-frame crop
  (8,192 samples at hop 256). Short utterances are padded for decoding but only
  their real samples enter metrics. Crops shorter than 1,025 valid samples fail
  clearly because the configured reflect-padded STFT cannot score them.
- Score posterior mean and seeded posterior sample **separately**. The shared
  sampler uses `mean + noise * exp(log_scale)`; the encoder's returned random
  latent is intentionally unused. Noise has its own generator. Evaluation
  restores CPU/CUDA RNG state and each module's original training flag.
- Fixed seeds improve repeatability, but do not promise bitwise equivalence
  across hardware, PyTorch versions, or different architectures.
- Metrics: log-mel L1 (80 bands, magnitude mel, HTK, no normalization, natural log,
  floor 1e-5), MR-STFT spectral convergence and log-magnitude L1 (floor 1e-7).
  STFT resolutions `(FFT, hop, window)` are `(512,128,512)`, `(1024,256,1024)`,
  `(2048,512,2048)`, matching the original loss. Hann windows, centered STFT and
  reflect padding are explicit. Lower is better. Each metric is computed per
  utterance and summarized by arithmetic mean, with no padding in the reduction.
- These reconstruction metrics overlap with training objectives. Listen to the
  exported matched samples before drawing conclusions about audible changes.
  The short crops are unsuitable for a meaningful full-utterance WER experiment.
- If both components change, the comparison identifies the effect of the pair;
  it cannot isolate the encoder's contribution from the decoder's contribution.

## Files to hand off

- `run_config.json`: model, optimizer, loss weights, seeds, data/code hashes,
  local revision/dirty state when available, precision and device.
- `evaluation/manifest.json`: actual batch tensor hash, sample IDs, crop starts,
  valid sample counts, metric configuration and environment.
- `training_history.json`: training losses from random crops; labels refer to
  the attempted training iteration, using its forward pass before the update.
- `evaluation/step_*/per_sample.json` and `summary.json`: fresh evaluation after
  the indicated iteration. Mode `mean` and mode `sample` are separate observations.
- Final step float WAVs: numbered target/mean/sample files; numbers map to the
  manifest sample IDs. Values are scored without clipping or normalization.
- `curves.png`, `status.json`, and final checkpoint. Checkpoints/audio stay out of
  Git; small reviewed reports can be added separately.

## New-model integration contract

The evaluator expects a PyTorch module with `posterior_encoder`, `decoder`,
`hop_length`, `segment_frames`, and `segment_samples`. The encoder returns
`(z, mean, log_scale, mask)` from `(spectrogram, spectrogram_lengths)`;
`log_scale` must be log standard deviation, not log variance. The decoder accepts
`[batch, latent_channels, 32]` and returns `[batch, 1, 8192]` for this protocol.
Add a thin adapter if necessary, while preserving these meanings.

In a separate PR, build the new components **before creating the trainer**, set
`MODEL_VARIANT = "new"`, and keep the original implementation selectable. Rebuild
the Kaggle export. Verify matching batch hashes, crop plans and protocol before
comparing results. Run both versions in the same validated environment. The
second evaluator owns integration, new-model runs and paired sample analysis;
the baseline owner supplies this common protocol and the actual old-model run.

## Validation and limits

Run `python -m unittest discover -s tests -v` in an environment with the notebook
dependencies plus matplotlib. Tests execute actual notebook model/trainer
definitions with synthetic tones: identity/degradation metrics, crop padding,
mode and RNG restoration (including failure), CUDA RNG, a real optimizer update,
checkpoint/audio export and plotting. Synthetic checks are not speech results.
A local real-data baseline has completed: 2,000 iterations, 1,988 actual updates,
and exact same-machine checkpoint rescoring. See the [measured baseline](LOCAL_BASELINE_RESULT.md).
Kaggle execution, held-out quality and the new-model comparison remain untested.
