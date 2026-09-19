# Local GPU execution and checkpoint rescoring

The local entry loads the original notebook model, loss and trainer definitions
while skipping demonstration cells. The network is not reduced. Install the
compatible PyTorch/TorchAudio pair and dependencies in `requirements/notebook.txt`.

## Prepare the fixed data and run

From the repository root:

```sh
python tools/prepare_ljspeech_subset.py
python tools/run_local_eval.py --data-dir data/LJSpeech-1.1 --steps 20 --eval-every 20
python tools/run_local_eval.py --data-dir data/LJSpeech-1.1 --steps 2000 --eval-every 500
```

The preparation script downloads complete metadata and only the eight fixed
audio files from a revision-pinned public mirror. It checks the mirror's SHA256
values and verifies 22,050 Hz mono 16-bit PCM audio. Provenance is recorded in
`data/LJSpeech-1.1/download_manifest.json`. This is a mirror integrity check,
not independent verification against the upstream archive.
Sources: [LJSpeech upstream](https://keithito.com/LJ-Speech-Dataset/) and
[file mirror](https://huggingface.co/datasets/flexthink/ljspeech).

Each training command creates a new `output/local-eval/original_<UTC time>/`
directory and initializes a fresh model. Do not mix resumed and independently
initialized runs when comparing a fixed training budget.

## Inspect execution records

Check `status.json`, `local_measurement.json` and `run_config.json` before scores:

- Completed iterations and actual optimizer updates can differ because AMP may
  skip a step. The local runner independently counts optimizer calls.
- `peak_allocated_bytes` records PyTorch memory, not total desktop GPU use.
- Timing includes training, evaluation and checkpoint saving, excluding initial
  data loading and model construction.
- Successful runs record `completed_at_utc`; failed/interrupted training runs
  record `ended_at_utc` instead. Per-evaluation completion is recorded in
  `evaluation/step_*/completion.json`.
- All experiment logs and reports are English. Completion uses ISO 8601 UTC;
  report-generation time is a separate field. Legacy inferred times must state
  their evidence and remain labeled approximate.

Generate an English report from a completed training or rescoring run:

```sh
python tools/report_local_eval.py output/local-eval/original_ACTUAL_RUN_TIME
```

This writes `REPORT.md`. Unknown historical update counts remain unknown.
Report regeneration does not change experiment-completion time or metric values.

## Rescore an existing checkpoint

Share the original run folder through the team's agreed artifact channel.
Weights and generated audio are not committed. Preserve the checkpoint and
evaluation directory layout, prepare the same data, then run:

```sh
python tools/run_local_eval.py --data-dir data/LJSpeech-1.1 --checkpoint output/local-eval/original_ACTUAL_RUN_TIME/checkpoints/vits_step_2000.pt
```

This does not train. It checks the reference data fingerprint, sampling settings,
sample order and crop plan, strictly loads weights, then records summary and
per-sample metric differences in the new run's `status.json`.
Same-machine agreement does not establish cross-machine or retraining agreement.

See the [shared protocol](EVALUATION.md) and [measured baseline](LOCAL_BASELINE_RESULT.md).
