# Evaluation implementation validation — 2026-09-19

Historical validation completed at (UTC): approximately
`2026-09-19T15:30:31+00:00`, reconstructed from the original
`tmp/eval-tests-final.log` last-write time. Exact runtime completion was not
recorded by that validation command.

Branch: `Turino`. This record covers implementation checks, not a speech benchmark.
The workspace also contains earlier, uncommitted repository layout changes.

**2026-09-20 update:** A subsequent full-configuration local run on eight real
LJSpeech utterances completed 2,000 iterations (1,988 actual updates; 12 AMP skips).
See [the measured local baseline](LOCAL_BASELINE_RESULT.md). The original
"Not executed" list below describes the earlier 2026-09-19 validation only.

## Executed checks

- Seven reconstruction behavior tests passed using the actual notebook class
  definitions, including an actual optimizer update on synthetic tone inputs,
  checkpoint save, float-WAV reload, plot generation, and CPU/CUDA RNG checks.
- The export bootstrap test passed: generated cells parse and the bootstrap
  reproduces the bundled UTF-8 source snapshots with LF newlines.
- Both the source notebook and the Kaggle export passed `nbformat.validate`
  and code-cell Python parsing. Neither contains execution outputs.
- Cell comparison against the Git baseline confirmed that only cells 41–46
  (zero-based) changed: trainer, evaluation setup, training instructions,
  model/run construction, execution and final inspection. Model definitions
  and earlier data/preprocessing cells are unchanged.

The reconstruction tests used Python from the existing local RVC environment,
PyTorch `2.7.1+cu118`, TorchAudio `2.7.1+cu118`, and an NVIDIA GeForce RTX 4070 Ti
SUPER for the CUDA check. The optimizer integration check ran on CPU with AMP
disabled. Notebook schema validation used the existing STAT3007 environment.
No packages were installed or modified in either environment.

## Not executed

- No LJSpeech training: the dataset is not available locally.
- No 2,000-step convergence result, audible speech-quality assessment, held-out
  evaluation, new-model comparison, or Kaggle GPU execution.
- No claim of full-size model training validation from the integration test:
  it uses the actual architecture with smaller encoder widths and the original
  decoder width. CUDA evaluation was checked, not CUDA training with AMP.

Use `python -m unittest discover -s tests -v` to repeat the behavioral checks.
Use `python tools/build_kaggle_eval.py` to regenerate the uploadable notebook,
then follow [the evaluation protocol](EVALUATION.md) for the real old-model run.

## English-record and completion-time validation

Completed at (UTC): `2026-09-19T16:30:31+00:00`. Source: runtime clock.

All 11 tests passed, including completion-time attribution for historical records, preserving runtime completion through report regeneration, rejecting failed runs as completed reports, and the existing reconstruction/export checks. Personal learning walkthroughs are local-only; shared operational documentation is English.

A real-data CUDA timestamp probe completed at (UTC): `2026-09-19T16:30:37+00:00` (runtime clock). It ran one attempted iteration, with zero actual updates because AMP skipped it. Both step completion sidecars and the run completion record were present; this probe validates recording, not learning or model quality.
