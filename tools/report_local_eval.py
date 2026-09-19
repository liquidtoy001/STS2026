"""Write an English report with an explicit, attributable completion time."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path


def completion_metadata(status, status_path):
    """Do not present legacy filesystem times as runtime measurements."""
    if status.get("completed_at_utc"):
        value = status["completed_at_utc"]
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.utcoffset() is None or parsed.utcoffset().total_seconds() != 0:
            raise ValueError("completed_at_utc must include a UTC offset.")
        return value, status.get("completion_time_source", "runtime_clock")
    return (
        datetime.fromtimestamp(status_path.stat().st_mtime, timezone.utc).isoformat(timespec="seconds"),
        "approximate: original status.json last-write time; runtime completion was not recorded",
    )


def write_report(run):
    run = Path(run).resolve()

    def read(relative):
        return json.loads((run / relative).read_text(encoding="utf-8"))

    status = read("status.json")
    if status["status"] not in ("completed", "checkpoint_evaluated"):
        raise ValueError("Only completed experiments can be reported as completed.")
    completed, source = completion_metadata(status, run / "status.json")
    generated = datetime.now(timezone.utc).isoformat(timespec="seconds")
    lines = ["# Local posterior reconstruction experiment", "",
             f"- Run: `{run.name}`", f"- Completed at (UTC): `{completed}`",
             f"- Completion time source: {source}",
             f"- Report generated at (UTC): `{generated}`", f"- Status: `{status['status']}`", ""]
    if status["status"] == "checkpoint_evaluated":
        lines += ["## Checkpoint rescoring", "",
                  "This run reloaded an existing checkpoint without retraining.", "",
                  f"- Checkpoint SHA256: `{status['checkpoint_sha256']}`",
                  f"- Training iteration recorded in checkpoint: {status['step']}", "",
                  "| Metric | Maximum absolute per-sample difference |", "|---|---:|"]
        for key, value in status["max_absolute_per_sample_differences"].items():
            lines.append(f"| {key} | {value:.9g} |")
        lines += ["", "This establishes same-checkpoint rescoring behavior only. It does not measure "
                  "cross-hardware reproducibility or variation between training runs.", "",
                  "- [Full rescoring record](status.json)"]
    else:
        measurements = read("local_measurement.json")
        history = read("evaluation/evaluation_history.json")
        manifest = read("evaluation/manifest.json")
        config = read("run_config.json")
        step = status["step"]
        lines += [f"Training seed: {config['train_seed']}. This is a single fixed-training-batch "
                  "overfit experiment, not held-out evaluation or an estimate of seed variation.", "",
                  "## Execution and resources", "",
                  f"- GPU: {measurements['device']}",
                  f"- Parameters: {measurements['parameters']:,}; original notebook model configuration.",
                  f"- Training iterations: {step}; actual optimizer updates: {measurements.get('actual_optimizer_updates', 'not recorded')}; AMP skips: {measurements.get('amp_skipped_updates', 'not recorded')}.",
                  f"- Training, evaluation and checkpoint-saving time: {measurements['wall_seconds_training_and_evaluation']:.2f} seconds.",
                  f"- Peak PyTorch allocated memory: {measurements['peak_allocated_bytes'] / 2**30:.3f} GiB; reserved memory: {measurements['peak_reserved_bytes'] / 2**30:.3f} GiB.",
                  f"- Batch SHA256: `{manifest['batch_sha256']}`", "",
                  "## Fixed-crop metrics (lower is better)", "",
                  "| Iteration | Mode | Log-mel L1 | STFT spectral convergence | STFT log magnitude |",
                  "|---:|---|---:|---:|---:|"]
        for row in history:
            lines.append(f"| {row['step']} | {row['mode']} | {row['log_mel_l1']:.6f} | {row['stft_spectral_convergence']:.6f} | {row['stft_log_magnitude']:.6f} |")
        lines += ["", "These metrics are not accuracy percentages. Mean and sampled reconstruction "
                  "are reported separately. Short training crops and training-related metrics do not "
                  "establish full-utterance naturalness or generalization.", "",
                  "![Training and evaluation curves](curves.png)", "",
                  "## Final per-sample metrics", "",
                  "| Sample | Mode | Start frame | Valid samples | Log-mel L1 | STFT spectral convergence |",
                  "|---|---|---:|---:|---:|---:|"]
        for row in read(f"evaluation/step_{step:06d}/per_sample.json"):
            lines.append(f"| {row['sample_id']} | {row['mode']} | {row['start_frame']} | {row['valid_samples']} | {row['log_mel_l1']:.6f} | {row['stft_spectral_convergence']:.6f} |")
        lines += ["", "## Evidence", "", "- [Run configuration](run_config.json)",
                  "- [Hardware and timing](local_measurement.json)",
                  "- [Evaluation protocol and input fingerprint](evaluation/manifest.json)",
                  "- [Evaluation history](evaluation/evaluation_history.json)",
                  f"- [Final per-sample scores](evaluation/step_{step:06d}/per_sample.json)",
                  f"- [Final checkpoint](checkpoints/vits_step_{step}.pt)", "",
                  "Audio file numbers map to sample_ids in the manifest. Compare target, mean and "
                  "sample audio with the same number. No human listening scores, cross-hardware "
                  "validation or multi-seed uncertainty estimates are provided."]
    path = run / "REPORT.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    print(write_report(parser.parse_args().run_dir))


if __name__ == "__main__":
    main()
