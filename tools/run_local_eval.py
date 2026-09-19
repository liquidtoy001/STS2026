"""Run the original notebook's model/training entry locally, without demo cells.

No model definitions are duplicated. AST loading executes imports, selected
configuration cells and top-level function/class definitions from the notebook.
This is an explicitly trusted repository notebook, not a general notebook loader.
"""

import argparse
import ast
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--eval-every", type=int, default=20)
    parser.add_argument("--checkpoint", type=Path,
                        help="Score a checkpoint from a previous run; do not train.")
    args = parser.parse_args()
    if args.steps < 1 or args.eval_every < 1:
        parser.error("steps and eval-every must be positive")
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    os.environ["STS_REPO_ROOT"] = str(root)
    os.environ["STS_RUN_ROOT"] = str(root / "output/local-eval")
    os.environ["MPLBACKEND"] = "Agg"
    notebook = json.loads((root / "notebooks/vits-light_v1.ipynb").read_text(encoding="utf-8"))
    cells = notebook["cells"]
    namespace = {"__name__": "sts_local_notebook"}

    def source(index):
        value = cells[index]["source"]
        return value if isinstance(value, str) else "".join(value)

    def execute(index, replacements=None):
        tree = ast.parse(source(index))
        for node in tree.body:
            if isinstance(node, ast.Assign) and len(node.targets) == 1:
                name = getattr(node.targets[0], "id", None)
                if replacements and name in replacements:
                    node.value = ast.Constant(replacements[name])
        exec(compile(ast.fix_missing_locations(tree), f"notebook_cell_{index}", "exec"), namespace)

    for index in (1, 3, 7):
        execute(index)
    for index, cell in enumerate(cells):
        if cell["cell_type"] != "code":
            continue
        definitions = [node for node in ast.parse(source(index)).body
                       if isinstance(node, (ast.FunctionDef, ast.ClassDef))]
        if definitions:
            exec(compile(ast.Module(body=definitions, type_ignores=[]),
                         f"notebook_definitions_{index}", "exec"), namespace)
    torch = namespace["torch"]
    if not torch.cuda.is_available():
        raise RuntimeError("This measured local run requires CUDA; CPU tests are available separately.")
    torch.set_num_threads(4)
    data_dir = args.data_dir.resolve()
    namespace.update(DATA_DIR=data_dir, WAV_DIR=data_dir / "wavs", METADATA_PATH=data_dir / "metadata.csv")
    namespace["hann_window"] = torch.hann_window(namespace["WIN_LENGTH"])
    namespace["metadata_df"] = namespace["pd"].read_csv(
        data_dir / "metadata.csv", sep="|", header=None,
        names=["file_name", "transcription", "normalized_transcription"],
    )
    execute(42, {"TOTAL_STEPS": args.steps, "EVAL_EVERY": args.eval_every, "TRAIN_SEED": args.seed})
    # The overfit method gets an explicit fixed_batch and never iterates this loader.
    namespace["train_loader"] = [namespace["fixed_batch"]]
    execute(44)
    run_dir = namespace["RUN_DIR"]
    trainer = namespace["trainer"]
    if args.checkpoint:
        checkpoint_path = args.checkpoint.resolve()
        reference_run = checkpoint_path.parent.parent
        reference_manifest = json.loads((reference_run / "evaluation/manifest.json").read_text(encoding="utf-8"))
        current_manifest = json.loads((run_dir / "evaluation/manifest.json").read_text(encoding="utf-8"))
        for key in ("protocol", "config", "batch_sha256", "sample_ids", "start_frames", "valid_samples"):
            if reference_manifest[key] != current_manifest[key]:
                raise ValueError(f"Checkpoint evaluation protocol differs: {key}")
        checkpoint = torch.load(checkpoint_path, map_location=trainer.device, weights_only=True)
        trainer.base_model.load_state_dict(checkpoint["model"], strict=True)
        step = checkpoint["step"]
        result = namespace["evaluator"].evaluate(trainer.base_model, step, save_audio=True)
        reference = json.loads((reference_run / f"evaluation/step_{step:06d}/summary.json").read_text(encoding="utf-8"))
        differences = []
        for row in result:
            previous = next(r for r in reference if r["mode"] == row["mode"])
            differences.append({"mode": row["mode"], **{
                key: abs(row[key] - previous[key]) for key in
                ("log_mel_l1", "stft_spectral_convergence", "stft_log_magnitude")
            }})
        relative_scores = f"evaluation/step_{step:06d}/per_sample.json"
        old_rows = json.loads((reference_run / relative_scores).read_text(encoding="utf-8"))
        new_rows = json.loads((run_dir / relative_scores).read_text(encoding="utf-8"))
        old_by_id = {(r["sample_id"], r["mode"]): r for r in old_rows}
        max_sample_difference = {
            key: max(abs(row[key] - old_by_id[(row["sample_id"], row["mode"])][key])
                     for row in new_rows)
            for key in ("log_mel_l1", "stft_spectral_convergence", "stft_log_magnitude")
        }
        import hashlib
        report = {"status": "checkpoint_evaluated", "step": step,
                  "completed_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                  "completion_time_source": "runtime_clock",
                  "checkpoint": str(checkpoint_path),
                  "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
                  "absolute_summary_differences": differences,
                  "max_absolute_per_sample_differences": max_sample_difference,
                  "scope": "same checkpoint reevaluation, not retraining or cross-hardware validation"}
        (run_dir / "status.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2), flush=True)
        print("RUN_DIR:", run_dir, flush=True)
        return
    optimizer_updates = []
    # GradScaler can skip optimizer.step on non-finite gradients. Count actual
    # calls rather than assuming every notebook loop iteration updated weights.
    hook = trainer.optimizer.register_step_post_hook(
        lambda optimizer, arguments, keywords: optimizer_updates.append(time.perf_counter())
    )
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    try:
        execute(45)
    finally:
        hook.remove()
        torch.cuda.synchronize()
        measurement = {
            "recorded_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "steps_requested": args.steps, "steps_completed": trainer.step,
            "actual_optimizer_updates": len(optimizer_updates),
            "amp_skipped_updates": trainer.step - len(optimizer_updates),
            "wall_seconds_training_and_evaluation": time.perf_counter() - start,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "parameters": sum(p.numel() for p in trainer.base_model.parameters()),
            "device": torch.cuda.get_device_name(),
            "scope": "full original configuration; eight real utterances; training plus evaluation timing",
        }
        (run_dir / "local_measurement.json").write_text(json.dumps(measurement, indent=2), encoding="utf-8")
        print(json.dumps(measurement, indent=2), flush=True)
        print("RUN_DIR:", run_dir, flush=True)
    from sts_eval import plot_history
    import matplotlib.pyplot as plt
    figure = plot_history(namespace["evaluator"].history, trainer.training_history, run_dir / "curves.png")
    plt.close(figure)


if __name__ == "__main__":
    main()
