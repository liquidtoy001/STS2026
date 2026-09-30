"""Import Jacques's ZIP into small Git records and the existing HTML report.

Source JSON bytes are preserved. Audio, screenshots and duplicate downloads are
indexed but remain in Drive/the original ZIP. No archive code is executed.
"""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import zipfile


ROOT = Path(__file__).resolve().parents[1]
REPORT_ID = "jacques-enc-dec"
ARMS = {"original": "vits-eval-original", "new-enc-dec": "vits-eval-new-enc-dec"}
METRICS = ("log_mel_l1", "stft_spectral_convergence", "stft_log_magnitude")
TRAIN_FIELDS = ("step", "optimizer_updates", "total", "mel", "stft", "duration", "kl",
                "wave", "gan", "sec_per_iter")


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def record_path(relative):
    """Return a normalized JSON record path; never extract arbitrary ZIP paths."""
    parts = PurePosixPath(relative).parts
    if len(parts) == 1 and parts[0] in ("run_config.json", "status.json", "training_history.json"):
        return Path(parts[0])
    if len(parts) == 1 and parts[0] in ("manifest.json", "manifest (1).json", "evaluation_history.json"):
        return Path("evaluation") / parts[0].replace("manifest (1).json", "manifest.json")
    if (len(parts) == 2 and len(parts[0]) == 11 and parts[0].startswith("Step_")
            and parts[0][5:].isdigit() and parts[1] in ("summary.json", "per_sample.json", "completion.json")):
        return Path("evaluation") / parts[0].lower() / parts[1]
    return None


def read_arm(archive, folder):
    prefix = f"Jacques Eval/{folder}/"
    records, inventory = {}, []
    for entry in archive.infolist():
        if entry.is_dir() or not entry.filename.startswith(prefix):
            continue
        relative = entry.filename[len(prefix):]
        if ".." in PurePosixPath(relative).parts or "\\" in relative:
            raise ValueError(f"Unsafe archive path: {entry.filename}")
        data = archive.read(entry)
        target = record_path(relative)
        if target is not None:
            if target in records:
                raise ValueError(f"Duplicate normalized record: {target}")
            records[target] = data
        inventory.append({"source_path": entry.filename, "bytes": len(data),
                          "sha256": sha256(data),
                          "git_record": target.as_posix() if target is not None else None})
    return records, inventory


def decode(records, name):
    return json.loads(records[Path(name)].decode("utf-8"))


def validate_arm(records):
    config = decode(records, "run_config.json")
    status = decode(records, "status.json")
    manifest = decode(records, "evaluation/manifest.json")
    history = decode(records, "evaluation/evaluation_history.json")
    training = decode(records, "training_history.json")
    if status.get("status") != "completed" or status.get("step") != config.get("total_steps"):
        raise ValueError("Run is not completed at its requested final iteration")
    timestamp = datetime.fromisoformat(status["completed_at_utc"])
    if timestamp.tzinfo is None:
        raise ValueError("Completion timestamp must include a timezone")
    expected_steps = list(range(0, status["step"] + 1, config["eval_every"]))
    if sorted({row["step"] for row in history}) != expected_steps:
        raise ValueError("Evaluation history does not cover the expected checkpoints")
    if training[-1]["optimizer_updates"] != status["actual_optimizer_updates"]:
        raise ValueError("Final optimizer counts disagree")
    if status["actual_optimizer_updates"] + status["amp_skipped_updates"] != status["step"]:
        raise ValueError("AMP iteration accounting disagrees")
    for step in expected_steps:
        stem = f"evaluation/step_{step:06d}"
        summary, per_sample = decode(records, f"{stem}/summary.json"), decode(records, f"{stem}/per_sample.json")
        for mode in ("mean", "sample"):
            rows = [row for row in per_sample if row["mode"] == mode]
            aggregate = next(row for row in summary if row["mode"] == mode)
            recorded = next(row for row in history if row["mode"] == mode and row["step"] == step)
            if sorted(row["sample_id"] for row in rows) != sorted(manifest["sample_ids"]):
                raise ValueError("Per-sample IDs disagree with the manifest")
            if len(rows) != aggregate["count"]:
                raise ValueError("Per-sample count disagrees")
            for metric in METRICS:
                mean = sum(row[metric] for row in rows) / len(rows)
                if not math.isfinite(mean) or not math.isclose(mean, aggregate[metric], abs_tol=1e-12):
                    raise ValueError(f"Inconsistent metric: {step}, {mode}, {metric}")
                if aggregate[metric] != recorded[metric]:
                    raise ValueError("Summary and history disagree")
    # Display-only derivation from the manifest; raw source JSON remains untouched.
    display_config = dict(config, batch_size=len(manifest["sample_ids"]),
                          torch=manifest["environment"]["torch"])
    return {"run": config.get("variant", "unknown"), "status": status,
            "config": display_config, "batch_sha256": manifest["batch_sha256"],
            "eval": history, "train": {key: [row.get(key) for row in training] for key in TRAIN_FIELDS},
            "per_sample_final": decode(records, f"evaluation/step_{status['step']:06d}/per_sample.json"),
            "checkpoints": []}, manifest


def build(archive_path, records_root=None, site_root=None):
    archive_path = Path(archive_path)
    records_root = Path(records_root) if records_root is not None else ROOT / "docs/records"
    site_root = Path(site_root) if site_root is not None else ROOT / "docs/site"
    report = {"id": REPORT_ID, "reference": "original", "arms": {}}
    bundles, manifests, inventories = {}, {}, {}
    # Validate all inputs before publishing either arm.
    with zipfile.ZipFile(archive_path) as archive:
        for arm, folder in ARMS.items():
            records, inventory = read_arm(archive, folder)
            display, manifest = validate_arm(records)
            display["run"] = folder  # Supplier folder label, not a model identity assertion.
            report["arms"][arm], bundles[arm], manifests[arm], inventories[arm] = display, records, manifest, inventory
    protocol_keys = ("protocol", "config", "batch_sha256", "sample_ids", "start_frames", "valid_samples", "environment")
    checks = {key: manifests["original"][key] == manifests["new-enc-dec"][key] for key in protocol_keys}
    if not all(checks.values()):
        raise ValueError("The paired reconstruction protocols differ")
    report["assessment"] = {"component_identity": "pending_verification", "protocol_matches": checks,
                            "identical_run_config_bytes": bundles["original"][Path("run_config.json")] == bundles["new-enc-dec"][Path("run_config.json")]}
    for arm, records in bundles.items():
        for relative, data in records.items():
            destination = records_root / REPORT_ID / arm / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(data)
    audit = {"source_archive": archive_path.name, "source_archive_sha256": sha256(archive_path.read_bytes()),
             "imported_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
             "scope": "Import and arithmetic validation; not a new training run or verified architecture ablation",
             "assessment": report["assessment"], "arms": inventories,
             "file_counts": {arm: dict(Counter(PurePosixPath(x["source_path"]).suffix for x in files)) for arm, files in inventories.items()}}
    (records_root / REPORT_ID / "inventory.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    output = site_root / "reports" / f"{REPORT_ID}.data.js"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("/* Generated by tools/import_jacques_eval.py; source values are unrounded. */\n"
                      f'(window.STS_REPORTS = window.STS_REPORTS || {{}})["{REPORT_ID}"] = '
                      + json.dumps(report, ensure_ascii=False, separators=(",", ":")) + ";\n", encoding="utf-8")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    args = parser.parse_args()
    print(build(args.archive))


if __name__ == "__main__":
    main()
