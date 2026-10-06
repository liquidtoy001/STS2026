"""Run every full-training notebook cell on two test tones, then resume once.

This is an integration check, NOT training on real speakers or a quality result.
Uses a temporary corpus and deletes its checkpoints when done. Run from the
repository root with the notebook environment; no downloads are required.
"""

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile

import torch
import torchaudio


def main():
    root = Path(__file__).resolve().parents[1]
    torch.set_num_threads(2)
    notebook = json.loads((root / "notebooks/vits-full-train.ipynb").read_text(encoding="utf-8"))
    previous_environment = dict(os.environ)
    evidence = {"scope": "synthetic_tones_not_speech_quality", "model_version": "TTS_0.02",
                "torch": torch.__version__, "sessions": []}
    try:
        with tempfile.TemporaryDirectory(prefix="speaker-smoke-", dir=root / "output") as temporary:
            corpus = Path(temporary)
            (corpus / "wavs").mkdir()
            for index, frequency in enumerate((220, 330)):
                time = torch.arange(12288) / 22050
                wave = (0.2 * torch.sin(2 * torch.pi * frequency * time)).unsqueeze(0)
                torchaudio.save(str(corpus / "wavs" / f"tone{index}.wav"), wave, 22050)
            manifest = corpus / "speakers.csv"
            manifest.write_text("file_name|normalized_transcription|speaker_name\n"
                                "tone0|Hi.|tone-a\ntone1|Hi.|tone-b\n", encoding="utf-8")
            # Avoid inheriting a user's real corpus, resume path or long-run settings.
            for key in list(os.environ):
                if key.startswith("STS_"):
                    del os.environ[key]
            os.environ.update(STS_DATA_DIR=str(corpus), STS_SPEAKER_MANIFEST=str(manifest),
                              STS_REPO_ROOT=str(root), STS_RUN_ROOT=str(corpus / "runs"),
                              STS_NUM_WORKERS="0", STS_TOTAL_STEPS="2", STS_EVAL_EVERY="1",
                              STS_SAVE_EVERY="1", STS_KEEP_EVERY="1", STS_LOG_EVERY="1",
                              STS_GAN_WEIGHT="1.0", STS_TIME_BUDGET_HOURS="0", MPLBACKEND="Agg")
            for session in range(2):
                namespace = {"__name__": "__main__"}
                for index, cell in enumerate(notebook["cells"]):
                    if cell["cell_type"] == "code":
                        print(f"Session {session + 1}, cell {index}", flush=True)
                        exec(compile("".join(cell["source"]), f"notebook_cell_{index}", "exec"), namespace)
                run_dir = namespace["RUN_DIR"]
                status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
                expected = "stopped_time_budget" if session == 0 else "completed"
                if status["status"] != expected or status["step"] != session + 1:
                    raise AssertionError(status)
                if status["actual_optimizer_updates"] != session + 1:
                    raise AssertionError("Resume did not restore optimizer-update count.")
                checkpoint = torch.load(run_dir / "checkpoints" / f"vits_step_{session + 1}.pt",
                                        map_location="cpu", weights_only=True)
                if checkpoint["speaker_config"]["speaker_to_id"] != {"tone-a": 0, "tone-b": 1}:
                    raise AssertionError("Speaker mapping was not retained.")
                if len(list((run_dir / "samples").glob(f"step_{session + 1}_sample_*.wav"))) != 2:
                    raise AssertionError("Conditioned text-only samples were not written.")
                evidence["sessions"].append(status)
                evidence["device"] = str(namespace["DEVICE"])
                evidence["gpu"] = namespace["run_config"]["gpu"]
                os.environ["STS_RESUME_FROM"] = str(run_dir)
                os.environ["STS_TIME_BUDGET_HOURS"] = "0.1"
                del namespace, checkpoint
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    finally:
        os.environ.clear()
        os.environ.update(previous_environment)
    evidence["completed_at_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    evidence["completion_time_source"] = "runtime_clock"
    output = root / "output" / "speaker-notebook-smoke.json"
    output.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
