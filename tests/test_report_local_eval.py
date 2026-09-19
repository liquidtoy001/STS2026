"""Completion provenance must survive report regeneration and failed runs."""

import json
import os
from pathlib import Path
import tempfile
import unittest

from tools.report_local_eval import completion_metadata, write_report


class ReportTests(unittest.TestCase):
    def test_legacy_completion_is_attributed_not_report_generation_time(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "status.json"
            path.write_text("{}", encoding="utf-8")
            os.utime(path, (1609459200, 1609459200))
            completed, source = completion_metadata({}, path)
            self.assertEqual(completed, "2021-01-01T00:00:00+00:00")
            self.assertIn("approximate", source)

    def test_rescore_report_preserves_runtime_completion_and_is_english(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            timestamp = "2026-09-19T15:27:56+00:00"
            (run / "status.json").write_text(json.dumps({
                "status": "checkpoint_evaluated", "step": 2000,
                "completed_at_utc": timestamp, "completion_time_source": "runtime_clock",
                "checkpoint_sha256": "test-checkpoint-fingerprint",
                "max_absolute_per_sample_differences": {"log_mel_l1": 0.0},
            }), encoding="utf-8")
            text = write_report(run).read_text(encoding="utf-8")
            self.assertIn(f"Completed at (UTC): `{timestamp}`", text)
            self.assertIn("Report generated at (UTC)", text)
            self.assertTrue(text.isascii())
            self.assertEqual(write_report(run).name, "REPORT.md")

    def test_failed_runs_and_naive_timestamps_are_not_completed_reports(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            (run / "status.json").write_text('{"status": "failed_or_interrupted"}', encoding="utf-8")
            with self.assertRaises(ValueError):
                write_report(run)
            self.assertFalse((run / "REPORT.md").exists())
            with self.assertRaises(ValueError):
                completion_metadata({"completed_at_utc": "2026-09-19T15:27:56"}, run / "status.json")
