"""tools/prepare_libritts_r.py on a tiny synthetic LibriTTS-R tree (tones, not speech)."""

import json
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import soundfile
import torch

from tools.notebook_defs import load_definitions
from tools.prepare_libritts_r import SOURCE_RATE, TARGET_RATE, clean_text, convert, draw_heldout

ROOT = Path(__file__).resolve().parents[1]


def tone(seconds, frequency=220.0, amplitude=0.3):
    time = np.arange(int(seconds * SOURCE_RATE)) / SOURCE_RATE
    return (amplitude * np.sin(2 * np.pi * frequency * time)).astype("float32")


class PrepareLibriTTSRTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def write_utterance(self, subset, utterance, text, seconds):
        speaker, chapter = utterance.split("_")[:2]
        folder = self.root / "LibriTTS_R" / subset / speaker / chapter
        folder.mkdir(parents=True, exist_ok=True)
        soundfile.write(str(folder / f"{utterance}.wav"), tone(seconds), SOURCE_RATE, subtype="PCM_16")
        (folder / f"{utterance}.normalized.txt").write_text(text, encoding="utf-8")

    def test_clean_text_maps_typographic_variants_and_reports_the_rest(self):
        self.assertEqual(clean_text("It’s  a test – ok.\n"), ("It's a test — ok.", []))
        self.assertEqual(clean_text("Room 101, café & co.")[1], sorted(set("01&é")))
        self.assertEqual(clean_text("café")[0], "café")  # NFC; é is then outside the table
        self.assertEqual(clean_text("café")[1], ["é"])

    def test_convert_resamples_to_pcm16_and_reuses_a_valid_output(self):
        source, target = self.root / "a.wav", self.root / "out" / "a.wav"
        soundfile.write(str(source), tone(1.3, amplitude=1.2), SOURCE_RATE, subtype="FLOAT")
        record = convert((str(source), str(target)))
        info = soundfile.info(str(target))
        expected = -(-int(1.3 * SOURCE_RATE) * TARGET_RATE // SOURCE_RATE)
        self.assertEqual((info.samplerate, info.channels, info.subtype, info.frames),
                         (TARGET_RATE, 1, "PCM_16", expected))
        self.assertGreater(record["clipped"], 0)
        self.assertFalse(record["reused"])
        self.assertTrue(convert((str(source), str(target)))["reused"])
        # A 220 Hz tone keeps its frequency after resampling.
        audio, _ = soundfile.read(str(target))
        spectrum = np.abs(np.fft.rfft(audio))
        self.assertAlmostEqual(np.argmax(spectrum) * TARGET_RATE / len(audio), 220.0, delta=2.0)

    def test_draw_heldout_is_seeded_one_per_speaker_and_only_from_train_clean_100(self):
        kept = [("train-clean-100", str(s), f"{s}_1_{u:06d}_000000", 2.0 + u, 5 + u % 3)
                for s in range(10, 40) for u in range(6)]
        kept += [("train-clean-360", "99", "99_1_000000_000000", 4.0, 9)]
        kept += [("train-clean-100", "77", "77_1_000000_000000", 1.5, 9)]  # too short to be a candidate
        first, second = draw_heldout(kept, 20, 2026), draw_heldout(list(reversed(kept)), 20, 2026)
        self.assertEqual(first, second)
        self.assertEqual(first, sorted(first))
        self.assertEqual(len({u.split("_")[0] for u in first}), 20)
        self.assertNotIn("99_1_000000_000000", first)
        self.assertNotIn("77_1_000000_000000", first)
        self.assertTrue(all(2.0 <= 2.0 + int(u.split("_")[2]) <= 10.0 for u in first))
        self.assertNotEqual(first, draw_heldout(kept, 20, 2027))
        with self.assertRaises(ValueError):
            draw_heldout(kept, 31, 2026)

    def test_end_to_end_manifest_loads_in_the_notebook_and_test_set_round_trips(self):
        rng = random.Random(5)
        for speaker, name in (("103", "Ann"), ("19", "Bob"), ("250", "Cy")):
            for index in range(3):
                self.write_utterance("train-clean-100", f"{speaker}_7_{index:06d}_000000",
                                     f"{name} says hello, number {'one two three'.split()[index]}.",
                                     2.0 + rng.random())
        self.write_utterance("train-clean-100", "19_7_000009_000000", "It costs 5 dollars.", 3.0)  # digit
        self.write_utterance("train-clean-100", "19_7_000010_000000", "Too short.", 0.5)
        self.write_utterance("train-clean-100", "19_7_000011_000000", "Far too long for the filter.", 4.5)
        self.write_utterance("train-clean-100", "103_7_000012_000000", "A failed restoration here.", 3.0)
        self.write_utterance("train-clean-360", "40_8_000000_000000", "From the other subset.", 3.0)
        self.write_utterance("train-clean-100", "250_7_000013_000000", "No text file.", 3.0)
        (self.root / "LibriTTS_R/train-clean-100/250/7/250_7_000013_000000.normalized.txt").unlink()
        failed = self.root / "failed"
        failed.mkdir()
        (failed / "train-clean-100_bad_sample_list.txt").write_text(
            "./train-clean-100/103/7/103_7_000012_000000.wav\n", encoding="utf-8")
        (failed / "train-clean-360_bad_sample_list.txt").write_text("", encoding="utf-8")
        out, heldout = self.root / "corpus", self.root / "heldout.txt"

        def run(subsets, name):
            command = [sys.executable, str(ROOT / "tools" / "prepare_libritts_r.py"), "--root",
                       str(self.root / "LibriTTS_R"), "--out", str(out), "--subsets", *subsets, "--name", name,
                       "--failed-lists", str(failed), "--max-seconds", "4.0", "--heldout", str(heldout),
                       "--heldout-count", "2", "--workers", "2"]
            subprocess.run(command, check=True, capture_output=True, text=True, cwd=ROOT)
            return json.loads((out / f"prepare_{name}.json").read_text(encoding="utf-8"))

        small = run(["train-clean-100"], "small")
        self.assertEqual(small["counts"], {"scanned": 14, "no_text": 1, "failed_restoration": 1, "characters": 1,
                                           "too_short": 1, "too_long": 1, "kept": 9})
        self.assertEqual(small["dropped_characters"]["5"]["examples"], ["19_7_000009_000000"])
        self.assertEqual(small["heldout"]["action"], "created")
        self.assertEqual(small["heldout"]["speakers"], 2)
        both = run(["train-clean-100", "train-clean-360"], "both")
        self.assertEqual(both["counts"]["kept"], 10)
        self.assertEqual(both["heldout"]["action"], "checked")
        self.assertEqual(both["audio"]["recordings_reused"], 9)
        self.assertEqual(both["kept"]["by_subset"], {"train-clean-100": 9, "train-clean-360": 1})

        ns = load_definitions()
        frame, speaker_to_id = ns["load_training_metadata"](out / "speakers_both.csv", conditioned=True)
        self.assertEqual(speaker_to_id, {"103": 0, "19": 1, "250": 2, "40": 3})
        self.assertEqual(len(frame), 10)
        for name in frame["file_name"]:
            info = soundfile.info(str(out / "wavs" / f"{name}.wav"))
            self.assertEqual((info.samplerate, info.subtype), (TARGET_RATE, "PCM_16"))
        dataset = ns["LJSpeechDataset"](frame, out / "wavs", table=ns["token_table"]("chars"))
        text, spectrogram, waveform, speaker_id = dataset[0]
        self.assertEqual(waveform.shape[0], 1)
        self.assertTrue(torch.isfinite(spectrogram).all())
        self.assertIn(speaker_id, speaker_to_id.values())
        ids = ns["load_id_list"](heldout)
        self.assertEqual(len(ids), 2)
        self.assertTrue(set(ids) <= set(frame["file_name"]))


if __name__ == "__main__":
    unittest.main()
