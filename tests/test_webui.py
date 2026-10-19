"""Web UI checks: text preparation, checkpoint loading and the HTTP routes.

Random-weight checkpoints establish wiring only (shapes, alignment, seeds, speaker handling),
never speech quality.
"""

import base64
import io
import json
from pathlib import Path
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
import wave

import torch

from tools.notebook_defs import literal_assignment, load_definitions
from webui import text_en
from webui.engine import Engine
from webui.server import Handler, ThreadingHTTPServer

SYMBOLS = frozenset(literal_assignment("PUNCTUATION") + literal_assignment("LETTERS"))


class TextTest(unittest.TestCase):
    def normalize(self, text):
        return text_en.normalize(text, SYMBOLS)

    def test_numbers_follow_the_ljspeech_convention(self):
        self.assertEqual(text_en.number(1905), "nineteen oh five")
        self.assertEqual(text_en.number(1500), "fifteen hundred")
        self.assertEqual(text_en.number(2000), "two thousand")
        self.assertEqual(text_en.number(2007), "two thousand seven")
        self.assertEqual(text_en.number(2026), "twenty twenty-six")
        self.assertEqual(text_en.number(3500), "three thousand five hundred")
        self.assertEqual(text_en.cardinal(1_000_021), "one million twenty-one")
        self.assertEqual(text_en.ordinal(21), "twenty-first")
        self.assertEqual(text_en.ordinal(40), "fortieth")
        self.assertEqual(text_en.ordinal(112), "one hundred twelfth")

    def test_normalize_expands_and_reports(self):
        text, dropped = self.normalize("Dr. Smith paid $12.50 at 3:30 on the 21st, i.e. 15% more.")
        self.assertEqual(text, "Doctor Smith paid twelve dollars, fifty cents at three thirty on the "
                               "twenty-first, that is fifteen percent more.")
        self.assertEqual(dropped, [])
        text, dropped = self.normalize("Café snake_case \U0001F600 § 3.14")
        self.assertEqual(text, "Cafe snake case three point one four.")
        self.assertEqual(dropped, ["§", "\U0001F600"])
        self.assertLessEqual(set(text), SYMBOLS)
        self.assertEqual(self.normalize("cut at a comma,")[0], "cut at a comma,")
        self.assertEqual(self.normalize('he said "yes"')[0], 'he said "yes".')

    def test_split_respects_abbreviations_initials_and_paragraphs(self):
        text = "Mr. Smith met Dr. J. Watson. They talked, e.g. about 3.5 cups.\nIt was fine!\n\nA heading"
        self.assertEqual(text_en.split_sentences(text), [
            "Mr. Smith met Dr. J. Watson.", "They talked, e.g. about 3.5 cups.", "It was fine!", "A heading"])

    def test_chunk_splits_long_sentences_at_clause_breaks(self):
        sentence = ", ".join(f"clause number {i} goes here" for i in range(12)) + "."
        pieces = text_en.chunk(sentence, len, 80)
        self.assertGreater(len(pieces), 1)
        self.assertTrue(all(len(p) <= 80 for p in pieces))
        self.assertEqual(" ".join(pieces).split(), sentence.split())


def save_checkpoint(path, n_speakers=0):
    """A random-weight TTS_0.10 checkpoint in the trainer's format."""
    ns = load_definitions()
    ns["PAD_ID"] = 0
    torch.manual_seed(0)
    vocab = 1 + len(literal_assignment("PUNCTUATION")) + len(literal_assignment("LETTERS"))
    gin = 8 if n_speakers else 0
    model = ns["VITSModel"](vocab_size=vocab, spec_channels=513, n_speakers=n_speakers, gin_channels=gin)
    speakers = {f"speaker_{i}": i for i in range(n_speakers)}
    torch.save({"model_version": "TTS_0.10", "switches": literal_assignment("TTS010_SWITCHES"),
                "speaker_config": {"n_speakers": n_speakers, "gin_channels": gin, "speaker_to_id": speakers},
                "step": 7, "model": model.state_dict()}, path)


class EngineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        save_checkpoint(root / "single.pt")
        save_checkpoint(root / "multi.pt", n_speakers=2)
        config = {"checkpoint_root": ".", "models": [
            {"id": "single", "label": "single", "checkpoint": "single.pt"},
            {"id": "multi", "label": "multi", "checkpoint": "multi.pt"},
            {"id": "missing", "label": "missing", "checkpoint": "runs/*/nothing.pt"}]}
        (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
        cls.engine = Engine(root / "config.json", device="cpu")

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_registry_reports_missing_checkpoints(self):
        rows = {row["id"]: row for row in self.engine.model_list()}
        self.assertTrue(rows["single"]["available"])
        self.assertFalse(rows["missing"]["available"])
        with self.assertRaises(FileNotFoundError):
            self.engine.load("missing")

    def test_synthesis_alignment_audio_and_seed(self):
        text = "Hi there."
        result = self.engine.synthesize("single", text, seed=5)
        alignment = result["alignment"]
        self.assertEqual("".join(alignment["tokens"]), text)
        self.assertTrue(alignment["hard"])
        self.assertEqual(sum(alignment["durations"]), alignment["frames"])
        with wave.open(io.BytesIO(base64.b64decode(result["wav"]))) as handle:
            self.assertEqual(handle.getframerate(), 22050)
            self.assertEqual(handle.getnframes(), alignment["frames"] * alignment["hop_length"])
        self.assertEqual(result["mel"]["frames"], alignment["frames"] + 1)  # centred STFT
        self.assertEqual(len(base64.b64decode(result["mel"]["data"])), 80 * result["mel"]["frames"])
        self.assertEqual(self.engine.synthesize("single", text, seed=5)["wav"], result["wav"])
        slower = self.engine.synthesize("single", text, seed=5, length_scale=2.0)
        self.assertGreater(slower["alignment"]["frames"], alignment["frames"])

    def test_speaker_conditioned_checkpoint_needs_a_speaker(self):
        self.assertEqual(self.engine.load("multi")["speakers"], ["speaker_0", "speaker_1"])
        with self.assertRaises(ValueError):
            self.engine.synthesize("multi", "Hi.", seed=1)
        self.assertGreater(self.engine.synthesize("multi", "Hi.", seed=1, speaker="speaker_1")["seconds"], 0)
        with self.assertRaises(ValueError):
            self.engine.synthesize("single", "Hi.", seed=1, speaker="speaker_1")

    def test_prepare_marks_segments_without_letters(self):
        segments = self.engine.prepare("First one. §§\n\nSecond 2.")
        self.assertEqual([s["normalized"] for s in segments], ["First one.", "", "Second two."])
        self.assertEqual([s["skip"] for s in segments], [False, True, False])

    def test_http_routes(self):
        Handler.engine = self.engine
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        base = f"http://127.0.0.1:{server.server_port}"

        def call(path, body=None):
            data = None if body is None else json.dumps(body).encode()
            request = urllib.request.Request(base + path, data=data, headers={"Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(request) as response:
                    return response.status, response.read()
            except urllib.error.HTTPError as error:
                return error.code, error.read()

        try:
            self.assertEqual(call("/")[0], 200)
            self.assertEqual(call("/static/app.js")[0], 200)
            self.assertEqual(call("/static/../server.py")[0], 404)
            self.assertEqual(call("/static/%2e%2e/server.py")[0], 404)
            status, body = call("/api/prepare", {"text": "One. Two."})
            self.assertEqual((status, len(json.loads(body)["segments"])), (200, 2))
            status, body = call("/api/synthesize", {"model": "nope", "text": "Hi."})
            self.assertEqual(status, 400)
            self.assertIn("nope", json.loads(body)["error"])
        finally:
            server.shutdown()
            server.server_close()


if __name__ == "__main__":
    unittest.main()
