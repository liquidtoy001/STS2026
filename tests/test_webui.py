"""Web UI checks: text preparation, checkpoint loading, chunked synthesis and the HTTP routes.

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

from tools import chunk_study
from tools.notebook_defs import literal_assignment, load_definitions
from webui import text_en
from webui.engine import Engine, chunk_span
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

    def test_chunk_span_matches_the_study(self):
        words = "The quick brown fox jumps over the lazy dog.".split()
        for start, end, lookahead, context in [(0, 1, 2, True), (3, 5, 2, True), (8, 9, 4, True), (2, 4, 0, False),
                                               (0, 9, 0, True), (7, 9, 0, False)]:
            self.assertEqual(chunk_span(words, start, end, lookahead, context),
                             chunk_study.spans(words, start, end, context, lookahead))
        text, span = chunk_span(words, 2, 4, 2, True)
        self.assertEqual((text, text[span[0]:span[1]]), ("The quick brown fox jumps over", "brown fox "))
        for bad in [(4, 4, 0, True), (0, 10, 0, True), (0, 1, 2, False), (0, 1, -1, True)]:
            with self.assertRaises(ValueError):
                chunk_span(words, *bad)

    def test_chunk_synthesis_reproduces_the_study_and_the_whole_sentence(self):
        loaded = self.engine.loaded.get("single") or (self.engine.load("single"), self.engine.loaded["single"])[1]
        words = "Hi there, how are you?".split()
        for start, end, lookahead, context in [(0, 1, 2, True), (1, 3, 2, True), (2, 4, 0, False), (4, 5, 4, True)]:
            text, span = chunk_span(words, start, end, lookahead, context)
            expected = chunk_study.synth(self.engine.ns, loaded["model"], text, span, 40 + start, self.engine.device)
            audio, _ = self.engine._infer_span(loaded, text, span, 0.667, 1.0, 40 + start, None)
            self.assertTrue(torch.equal(audio.cpu(), expected))
        # Only the chunk is voiced, with the seed shifted by its first word as in the study.
        result = self.engine.synthesize_chunk("single", " ".join(words), 1, 3, lookahead=2, seed=40)
        self.assertEqual((result["voiced"], result["encoder_text"], result["seed"]),
                         ("there, how", "Hi there, how are you?", 41))
        # The whole sentence as one chunk is the reader's whole-sentence synthesis.
        whole = self.engine.synthesize_chunk("single", " ".join(words), 0, len(words), seed=9)
        self.assertEqual(whole["wav"], self.engine.synthesize("single", " ".join(words), seed=9)["wav"])
        with self.assertRaises(ValueError):
            self.engine.synthesize_chunk("single", " ".join(words), 0, 1, lookahead=2, mode="independent")
        with self.assertRaises(ValueError):
            self.engine.synthesize_chunk("multi", "Hi there.", 0, 1, seed=1)
        self.assertGreater(self.engine.synthesize_chunk("multi", "Hi there.", 0, 1, seed=1,
                                                        speaker="speaker_0")["seconds"], 0)

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
            status, body = call("/api/synthesize_chunk", {"model": "single", "text": "Hi there, how", "start": 1,
                                                          "end": 2, "lookahead": 1, "mode": "context", "seed": 3})
            result = json.loads(body)
            self.assertEqual((status, result["voiced"], result["encoder_text"]), (200, "there,", "Hi there, how"))
            with wave.open(io.BytesIO(base64.b64decode(result["wav"]))) as handle:
                self.assertEqual(handle.getnframes() / handle.getframerate(), result["seconds"])
            status, body = call("/api/synthesize_chunk", {"model": "single", "text": "Hi there", "start": 1, "end": 5})
            self.assertEqual(status, 400)
            self.assertIn("outside", json.loads(body)["error"])
            self.assertEqual(call("/api/synthesize_chunk", {"model": "single", "text": "Hi"})[0], 400)
        finally:
            server.shutdown()
            server.server_close()


if __name__ == "__main__":
    unittest.main()
