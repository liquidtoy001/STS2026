"""Chunked-synthesis study tools: chunk text spans, waiting words, corpus WER and paired intervals."""

import json
from pathlib import Path
import tempfile
import unittest

from tools.chunk_analysis import analyze, compare, markdown, wait_words
from tools.chunk_study import CONDITIONS, spans


def write_study(folder, errors, words, natural=True):
    """asr.json/timing.json for channels {name: [errors per sentence]} with words[i] reference words."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    channels = dict(errors, **({"natural": [0] * len(words)} if natural else {}))
    utterances = [{"id": f"S{i}", "channel": channel, "word_s": count, "word_d": 0, "word_i": 0, "word_n": words[i]}
                  for channel, counts in channels.items() for i, count in enumerate(counts)]
    (folder / "asr.json").write_text(json.dumps({"asr": {"model": "medium"}, "utterances": utterances}))
    (folder / "timing.json").write_text(json.dumps({"summary": {c: {"duration_ratio_mean": 1.0} for c in errors}}))
    return folder


class ChunkStudyTests(unittest.TestCase):
    def test_conditions_and_chunk_spans(self):
        self.assertEqual(len(CONDITIONS), 17)
        words = "a bb ccc dddd eeeee".split()
        text, (start, end) = spans(words, 2, 4, False, 0)  # independent chunk: the chunk alone
        self.assertEqual((text, text[start:end]), ("ccc dddd", "ccc dddd"))
        text, (start, end) = spans(words, 2, 4, True, 1)  # left context + chunk + one lookahead word
        self.assertEqual((text, text[start:end]), ("a bb ccc dddd eeeee", "ccc dddd "))
        text, (start, end) = spans(words, 0, 2, True, 4)  # lookahead beyond the sentence end
        self.assertEqual((text, text[start:end]), ("a bb ccc dddd eeeee", "a bb "))
        text, (start, end) = spans(words, 4, 5, True, 2)  # last chunk: no following space
        self.assertEqual(text[start:end], "eeeee")

    def test_waiting_words(self):
        words = {"S0": 10, "S1": 3}
        self.assertEqual(wait_words("full", words), 6.5)
        self.assertEqual(wait_words("naive_2", words), 2.0)  # min(2, 10), min(2, 3)
        self.assertEqual(wait_words("ctx_2_k4", words), 4.5)  # min(6, 10), min(6, 3)

    def test_analyze_and_compare(self):
        words = [10, 20, 10, 10, 10]
        base = {"full": [1, 2, 1, 1, 0], "naive_2": [4, 6, 3, 2, 5], "naive_4": [2, 4, 2, 2, 1],
                "ctx_2_k2": [1, 3, 1, 1, 1]}
        with tempfile.TemporaryDirectory() as root:
            reference = write_study(Path(root) / "R0", base, words)
            result = analyze(reference)
            self.assertAlmostEqual(result["conditions"]["naive_2"]["wer"], 20 / 60)  # corpus rate, not a mean
            low, high = result["conditions"]["naive_2"]["vs_full_ci95"]
            self.assertTrue(low <= 20 / 60 - 5 / 60 <= high)
            self.assertIn("ctx_2_k2 vs naive_4", result["matched_wait"])
            self.assertEqual(result["natural_wer"], 0.0)
            self.assertTrue((reference / "tradeoff.png").is_file())

            better = {name: [max(0, count - 1) for count in counts] for name, counts in base.items()}
            chunk = write_study(Path(root) / "chunk", better, words, natural=False)
            control = write_study(Path(root) / "control", base, words, natural=False)
            result = compare({"R0": reference, "chunk": chunk, "control": control}, {})
            entry = result["chunk_study"]["conditions"]["naive_2"]
            self.assertEqual(entry["control"]["vs_R0"], {"diff": 0.0, "ci95": [0.0, 0.0]})
            self.assertAlmostEqual(entry["chunk"]["vs_control"]["diff"], -5 / 60)
            self.assertLess(entry["chunk"]["vs_control"]["ci95"][1], 0)  # one error fewer in every sentence
            self.assertEqual(result["chunk_study"]["natural_wer"], 0.0)
            table = markdown(result)
            self.assertIn("| independent 2-word chunks | 33.3 | 25.0 | 33.3 |", table)
            short = write_study(Path(root) / "short", {k: v[:4] for k, v in base.items()}, words[:4], natural=False)
            with self.assertRaisesRegex(ValueError, "same sentences"):
                compare({"R0": reference, "short": short}, {})


if __name__ == "__main__":
    unittest.main()
