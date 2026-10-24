"""English text -> phoneme input for TTS_0.20 arms with text_input "phonemes" (g2p_en).

Every whitespace word of the (normalized) text becomes one phoneme word, so word boundaries,
chunking and chunk-aware training work as with characters. Each ARPAbet phoneme (stress kept)
is written as one character, chr(0x100 + index), so a phoneme word is a plain string and
the notebook's encode_text/word_spans need no change. Punctuation g2p_en keeps (, . ? ! - ')
stays as itself; ; and : become commas; a word with nothing to pronounce becomes a comma (a
pause), so the word count never changes.

Pronunciations follow g2p_en: homographs by part-of-speech tag (tagged over the whole text),
then CMUdict, then its neural model for out-of-vocabulary words.

  python tools/phonemize.py --metadata data/LJSpeech-1.1/metadata.csv \
      --out data/LJSpeech-1.1/phonemes_g2p_en.tsv

needs g2p_en and its NLTK data (averaged_perceptron_tagger_eng, cmudict) under $NLTK_DATA.
"""

import argparse
import csv
import hashlib
import re
import sys
import unicodedata
from pathlib import Path

ARPABET = [
    "AA0", "AA1", "AA2", "AE0", "AE1", "AE2", "AH0", "AH1", "AH2", "AO0", "AO1", "AO2", "AW0", "AW1",
    "AW2", "AY0", "AY1", "AY2", "B", "CH", "D", "DH", "EH0", "EH1", "EH2", "ER0", "ER1", "ER2", "EY0",
    "EY1", "EY2", "F", "G", "HH", "IH0", "IH1", "IH2", "IY0", "IY1", "IY2", "JH", "K", "L", "M", "N",
    "NG", "OW0", "OW1", "OW2", "OY0", "OY1", "OY2", "P", "R", "S", "SH", "T", "TH", "UH0", "UH1", "UH2",
    "UW", "UW0", "UW1", "UW2", "V", "W", "Y", "Z", "ZH",
]
PHONEME_CHAR = {phoneme: chr(0x100 + index) for index, phoneme in enumerate(ARPABET)}
CHAR_PHONEME = {char: phoneme for phoneme, char in PHONEME_CHAR.items()}
KEPT_PUNCTUATION = ",.?!-'"
FORMAT = "sts2026-phonemes-g2p_en-v1"


def clean_word(word):
    """g2p_en's character cleaning, applied to one word (; and : kept as commas)."""
    word = "".join(c for c in unicodedata.normalize("NFD", word) if unicodedata.category(c) != "Mn").lower()
    word = re.sub("[;:]", ",", word)
    return re.sub(r"[^a-z'.,?!\-]", "", word)


class Phonemizer:
    def __init__(self):
        from g2p_en import G2p
        from g2p_en.g2p import normalize_numbers, word_tokenize
        from nltk import pos_tag
        self.g2p, self.normalize_numbers, self.tokenize, self.pos_tag = G2p(), normalize_numbers, word_tokenize, pos_tag

    def _pronounce(self, token, tag):
        if re.search("[a-z]", token) is None:
            return [c for c in token if c in KEPT_PUNCTUATION]
        if token in self.g2p.homograph2features:
            first, second, tag_prefix = self.g2p.homograph2features[token]
            return first if tag.startswith(tag_prefix) else second
        if token in self.g2p.cmu:
            return self.g2p.cmu[token][0]
        return self.g2p.predict(token)

    def words(self, text):
        """[(word, [phoneme or punctuation, ...])] for every whitespace word after number normalization."""
        words = self.normalize_numbers(text).split()
        pieces = [self.tokenize(clean_word(word)) for word in words]
        tags = iter(self.pos_tag([t for p in pieces for t in p]) if any(pieces) else [])
        result = []
        for word, tokens in zip(words, pieces):
            symbols = []
            for token in tokens:
                _, tag = next(tags)
                symbols += self._pronounce(token, tag)
            unknown = [s for s in symbols if s not in PHONEME_CHAR and s not in KEPT_PUNCTUATION]
            if unknown:
                raise ValueError(f"g2p_en produced symbols outside the phoneme table: {unknown}")
            result.append((word, symbols or [","]))
        return result

    def __call__(self, text):
        """Phoneme string: one character per phoneme, words separated by single spaces."""
        return " ".join("".join(PHONEME_CHAR.get(s, s) for s in symbols) for _, symbols in self.words(text))


def readable(phoneme_string):
    """'HH AH0 L OW1 | W ER1 L D' style rendering of a phoneme string, for logs and tests."""
    return " | ".join(" ".join(CHAR_PHONEME.get(c, c) for c in word) for word in phoneme_string.split(" "))


def read_cache(path):
    """{file_name: phoneme string} and the header fields of a cache written by this tool."""
    header, mapping = {}, {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.startswith("# "):
            key, _, value = line[2:].partition(": ")
            header[key] = value
        elif line:
            name, _, phonemes = line.partition("|")
            mapping[name] = phonemes
    if header.get("format") != FORMAT:
        raise ValueError(f"{path} is not a {FORMAT} file.")
    return mapping, header


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--metadata", required=True, help="LJSpeech-style metadata.csv (id|text|normalized text)")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    import g2p_en
    import nltk
    phonemizer = Phonemizer()
    with open(args.metadata, encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle, delimiter="|", quoting=csv.QUOTE_NONE))
    lines = [f"# format: {FORMAT}",
             f"# metadata_sha256: {hashlib.sha256(Path(args.metadata).read_bytes()).hexdigest()}",
             f"# g2p_en: {getattr(g2p_en, '__version__', 'unknown')}; nltk: {nltk.__version__}",
             "# columns: file_name|phonemes (one character per ARPAbet phoneme, see tools/phonemize.py)"]
    for index, row in enumerate(rows):
        text = row[2].strip()
        phonemes = phonemizer(text)
        if len(phonemes.split(" ")) != len(phonemizer.normalize_numbers(text).split()):
            raise AssertionError(f"word count changed for {row[0]}")
        lines.append(f"{row[0]}|{phonemes}")
        if index % 2000 == 0:
            print(index, row[0], readable(phonemes)[:120], flush=True)
    Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(args.out, len(rows))


if __name__ == "__main__":
    sys.exit(main())
