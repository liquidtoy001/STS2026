"""English text preparation for the web UI: sentence splitting and LJSpeech-style normalisation.

The models were trained on LJSpeech `normalized_transcription` text, in which numbers and
common abbreviations are written out. `normalize` approximates that convention:

- expands integers, decimals, ordinals (21st), times (3:30), percentages, $ and pound
  amounts, and 1001-2999 in the LJSpeech/Tacotron style (1984 -> nineteen eighty-four)
- expands a fixed list of titles and abbreviations (Mr., Dr., St. -> Saint, e.g., etc.)
- replaces a few symbols with words (&, +, =, @, #) and strips accents (cafe for café)
- drops every remaining character outside the model's symbol table and reports it
- adds a full stop when a segment ends in a letter or digit

It does not handle acronyms (GPU is read as a word, not spelled out), roman numerals, dates
written with slashes, units (km, kg), URLs, homographs, or other languages.
"""

import re
import unicodedata

ONES = ("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen "
        "fifteen sixteen seventeen eighteen nineteen").split()
TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()
SCALES = ((10 ** 12, "trillion"), (10 ** 9, "billion"), (10 ** 6, "million"), (1000, "thousand"))
ORDINAL_WORDS = {"one": "first", "two": "second", "three": "third", "five": "fifth", "eight": "eighth",
                 "nine": "ninth", "twelve": "twelfth"}

# Titles and abbreviations followed by a full stop (from the Tacotron English cleaners, plus a few).
ABBREVIATIONS = {
    "mrs": "misess", "mr": "mister", "ms": "miss", "dr": "doctor", "drs": "doctors", "st": "saint",
    "co": "company", "jr": "junior", "sr": "senior", "maj": "major", "gen": "general", "rev": "reverend",
    "lt": "lieutenant", "hon": "honorable", "sgt": "sergeant", "capt": "captain", "esq": "esquire",
    "ltd": "limited", "col": "colonel", "ft": "fort", "prof": "professor", "inc": "incorporated",
    "vs": "versus", "etc": "et cetera", "approx": "approximately", "dept": "department",
}
PHRASES = {"e.g.": "for example", "i.e.": "that is"}
SYMBOL_WORDS = {"&": " and ", "+": " plus ", "=": " equals ", "@": " at ", "/": " ", "*": " ",
                "_": " ", "~": " ", "|": " ", "\t": " "}
CHARACTER_MAP = {"‘": "'", "’": "'", "–": "-", "−": "-", " ": " "}
CLOSERS = "\"'”»)]"


def cardinal(n):
    """Integer to words without 'and' or commas: 1234 -> one thousand two hundred thirty-four."""
    if n < 0:
        return "minus " + cardinal(-n)
    if n < 20:
        return ONES[n]
    if n < 100:
        return TENS[n // 10] + ("-" + ONES[n % 10] if n % 10 else "")
    if n < 1000:
        return ONES[n // 100] + " hundred" + (" " + cardinal(n % 100) if n % 100 else "")
    for value, name in SCALES:
        if n >= value:
            head, rest = divmod(n, value)
            return cardinal(head) + " " + name + (" " + cardinal(rest) if rest else "")
    raise AssertionError(n)


def number(n):
    """LJSpeech/Tacotron reading: 1001-2999 as years (1905 -> nineteen oh five), otherwise cardinal."""
    if 1000 < n < 3000:
        if n == 2000:
            return "two thousand"
        if 2000 < n < 2010:
            return "two thousand " + cardinal(n % 100)
        if n % 100 == 0:
            return cardinal(n // 100) + " hundred"
        tail = n % 100
        return cardinal(n // 100) + " " + ("oh " + cardinal(tail) if tail < 10 else cardinal(tail))
    return cardinal(n)


def ordinal(n):
    words = cardinal(n)
    head, sep, last = words.rpartition(" ")
    prefix, hyphen, last = last.rpartition("-")
    if last in ORDINAL_WORDS:
        last = ORDINAL_WORDS[last]
    elif last.endswith("y"):
        last = last[:-1] + "ieth"
    else:
        last += "th"
    return head + sep + prefix + hyphen + last


def digits(text):
    return " ".join(ONES[int(d)] for d in text)


def decimal(text):
    whole, _, fraction = text.partition(".")
    words = number(int(whole)) if whole else "zero"
    return words + (" point " + digits(fraction) if fraction else "")


def money(match):
    symbol, amount, scale = match.group(1), match.group(2).replace(",", ""), match.group(3)
    unit, units, sub, subs = (("pound", "pounds", "penny", "pence") if symbol == "£"
                              else ("dollar", "dollars", "cent", "cents"))
    if scale:
        return f"{decimal(amount)} {scale} {units}"
    whole, _, fraction = amount.partition(".")
    major = int(whole or 0)
    minor = int((fraction + "00")[:2]) if fraction else 0
    parts = []
    if major or not minor:
        parts.append(f"{cardinal(major)} {unit if major == 1 else units}")
    if minor:
        parts.append(f"{cardinal(minor)} {sub if minor == 1 else subs}")
    return ", ".join(parts)


def time_of_day(match):
    hours, minutes = int(match.group(1)), int(match.group(2))
    if minutes == 0:
        return f"{cardinal(hours)} o'clock"
    return f"{cardinal(hours)} {'oh ' if minutes < 10 else ''}{cardinal(minutes)}"


def match_case(word, source):
    return word[0].upper() + word[1:] if source[:1].isupper() else word


def expand_abbreviations(text):
    for phrase, words in PHRASES.items():
        text = re.sub(re.escape(phrase), words, text, flags=re.IGNORECASE)
    pattern = r"\b(" + "|".join(sorted(ABBREVIATIONS, key=len, reverse=True)) + r")\."
    text = re.sub(pattern, lambda m: match_case(ABBREVIATIONS[m.group(1).lower()], m.group(1)),
                  text, flags=re.IGNORECASE)
    return re.sub(r"\b(No|no)\.\s*(?=\d)", lambda m: match_case("number ", m.group(1)), text)


def expand_numbers(text):
    text = re.sub(r"([$£])\s?(\d[\d,]*(?:\.\d+)?)(?:\s?(million|billion|trillion)\b)?", money, text)
    text = re.sub(r"\b(\d{1,2}):(\d{2})\b", time_of_day, text)
    text = re.sub(r"(\d),(?=\d{3}\b)", r"\1", text)
    text = re.sub(r"(\d+(?:\.\d+)?)\s?%", lambda m: decimal(m.group(1)) + " percent", text)
    text = re.sub(r"\b(\d+)(st|nd|rd|th)\b", lambda m: ordinal(int(m.group(1))), text, flags=re.IGNORECASE)
    text = re.sub(r"#\s?(?=\d)", "number ", text)
    text = re.sub(r"\d*\.\d+|\d+", lambda m: decimal(m.group(0)) if "." in m.group(0) else number(int(m.group(0))),
                  text)
    return text


def strip_accents(text, keep):
    """Decompose accented letters to their base letter unless the character itself is a symbol."""
    out = []
    for char in text:
        if char in keep:
            out.append(char)
            continue
        out.append("".join(c for c in unicodedata.normalize("NFKD", char) if not unicodedata.combining(c)))
    return "".join(out)


def normalize(text, symbols):
    """Return (normalised text containing only characters in `symbols`, sorted dropped characters).

    `symbols` must not contain the padding symbol, which would otherwise pass through.
    """
    text = "".join(CHARACTER_MAP.get(char, char) for char in text)
    text = expand_abbreviations(text)
    text = expand_numbers(text)
    text = "".join(SYMBOL_WORDS.get(char, char) for char in text)
    text = strip_accents(text, symbols)
    dropped = sorted({char for char in text if char not in symbols and not char.isspace()})
    text = "".join(char if char in symbols else " " if char.isspace() else "" for char in text)
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s+([,.;:!?…])", r"\1", text)
    if text.rstrip(CLOSERS)[-1:].isalnum():
        text += "."  # a clause cut at a comma keeps its comma, as many LJSpeech clips do
    return text, dropped


# ---------------------------------------------------------------------------------------------
# Sentence splitting

BOUNDARY = re.compile(r"[.!?…]+[" + re.escape(CLOSERS) + r"]*\s+")
NO_SPLIT_AFTER = set(ABBREVIATIONS) | {"e.g", "i.e", "no", "vol", "fig", "p", "pp"}
BREAKS = re.compile(r"[,;:—]\s+|\s[-—]\s")


def split_sentences(text):
    """Split into sentences. Blank lines end a paragraph; single line breaks are joined."""
    sentences = []
    for paragraph in re.split(r"\n\s*\n", text.replace("\r\n", "\n")):
        paragraph = re.sub(r"\s+", " ", paragraph).strip()
        start = 0
        for match in BOUNDARY.finditer(paragraph):
            before = paragraph[start:match.start()].split(" ")[-1]
            after = paragraph[match.end():match.end() + 1]
            token = before.lower().rstrip(".")
            if paragraph[match.start()] == "." and (token in NO_SPLIT_AFTER
                                                      or re.fullmatch(r"[A-Za-z]", token)):
                continue  # Mr. Smith, e.g. this, J. K. Rowling
            if after.islower():
                continue
            sentences.append(paragraph[start:match.end()].strip())
            start = match.end()
        if paragraph[start:].strip():
            sentences.append(paragraph[start:].strip())
    return sentences


def chunk(sentence, measure, limit):
    """Split a sentence whose measured length exceeds `limit`, preferring clause breaks near the middle."""
    if measure(sentence) <= limit:
        return [sentence]
    middle = len(sentence) / 2
    cuts = [m.end() for m in BREAKS.finditer(sentence) if 0 < m.end() < len(sentence)]
    cuts = cuts or [m.end() for m in re.finditer(r"\s+", sentence) if 0 < m.end() < len(sentence)]
    if not cuts:
        cut = len(sentence) // 2
    else:
        cut = min(cuts, key=lambda c: abs(c - middle))
    left, right = sentence[:cut].strip(), sentence[cut:].strip()
    return chunk(left, measure, limit) + chunk(right, measure, limit)
