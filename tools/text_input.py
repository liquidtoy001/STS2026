"""Text -> model input for any checkpoint: characters (TTS_0.10) or phonemes (TTS_0.20 option).

The checkpoint's switches decide: text_input "phonemes" runs tools/phonemize.py (g2p_en) first.
The phonemizer is created once per process, on first use.
"""

import functools


def text_input_kind(switches):
    return (switches or {}).get("text_input", "chars")


@functools.lru_cache(maxsize=1)
def phonemizer():
    from tools.phonemize import Phonemizer
    return Phonemizer()


def model_text(text, switches):
    """The string the model's token table encodes: the text itself or its phoneme string."""
    return phonemizer()(text) if text_input_kind(switches) == "phonemes" else text


def model_words(text, switches):
    """Words of the model input, one per whitespace word (phoneme words for phoneme models)."""
    return model_text(text, switches).split(" ") if text_input_kind(switches) == "phonemes" else text.split()


def table_for(ns, switches):
    return ns["token_table"](text_input_kind(switches))
