"""Convert a checkpoint saved by benlu's final-vits trainer into the TTS_0.10 checkpoint format.

benlu's file holds {"step", "g", "d"} with generator prefixes enc_p/enc_q/dec/dp/flow and no
optimizer state. The output holds the TTS_0.10 model and discriminator weights under the
notebook's names, all switches at TTS_0.10, and no optimizer or scaler state, so it is for
inference, evaluation and warm starts, not for resuming that run.

python tools/convert_benlu_checkpoint.py weight_tts_epoch5.zip checkpoints/benlu/tts010_benlu_step40000.pt
"""

import argparse
import hashlib
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.notebook_defs import literal_assignment, load_definitions  # noqa: E402

PREFIXES = {
    "enc_p.": "text_encoder.",
    "enc_q.": "posterior_encoder.",
    "flow.": "flow.",
    "dp.": "duration_predictor.",
    "dec.": "decoder.",
}


def convert_generator_state(state):
    converted = {}
    for key, value in state.items():
        prefix = next((p for p in PREFIXES if key.startswith(p)), None)
        if prefix is None:
            raise KeyError(f"Unexpected generator parameter {key!r}")
        converted[PREFIXES[prefix] + key[len(prefix):]] = value
    return converted


def build_tts010(namespace=None):
    """Unconditioned TTS_0.10 model and discriminator with the notebook's default sizes."""
    ns = load_definitions() if namespace is None else namespace
    symbols = literal_assignment("PUNCTUATION"), literal_assignment("LETTERS")
    vocab_size = 1 + len(symbols[0]) + len(symbols[1])
    model = ns["VITSModel"](vocab_size=vocab_size, spec_channels=513, padding_id=0)
    return model, ns["MultiPeriodDiscriminator"]()


def convert(source, output):
    checkpoint = torch.load(source, map_location="cpu", weights_only=True)
    if set(checkpoint) != {"step", "g", "d"}:
        raise KeyError(f"Expected benlu's keys step/g/d, found {sorted(checkpoint)}")
    model, discriminator = build_tts010()
    generator_state = convert_generator_state(checkpoint["g"])
    model.load_state_dict(generator_state, strict=True)
    discriminator.load_state_dict(checkpoint["d"], strict=True)
    switches = literal_assignment("TTS010_SWITCHES")
    converted = {
        "model_version": "TTS_0.10",
        "switches": switches,
        "speaker_config": {"n_speakers": 0, "gin_channels": 0, "speaker_to_id": {}},
        "step": int(checkpoint["step"]),
        "optimizer_updates": None,
        "counters": {},
        "model": model.state_dict(),
        "discriminator": discriminator.state_dict(),
        "source": {
            "trainer": "benlu final-vits.ipynb (docs/records/tts-0.10/benlu-source/)",
            "file": Path(source).name,
            "sha256": hashlib.sha256(Path(source).read_bytes()).hexdigest(),
            "note": "no optimizer or scaler state; not resumable",
        },
    }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(converted, output)
    return output, converted


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source")
    parser.add_argument("output")
    args = parser.parse_args()
    path, result = convert(args.source, args.output)
    print(path, "step", result["step"], "sha256", result["source"]["sha256"])
