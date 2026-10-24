"""TTS_0.10 checks against benlu's archived source and the tts-v0.02 release.

Synthetic tones and random weights establish equivalence and wiring only, never speech quality.
"""

import ast
import contextlib
import csv
import hashlib
import importlib
import json
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock

import numpy as np
import torch
import torchaudio

from tools.build_kaggle_full_train import build
from tools.convert_benlu_checkpoint import convert, convert_generator_state
from tools.notebook_defs import literal_assignment, load_definitions

ROOT = Path(__file__).resolve().parents[1]
BENLU = ROOT / "docs/records/tts-0.10/benlu-source"
BENLU_MODULES = ("blocks", "util", "flow", "modules", "monotonic_align", "mel_process")


@contextlib.contextmanager
def benlu_import_path():
    """Import benlu's flat modules (blocks, util, ...) without leaking them into sys.modules."""
    saved = {name: sys.modules.pop(name) for name in BENLU_MODULES if name in sys.modules}
    sys.path.insert(0, str(BENLU))
    try:
        yield
    finally:
        sys.path.remove(str(BENLU))
        for name in BENLU_MODULES:
            sys.modules.pop(name, None)
        sys.modules.update(saved)


def benlu_namespace():
    """AudioSyn, discriminators, losses and MAS from the archived final-vits sources."""
    with benlu_import_path():
        modules = {name: importlib.import_module(name) for name in ("modules", "flow", "util", "monotonic_align")}
    notebook = json.loads((BENLU / "final-vits.ipynb").read_text(encoding="utf-8"))
    ns = {"torch": torch, "nn": torch.nn, "F": torch.nn.functional, "math": __import__("math"),
          "np": np, "Conv1d": torch.nn.Conv1d, "Conv2d": torch.nn.Conv2d,
          "weight_norm": torch.nn.utils.weight_norm, "monotonic_align": modules["monotonic_align"],
          "ResidualCouplingBlock": modules["flow"].ResidualCouplingBlock}
    ns.update({name: getattr(modules["modules"], name)
               for name in ("TextEncoder", "PosteriorEncoder", "Decoder", "DurationPredictor")})
    ns.update({name: getattr(modules["util"], name)
               for name in ("sequence_mask", "generate_path", "slice_segments")})
    for index in (3, 4, 7, 8, 9):  # symbols, spectrogram, discriminators, AudioSyn, losses
        exec("".join(notebook["cells"][index]["source"]), ns)
    return ns


def tone_batch(ns, frames=(48, 40), texts=("Hello there.", "Hi!")):
    samples = []
    for index, count in enumerate(frames):
        time = torch.arange(count * 256) / 22050
        wav = (0.2 * torch.sin(2 * torch.pi * (220 + 110 * index) * time)).unsqueeze(0)
        samples.append((ns["encode_text"](texts[index]), ns["waveform_to_spectrogram"](wav), wav))
    return ns["collate_tts_batch"](samples)


class TTS010Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)
        cls.ns = load_definitions()
        cls.ns["PAD_ID"] = 0
        cls.ns["symbol_to_id"] = {s: i for i, s in enumerate(
            ["_"] + list(literal_assignment("PUNCTUATION")) + list(literal_assignment("LETTERS")))}
        cls.benlu = benlu_namespace()
        cls.batch = tone_batch(cls.ns)

    def test_symbols_match_benlu(self):
        self.assertEqual(list(self.ns["symbol_to_id"]), self.benlu["symbols"])

    def test_mas_matches_benlu_loop_including_ties(self):
        rng = np.random.default_rng(3)
        for trial in range(40):
            t_x = rng.integers(3, 30, 3)
            t_y = t_x + rng.integers(0, 40, 3)
            values = (rng.integers(-2, 2, (3, t_y.max(), t_x.max())) if trial % 2
                      else rng.standard_normal((3, t_y.max(), t_x.max())) * 5).astype(np.float32)
            mask = np.zeros_like(values)
            for i in range(3):
                mask[i, :t_y[i], :t_x[i]] = 1
            score, mask = torch.from_numpy(values * mask), torch.from_numpy(mask)
            expected = self.benlu["monotonic_align"].maximum_path(score.clone(), mask)
            self.assertTrue(torch.equal(self.ns["maximum_path"](score.clone(), mask), expected), trial)

    def test_spectrogram_and_slaney_mel_match_benlu_and_librosa_definition(self):
        wave = self.batch[4][:, 0]
        self.assertTrue(torch.equal(self.ns["waveform_to_spectrogram"](wave),
                                    self.benlu["spectrogram_torch"](wave, 1024, 22050, 256, 1024)))

        def hz_to_mel(f):  # librosa Slaney scale
            f = np.asarray(f, dtype=np.float64)
            mel = f / (200.0 / 3)
            log_region = f >= 1000.0
            return np.where(log_region, 15.0 + np.log(np.maximum(f, 1e-10) / 1000.0) / (np.log(6.4) / 27.0), mel)

        def mel_to_hz(m):
            m = np.asarray(m, dtype=np.float64)
            return np.where(m >= 15.0, 1000.0 * np.exp((np.log(6.4) / 27.0) * (m - 15.0)), m * 200.0 / 3)

        fft = np.linspace(0, 11025, 513)
        mel_f = mel_to_hz(np.linspace(hz_to_mel(0.0), hz_to_mel(11025.0), 82))
        ramps = np.subtract.outer(mel_f, fft)
        fdiff = np.diff(mel_f)
        weights = np.stack([np.maximum(0, np.minimum(-ramps[i] / fdiff[i], ramps[i + 2] / fdiff[i + 1]))
                            for i in range(80)])
        weights *= (2.0 / (mel_f[2:82] - mel_f[:80]))[:, None]
        torch.testing.assert_close(self.ns["slaney_mel_basis"](), torch.from_numpy(weights).float(),
                                   rtol=1e-4, atol=1e-7)

    def test_tts010_generator_matches_benlu_audiosyn(self):
        torch.manual_seed(11)
        reference = self.benlu["AudioSyn"](76, 192, 192, 768, 513).eval()
        model = self.ns["VITSModel"](vocab_size=76, spec_channels=513).eval()
        model.load_state_dict(convert_generator_state(reference.state_dict()), strict=True)
        x, x_lengths, spec, spec_lengths = self.batch[:4]
        with torch.no_grad():
            torch.manual_seed(5)
            o, l_length, attn, ids_slice, _, _, (_, z_p, m_p, logs_p, m_q, logs_q) = reference(
                x, x_lengths, spec, spec_lengths)
            torch.manual_seed(5)
            out = self.ns["VITSModel"].forward(model, x, x_lengths, spec, spec_lengths)
        for name, expected in (("generated_audio", o), ("attention", attn), ("start_frames", ids_slice),
                               ("z_prior", z_p), ("prior_mean", m_p), ("prior_log_scale", logs_p),
                               ("posterior_mean", m_q), ("posterior_log_scale", logs_q)):
            self.assertTrue(torch.equal(out[name], expected), name)
        torch.testing.assert_close(self.ns["VITSLoss"]._duration_loss(out), torch.sum(l_length))
        with torch.no_grad():
            torch.manual_seed(7)
            expected = reference.infer(x, x_lengths, noise_scale=0.667)[0]
            torch.manual_seed(7)
            result = model.infer(x, x_lengths, noise_scale=0.667)
        self.assertTrue(torch.equal(result["generated_audio"], expected))

    def test_tts010_discriminator_and_adversarial_losses_match_benlu(self):
        torch.manual_seed(2)
        reference = self.benlu["MultiPeriodDiscriminator"]()
        ours = self.ns["MultiPeriodDiscriminator"]()
        ours.load_state_dict(reference.state_dict(), strict=True)
        y, y_hat = torch.randn(2, 1, 8192) * 0.1, torch.randn(2, 1, 8192) * 0.1
        with torch.no_grad():
            expected, result = reference(y, y_hat), ours(y, y_hat)
        for left, right in zip(expected, result):
            for a, b in zip(left, right):
                if isinstance(a, list):
                    self.assertTrue(all(torch.equal(p, q) for p, q in zip(a, b)))
                else:
                    self.assertTrue(torch.equal(a, b))
        real, fake, fmap_r, fmap_g = result
        torch.testing.assert_close(self.ns["discriminator_lsgan_loss"](real, fake)["total"],
                                   self.benlu["discriminator_loss"](real, fake)[0])
        torch.testing.assert_close(self.ns["generator_lsgan_loss"](fake), self.benlu["generator_loss"](fake)[0])
        torch.testing.assert_close(2 * self.ns["feature_matching_loss"](fmap_r, fmap_g),
                                   self.benlu["feature_loss"](fmap_r, fmap_g))

    def test_legacy_switches_match_tts002_release(self):
        source = subprocess.run(["git", "show", "tts-v0.02:notebooks/vits-full-train.ipynb"], cwd=ROOT,
                                check=True, capture_output=True).stdout.decode("utf-8")
        old = load_definitions(source=source)
        old["PAD_ID"] = 0
        sizes = dict(vocab_size=76, spec_channels=513, latent_channels=8, hidden_channels=8, feedforward_channels=16)
        torch.manual_seed(31)
        previous = old["VITSModel"](**sizes).eval()
        torch.manual_seed(31)
        current = self.ns["VITSModel"](**sizes, text_encoder="tts002", posterior_encoder="tts002",
                                       decoder="tts002").eval()
        self.assertEqual(list(previous.state_dict()), list(current.state_dict()))
        for key, value in previous.state_dict().items():
            self.assertTrue(torch.equal(value, current.state_dict()[key]), key)
        outputs = []
        for model in (previous, current):
            torch.manual_seed(9)
            with torch.no_grad():
                outputs.append((model(*self.batch[:4]), model.infer(*self.batch[:2], max_frames=16)))
        for (train_a, infer_a), (train_b, infer_b) in [outputs]:
            for key in train_a:
                self.assertTrue(torch.equal(train_a[key], train_b[key]), key)
            for key in infer_a:
                self.assertTrue(torch.equal(infer_a[key], infer_b[key]), key)
        torch.manual_seed(4)
        old_disc = old["Discriminator"](base_channels=4)
        torch.manual_seed(4)
        new_disc = self.ns["LegacyDiscriminator"](base_channels=4)
        audio = torch.randn(2, 1, 8192) * 0.1
        self.assertTrue(torch.equal(old_disc(audio), new_disc.score(audio)[0]))
        target = torch.randn(2, 1, 8192) * 0.1
        train = outputs[0][0]
        old_loss = old["VITSLoss"](mel_weight=1.0, stft_weight=0.5, duration_weight=1.0, kl_weight=1.0,
                                   wave_weight=10.0, gan_weight=1.0)(train, target, gan_loss=torch.tensor(0.3))
        new_loss = self.ns["VITSLoss"](recipe="tts002", feature_matching=False)(
            train, target, gan_loss=torch.tensor(0.3))
        for key in ("mel", "stft", "wave", "duration", "kl", "gan"):
            self.assertTrue(torch.equal(old_loss[key], new_loss[key]), key)
        torch.testing.assert_close(old_loss["total"], new_loss["total"])

    def make_trainer(self, switches, root, model=None):
        model = model or self.ns["VITSModel"](
            vocab_size=76, spec_channels=513, latent_channels=8, hidden_channels=8, feedforward_channels=16,
            text_encoder=switches["text_encoder"], posterior_encoder=switches["posterior_encoder"],
            decoder=switches["decoder"])
        discriminator = (self.ns["MultiPeriodDiscriminator"]() if switches["discriminator"] == "tts010"
                         else self.ns["LegacyDiscriminator"](base_channels=4))
        preset = literal_assignment("OPTIM_PRESETS")[switches["optim"]]
        return self.ns["VITSTrainer"](
            model, discriminator, self.ns["VITSLoss"](recipe=switches["recon_loss"],
                                                      feature_matching=switches["feature_matching"]),
            [self.batch], learning_rate=preset["learning_rate"], use_amp=preset["amp"],
            max_grad_norm=preset["max_grad_norm"], device="cpu", switches=switches,
            checkpoint_dir=Path(root) / "checkpoints", sample_dir=Path(root) / "samples")

    def arm_switches(self, arm):
        arms = literal_assignment("ARMS")
        new, old = literal_assignment("TTS010_SWITCHES"), literal_assignment("TTS002_SWITCHES")
        return {name: (old if name in arms[arm]["revert"] else new)[name] for name in new}

    def test_every_arm_trains(self):
        arms = literal_assignment("ARMS")
        self.assertEqual(set(arms), {"R0", "R0b", "E0", "G1", "G2", "G3", "A1", "A2", "L1", "O1", "D1", "T011",
                                     "V1", "V2", "V3", "V4"})
        ablation = {name: spec for name, spec in arms.items()
                    if name not in ("R0", "R0b", "E0") and "heldout_extra" not in spec}
        self.assertEqual(sorted(len(spec["revert"]) for spec in ablation.values()), [1] * 8)
        self.assertEqual({name for spec in arms.values() for name in spec["revert"]},
                         set(literal_assignment("TTS010_SWITCHES")))
        for arm in arms:
            if arm == "R0b" or "heldout_extra" in arms[arm]:
                continue  # same switches as R0; T011 and the TTS_0.20 arms are tested below
            with self.subTest(arm=arm), tempfile.TemporaryDirectory() as root:
                switches = self.arm_switches(arm)
                torch.manual_seed(1)
                trainer = self.make_trainer(switches, root)
                before = {k: v.clone() for k, v in trainer.model.state_dict().items()}
                for _ in range(2):
                    _, _, losses = trainer._training_step(self.batch)
                    self.assertTrue(all(torch.isfinite(v) for k, v in losses.items() if k != "amp_scale"), losses)
                self.assertEqual(trainer.optimizer_update_stats["updates"], 2)
                self.assertEqual(trainer.counters, {"nonfinite_skips": 0, "amp_backoffs": 0})
                changed = sum(not torch.equal(v, trainer.model.state_dict()[k]) for k, v in before.items())
                self.assertGreater(changed, len(before) // 2)
                expected_fm = 0.0 if not switches["feature_matching"] else None
                if expected_fm is not None:
                    self.assertEqual(losses["feature_matching"].item(), 0.0)
                else:
                    self.assertGreater(losses["feature_matching"].item(), 0.0)
                if switches["recon_loss"] == "tts010":
                    self.assertEqual(losses["stft"].item(), 0.0)
                else:
                    self.assertGreater(losses["stft"].item(), 0.0)

    def test_tts020_arms_train(self):
        arms, defaults = literal_assignment("ARMS"), literal_assignment("TTS020_OPTIONS")
        for arm in ("V1", "V2", "V3", "V4"):
            with self.subTest(arm=arm), tempfile.TemporaryDirectory() as root:
                options = {**defaults, **arms[arm]["options"]}
                self.assertEqual(set(options), set(defaults))
                self.assertEqual(arms[arm]["revert"], arms["T011"]["revert"])
                switches = {**self.arm_switches(arm), **arms[arm]["options"]}
                table = self.ns["token_table"](options["text_input"])
                torch.manual_seed(1)
                model = self.ns["VITSModel"](
                    vocab_size=len(table), spec_channels=513, latent_channels=8, hidden_channels=8,
                    feedforward_channels=16, duration_predictor=options["duration_predictor"],
                    encoder_window=options["encoder_window"], space_id=table[" "])
                trainer = self.make_trainer(switches, root, model=model)
                for _ in range(2):
                    _, _, losses = trainer._training_step(self.batch)
                    self.assertTrue(all(torch.isfinite(v) for k, v in losses.items() if k != "amp_scale"), losses)
                self.assertEqual(trainer.optimizer_update_stats["updates"], 2)
                model.eval()
                tokens = self.batch[0][:1, :int(self.batch[1][0])]
                audio = model.infer(tokens, self.batch[1][:1], noise_scale=0.667)["generated_audio"]
                self.assertTrue(torch.isfinite(audio).all())

    def test_tts010_arms_keep_their_switches(self):
        arms = literal_assignment("ARMS")
        for arm, spec in arms.items():
            if "options" not in spec:
                self.assertNotIn("options", spec)
        # Default model construction is unchanged: same parameter names and shapes as before the options.
        torch.manual_seed(0)
        default = self.ns["VITSModel"](vocab_size=76, spec_channels=513)
        explicit = self.ns["VITSModel"](vocab_size=76, spec_channels=513, duration_predictor="deterministic",
                                        encoder_window=None, space_id=None)
        self.assertEqual({k: v.shape for k, v in default.state_dict().items()},
                         {k: v.shape for k, v in explicit.state_dict().items()})
        self.assertEqual(self.ns["token_table"]("chars"), self.ns["symbol_to_id"])

    def test_windowed_encoder_is_exact_for_streaming(self):
        table = self.ns["token_table"]("chars")
        torch.manual_seed(0)
        encoder = self.ns["TextEncoder"](vocab_size=len(table), window=(12, 2), space_id=table[" "]).eval()
        words = "the quick brown fox jumps over the lazy dog again and again".split()

        def prior(count, lookahead):
            encoder.lookahead_words = lookahead
            tokens = self.ns["encode_text"](" ".join(words[:count]), table).unsqueeze(0)
            return encoder(tokens, torch.tensor([tokens.size(1)]))[1]

        full = prior(len(words), 2)
        for count in (1, 3, 6):
            keep = len(" ".join(words[:count]))
            torch.testing.assert_close(prior(count + 2, 2)[..., :keep], full[..., :keep], atol=1e-5, rtol=0)
            self.assertGreater((prior(count + 1, 2)[..., :keep] - full[..., :keep]).abs().max().item(), 1e-3)

    def test_stochastic_duration_spline_is_invertible(self):
        torch.manual_seed(0)
        x = torch.randn(3, 1, 9) * 3
        widths, heights, slopes = torch.randn(3, 1, 9, 10), torch.randn(3, 1, 9, 10), torch.randn(3, 1, 9, 9)
        transform = self.ns["piecewise_rational_quadratic_transform"]
        y, forward_logdet = transform(x, widths, heights, slopes)
        x_back, inverse_logdet = transform(y, widths, heights, slopes, inverse=True)
        torch.testing.assert_close(x_back, x, atol=1e-4, rtol=0)
        torch.testing.assert_close(forward_logdet + inverse_logdet, torch.zeros_like(x), atol=1e-4, rtol=0)

    def test_punctuation_crops_end_at_punctuation(self):
        table = self.ns["token_table"]("chars")
        tokens = self.ns["encode_text"]("one two, three four five. six seven eight nine ten", table)
        boundary_ids = {table[c] for c in ",.?!;:"}
        rng = random.Random(3)
        for _ in range(50):
            start, end, frame_start, frame_end = self.ns["word_crop_bounds"](
                tokens, [2] * tokens.numel(), table[" "], rng, 1, 8, 0.5, 4, boundary_ids)
            self.assertTrue(int(tokens[end - 1]) in boundary_ids or end == tokens.numel(), (start, end))
            self.assertEqual((frame_start, frame_end), (2 * start, 2 * end))

    def test_phoneme_cache_matches_the_corpus_words(self):
        sys.path.insert(0, str(ROOT))
        from tools.phonemize import ARPABET, PHONEME_CHAR, read_cache
        phonemes, header = read_cache(ROOT / "sts_eval" / "phonemes" / "ljspeech_g2p_en.tsv")
        self.assertEqual(header["metadata_sha256"], "852bce5c79e5184ba5bf2ea98aad4bcf49a5d787def5fd1f04fa2ab8aa6dc535")
        self.assertEqual(len(phonemes), 13100)
        table = self.ns["token_table"]("phonemes")
        self.assertEqual([table[PHONEME_CHAR[p]] for p in ARPABET], list(range(8, 8 + len(ARPABET))))
        used = set("".join(phonemes.values()))
        self.assertLessEqual(used, set(table))

    def test_t011_holds_out_the_long_sentences(self):
        arms = literal_assignment("ARMS")
        self.assertEqual(arms["T011"]["revert"], arms["D1"]["revert"])
        self.assertEqual(arms["T011"]["seed"], arms["D1"]["seed"])
        splits = ROOT / "sts_eval" / "splits"
        base, long = splits / "ljspeech_heldout_100.txt", splits / arms["T011"]["heldout_extra"][0]
        ns = self.ns
        short_ids, long_ids = ns["load_id_list"](base), ns["load_id_list"](long)
        self.assertEqual((len(short_ids), len(long_ids)), (100, 100))
        self.assertFalse(set(short_ids) & set(long_ids))
        combined = ns["heldout_ids"](base, [long])
        self.assertEqual(combined, short_ids + long_ids)  # base first: the held-out probe stays the same
        self.assertEqual(ns["heldout_ids"](base), short_ids)
        for arm, spec in arms.items():
            if arm != "T011" and "options" not in spec:  # TTS_0.10 arms keep the 100-sentence hold-out
                self.assertNotIn("heldout_extra", spec)

    def test_checkpoint_round_trip_and_switch_mismatch(self):
        with tempfile.TemporaryDirectory() as root:
            switches = self.arm_switches("A1")
            trainer = self.make_trainer(switches, root)
            trainer._training_step(self.batch)
            trainer.step = 1
            trainer.save_checkpoint()
            path = trainer.checkpoint_dir / "vits_step_1.pt"
            saved = {k: v.clone() for k, v in trainer.model.state_dict().items()}
            stored = torch.load(path, map_location="cpu", weights_only=True)
            self.assertEqual(stored["switches"], switches)
            self.assertIn("scaler", stored)
            fresh = self.make_trainer(switches, Path(root) / "fresh")
            fresh.load_checkpoint(path)
            self.assertTrue(all(torch.equal(v, fresh.model.state_dict()[k]) for k, v in saved.items()))
            self.assertEqual((fresh.step, fresh.optimizer_update_stats["updates"]), (1, 1))
            with self.assertRaisesRegex(ValueError, "switches"):
                self.make_trainer({**switches, "feature_matching": False}, Path(root) / "x").load_checkpoint(path)
            fresh.save_samples(self.batch)
            self.assertEqual(len(list(fresh.sample_dir.glob("*.wav"))), 2)

    def test_init_from_checkpoint_starts_a_new_run(self):
        with tempfile.TemporaryDirectory() as root:
            switches = self.arm_switches("A1")
            source = self.make_trainer(switches, Path(root) / "source")
            source._training_step(self.batch)
            source.step = 7
            source.save_checkpoint()
            path = source.checkpoint_dir / "vits_step_7.pt"
            stored = torch.load(path, map_location="cpu", weights_only=True)
            self.assertNotIn("init_from", stored)  # ordinary runs keep the checkpoint format

            torch.manual_seed(99)  # different initial weights, replaced by the checkpoint's
            tuned = self.make_trainer(switches, Path(root) / "tuned")
            record = tuned.init_from_checkpoint(path)
            self.assertEqual((tuned.step, tuned.optimizer_update_stats["updates"]), (0, 0))
            self.assertEqual(tuned.counters, {"nonfinite_skips": 0, "amp_backoffs": 0})
            self.assertEqual((record["step"], record["optimizer"]), (7, "carried"))
            self.assertEqual(record["sha256"], hashlib.sha256(path.read_bytes()).hexdigest())
            for name in ("model", "discriminator"):
                for key, value in stored[name].items():
                    state = (tuned.base_model if name == "model" else tuned.discriminator).state_dict()
                    self.assertTrue(torch.equal(state[key], value), key)
            moments = tuned.generator_optimizer.state_dict()["state"]
            self.assertTrue(all(torch.equal(moments[k]["exp_avg"], v["exp_avg"])
                                for k, v in stored["optimizer"]["state"].items()))

            fresh = self.make_trainer(switches, Path(root) / "fresh")
            self.assertEqual(fresh.init_from_checkpoint(path, carry_optimizer=False)["optimizer"], "fresh")
            self.assertEqual(fresh.generator_optimizer.state_dict()["state"], {})
            with self.assertRaisesRegex(ValueError, "switches.*initialisation"):
                self.make_trainer({**switches, "feature_matching": False},
                                  Path(root) / "x").init_from_checkpoint(path)
            weights_only = Path(root) / "weights_only.pt"
            torch.save({k: v for k, v in stored.items() if "optimizer" not in k and k != "scaler"}, weights_only)
            with self.assertRaisesRegex(ValueError, "no optimizer state"):
                self.make_trainer(switches, Path(root) / "y").init_from_checkpoint(weights_only)

            # The new run counts its own steps and records its source in every checkpoint; resume keeps it.
            tuned._training_step(self.batch)
            tuned.step = 1
            tuned.save_checkpoint()
            resumed = self.make_trainer(switches, Path(root) / "resumed")
            resumed.load_checkpoint(tuned.checkpoint_dir / "vits_step_1.pt")
            self.assertEqual(resumed.init_from, record)
            self.assertEqual(resumed.step, 1)

    def test_word_crop_keeps_whole_words_and_their_audio(self):
        ns, space = self.ns, self.ns["symbol_to_id"][" "]
        text = "Ab cd, efg h ij klm no."
        tokens = ns["encode_text"](text)
        rng = np.random.default_rng(0)
        durations = rng.integers(2, 9, tokens.numel()).tolist()
        bounds = np.concatenate([[0], np.cumsum(durations)])
        self.assertEqual(ns["word_spans"](tokens, space), [(0, 2), (3, 6), (7, 10), (11, 12), (13, 15),
                                                           (16, 19), (20, 23)])
        self.assertIsNone(ns["word_crop_bounds"](ns["encode_text"]("Hello."), [3] * 6, space, random.Random(0)))
        words = text.split()
        prefixes = 0
        for seed in range(300):
            min_frames = (0, 10, 40)[seed % 3]
            t0, t1, f0, f1 = ns["word_crop_bounds"](tokens, durations, space, random.Random(seed), 1, 3, 0.3,
                                                    min_frames)
            kept = text[t0:t1]
            first = len(text[:t0].split())
            self.assertEqual(kept.split(), words[first:first + len(kept.split())], seed)  # whole words
            self.assertFalse(kept.startswith(" ") or kept.endswith(" "), seed)
            self.assertEqual((f0, f1), (bounds[t0], bounds[t1]), seed)
            self.assertTrue(f1 - f0 >= min_frames or len(kept.split()) == len(words), seed)
            if min_frames == 0:
                self.assertLessEqual(len(kept.split()), 3, seed)
            prefixes += t0 == 0
        self.assertTrue(80 < prefixes < 200, prefixes)  # prefix share 0.3 plus spans that start at word 0
        self.assertTrue(all(ns["word_crop_bounds"](tokens, durations, space, random.Random(s), 1, 3, 1.0)[0] == 0
                            for s in range(20)))
        self.assertEqual(ns["word_crop_bounds"](tokens, durations, space, random.Random(1), 1, 1, 0.0, 10**6)[:2],
                         (0, tokens.numel()))

    def test_chunk_dataset_off_matches_release_and_on_crops_text_and_audio_together(self):
        ns, space = self.ns, self.ns["symbol_to_id"][" "]
        text = "Ab cd, efg h ij klm no."
        tokens = ns["encode_text"](text)
        durations = np.random.default_rng(1).integers(3, 12, tokens.numel()).tolist()
        bounds = np.concatenate([[0], np.cumsum(durations)]) * 256
        frequency = [200.0 + 150.0 * index for index in range(tokens.numel())]
        wave = torch.zeros(1, int(bounds[-1]) + 100)  # frames = (samples - 256) // 256 + 1 = sum(durations)
        for index, (start, end) in enumerate(zip(bounds[:-1], bounds[1:])):
            time = torch.arange(int(end - start)) / 22050
            wave[0, int(start):int(end)] = 0.3 * torch.sin(2 * torch.pi * frequency[index] * time)
        release = load_definitions(source=subprocess.run(
            ["git", "show", "tts-v0.10:notebooks/vits-full-train.ipynb"], cwd=ROOT, check=True,
            capture_output=True).stdout.decode("utf-8"))
        release["symbol_to_id"] = ns["symbol_to_id"]
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            (root / "wavs").mkdir()
            torchaudio.save(str(root / "wavs" / "A.wav"), wave, 22050)
            torchaudio.save(str(root / "wavs" / "B.wav"), wave[:, :30000], 22050)
            metadata = __import__("pandas").DataFrame({"file_name": ["A", "B"],
                                                       "normalized_transcription": [text, "Hi there!"]})
            # Off: identical to the tts-v0.10 release and no random numbers drawn.
            state = random.getstate()
            for index in range(2):
                for a, b in zip(ns["LJSpeechDataset"](metadata, root / "wavs")[index],
                                release["LJSpeechDataset"](metadata, root / "wavs")[index]):
                    self.assertTrue(torch.equal(a, b))
            self.assertEqual(random.getstate(), state)

            # The durations file format written by tools/chunk_alignments.py is the one the notebook reads.
            from tools.chunk_alignments import FORMAT
            cache = root / "durations.json"
            cache.write_text(json.dumps({"format": FORMAT, "hop_length": 256, "durations": {"A": durations}}))
            loaded, provenance = ns["load_chunk_durations"](cache, 256)
            self.assertEqual((loaded, provenance["clips"]), ({"A": durations}, 1))
            settings = {"prob": 1.0, "min_words": 1, "max_words": 2, "prefix_share": 0.5, "min_frames": 8,
                        "space_id": space, "durations": provenance}
            with self.assertRaisesRegex(ValueError, "lack 1 clips"):
                ns["LJSpeechDataset"](metadata, root / "wavs", chunk_training=settings, chunk_durations=loaded)
            with self.assertRaisesRegex(ValueError, "do not match its text"):
                ns["LJSpeechDataset"](metadata.iloc[:1], root / "wavs", chunk_training=settings,
                                      chunk_durations={"A": durations[:-1]})
            dataset = ns["LJSpeechDataset"](metadata.iloc[:1], root / "wavs", chunk_training=settings,
                                            chunk_durations=loaded)
            symbols = {i: s for s, i in ns["symbol_to_id"].items()}
            seen = set()
            random.seed(5)
            for _ in range(40):
                cropped, spectrogram, audio = dataset[0]
                kept = "".join(symbols[int(i)] for i in cropped)
                self.assertIn(kept, text)
                self.assertTrue(1 <= len(kept.split()) <= 2 or audio.size(-1) >= 8 * 256, kept)
                start = text.index(kept)
                self.assertTrue(start == 0 or text[start - 1] == " ")
                self.assertEqual(spectrogram.size(-1), sum(durations[start:start + len(kept)]))
                self.assertTrue(torch.equal(spectrogram, ns["waveform_to_spectrogram"](audio)))
                # Every kept character's frames carry that character's tone.
                offset = 0
                for index in range(start, start + len(kept)):
                    segment = audio[0, offset * 256:(offset + durations[index]) * 256]
                    offset += durations[index]
                    peak = torch.fft.rfft(segment).abs().argmax().item() * 22050 / segment.numel()
                    self.assertLess(abs(peak - frequency[index]), 22050 / segment.numel() + 1, (kept, index))
                seen.add(kept)
            self.assertGreater(len(seen), 5)
            bad = ns["LJSpeechDataset"](metadata.iloc[:1], root / "wavs", chunk_training=settings,
                                        chunk_durations={"A": [durations[0] + 1] + durations[1:]})
            with self.assertRaisesRegex(ValueError, "frames"):
                bad[0]

    def test_chunk_alignment_durations_equal_training_mas(self):
        from tools.chunk_alignments import alignment_durations
        torch.manual_seed(3)
        model = self.ns["VITSModel"](vocab_size=76, spec_channels=513, latent_channels=8, hidden_channels=8,
                                     feedforward_channels=16).eval()
        x, x_lengths, spec, spec_lengths = self.batch[:4]
        durations = alignment_durations(self.ns, model, self.batch)
        with torch.no_grad(), mock.patch("torch.randn_like", side_effect=torch.zeros_like):
            attention = model(x, x_lengths, spec, spec_lengths)["attention"]  # posterior mean, as the tool
        expected = attention.sum(dim=2)[:, 0].long()
        for index, values in enumerate(durations):
            self.assertEqual(values, expected[index, :int(x_lengths[index])].tolist())
            self.assertEqual(sum(values), int(spec_lengths[index]))
            self.assertGreaterEqual(min(values), 1)

    def test_training_step_without_chunks_matches_release(self):
        release = load_definitions(source=subprocess.run(
            ["git", "show", "tts-v0.10:notebooks/vits-full-train.ipynb"], cwd=ROOT, check=True,
            capture_output=True).stdout.decode("utf-8"))
        release["PAD_ID"] = 0
        switches = self.arm_switches("A1")
        results = []
        with tempfile.TemporaryDirectory() as root:
            for index, ns in enumerate((release, self.ns)):
                torch.manual_seed(21)
                model = ns["VITSModel"](vocab_size=76, spec_channels=513, latent_channels=8, hidden_channels=8,
                                        feedforward_channels=16, text_encoder="tts010", posterior_encoder="tts010",
                                        decoder="tts010")
                trainer = ns["VITSTrainer"](
                    model, ns["LegacyDiscriminator"](base_channels=4), ns["VITSLoss"](), [self.batch],
                    learning_rate=2e-4, use_amp=False, device="cpu", switches=switches,
                    checkpoint_dir=Path(root) / str(index) / "c", sample_dir=Path(root) / str(index) / "s")
                torch.manual_seed(8)
                _, _, losses = trainer._training_step(self.batch)
                results.append((losses, trainer.model.state_dict(), trainer.discriminator.state_dict()))
        (old_losses, old_model, old_disc), (new_losses, new_model, new_disc) = results
        for key, value in old_losses.items():
            self.assertTrue(torch.equal(value, new_losses[key]) or (value.isnan() & new_losses[key].isnan()).all(),
                            key)
        for old, new in ((old_model, new_model), (old_disc, new_disc)):
            self.assertTrue(all(torch.equal(value, new[key]) for key, value in old.items()))

    def test_nonfinite_steps_are_skipped_then_stop(self):
        with tempfile.TemporaryDirectory() as root:
            trainer = self.make_trainer(self.arm_switches("E0"), root)
            original = trainer.loss_function.forward
            trainer.loss_function.forward = lambda *a, **k: {**original(*a, **k), "total": torch.tensor(float("nan"))}
            before = {k: v.clone() for k, v in trainer.model.state_dict().items()}
            trainer._training_step(self.batch)
            self.assertEqual(trainer.counters["nonfinite_skips"], 1)
            self.assertTrue(all(torch.equal(v, trainer.model.state_dict()[k]) for k, v in before.items()))
            trainer.MAX_CONSECUTIVE_NONFINITE = 2
            with self.assertRaises(FloatingPointError):
                trainer._training_step(self.batch)

    def test_metadata_parsing_filter_and_heldout_split(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "metadata.csv"
            path.write_text('A|x|He said "yes" today.\nB|y|' + "a" * 101 + '\nC|z|NA\n', encoding="utf-8")
            frame, mapping = self.ns["load_training_metadata"](path)
            self.assertEqual(mapping, {})
            self.assertEqual(frame["normalized_transcription"].tolist()[0], 'He said "yes" today.')
            self.assertEqual(frame["normalized_transcription"].tolist()[2], "NA")
            self.assertEqual(self.ns["filter_text_length"](frame, 100)["file_name"].tolist(), ["A", "C"])
            self.assertEqual(len(self.ns["filter_text_length"](frame, 0)), 3)
        split = ROOT / "sts_eval/splits/ljspeech_heldout_100.txt"
        ids = self.ns["load_id_list"](split)
        self.assertEqual(len(ids), 100)
        self.assertEqual(ids, sorted(set(ids)))
        self.assertFalse(set(ids) & {f"LJ001-{i:04d}" for i in range(1, 9)})
        metadata = Path(ROOT.parents[2] if ROOT.parent.name == "worktrees" else ROOT) / "data/LJSpeech-1.1/metadata.csv"
        if metadata.is_file():
            from tools.make_heldout_split import read_metadata
            rows = {row[0]: row[2] for row in read_metadata(metadata)}
            self.assertTrue(all(1 <= len(rows[i]) <= 100 for i in ids))

    def test_asr_alignment_counts_and_corpus_rate(self):
        from tools.asr_wer import align_counts, corpus_rates
        # The worked example of the team's ASR evaluation note: S=1, D=0, I=1, WER 50%.
        self.assertEqual(align_counts("the quick brown fox".split(), "the quick crown fox now".split()), (1, 0, 1))
        self.assertEqual(align_counts(list("abc"), list("ac")), (0, 1, 0))
        self.assertEqual(align_counts([], list("ab")), (0, 0, 2))
        rows = [{"word_n": 4, "word_s": 1, "word_d": 0, "word_i": 1},
                {"word_n": 6, "word_s": 0, "word_d": 0, "word_i": 0}]
        self.assertAlmostEqual(corpus_rates(rows, "word")["rate"], 0.2)  # corpus rate, not mean of 0.5 and 0

    def test_evaluate_runs_discovery_labels_and_noise_floor(self):
        from tools import evaluate_runs as er
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            values = {"R0": (0.50, 0.10), "R0b": (0.52, 0.12), "G1": (0.60, 0.30), "D1": (0.51, 0.11)}
            for arm, (mel, wer) in values.items():
                run = root / "drive" / arm / f"tts010_{arm}_20261012T000000Z"
                (run / "checkpoints").mkdir(parents=True)
                (run / "checkpoints" / "vits_step_40000.pt").write_bytes(b"")
                (run / "run_config.json").write_text(json.dumps({"model_version": "TTS_0.10", "arm": arm}))
                (run / "status.json").write_text(json.dumps({"status": "completed", "step": 40000}))
                label = root / "eval" / f"{arm}_40k"
                label.mkdir(parents=True)
                (label / "metrics.json").write_text(json.dumps({
                    "checkpoint": str(run / "checkpoints" / "vits_step_40000.pt"), "checkpoint_step": 40000,
                    "switches": {"optim": "tts010"},
                    "heldout_full": {"log_mel_l1": mel, "stft_spectral_convergence": 0.5, "count": 100},
                    "heldout_tts": {"duration_ratio_mean": 1.0, "rms_ratio_mean": 0.9}}))
                channel = {"wer": {"rate": wer}, "cer": {"rate": wer / 2}, "failures": 0}
                (label / "asr.json").write_text(json.dumps({"asr": {"model": "medium"}, "summary": {
                    "synthesized": channel, "natural": {"wer": {"rate": 0.03}}, "delta_wer": wer - 0.03}}))
            unfinished = root / "drive" / "A1" / "tts010_A1_x"
            (unfinished / "checkpoints").mkdir(parents=True)
            (unfinished / "run_config.json").write_text(json.dumps({"model_version": "TTS_0.10", "arm": "A1"}))
            (unfinished / "status.json").write_text(json.dumps({"status": "stopped_time_budget", "step": 20000}))
            runs = er.find_runs(root / "drive", 40000)
            self.assertEqual(sorted(r["arm"] for r in runs), ["D1", "G1", "R0", "R0b"])
            self.assertEqual(er.label_for(runs[0], root / "eval", 40000), f"{runs[0]['arm']}_40k")
            other = dict(runs[0], checkpoint=root / "elsewhere.pt", run=root / "tts010_X_20261013T000000Z")
            self.assertEqual(er.label_for(other, root / "eval", 40000), f"{runs[0]['arm']}_40k_20261013T000000Z")
            rows, has_r0, has_r0b = er.build_table(root / "eval")
            self.assertTrue(has_r0 and has_r0b)
            self.assertEqual([r["arm"] for r in rows], ["R0", "R0b", "G1", "D1"])
            by_arm = {r["arm"]: r for r in rows}
            self.assertAlmostEqual(by_arm["G1"]["wer_vs_R0"], 0.20)
            self.assertTrue(by_arm["G1"]["wer_beyond_noise"])      # 0.20 > |0.12 - 0.10|
            self.assertFalse(by_arm["D1"]["wer_beyond_noise"])     # 0.01 <= 0.02
            self.assertNotIn("wer_beyond_noise", by_arm["R0b"])
            summary = er.write_outputs(rows, has_r0, has_r0b, root / "eval").read_text(encoding="utf-8")
            self.assertIn("+20.0 pt*", summary)

    def test_converter_round_trip(self):
        torch.manual_seed(13)
        generator = self.benlu["AudioSyn"](76, 192, 192, 768, 513)
        discriminator = self.benlu["MultiPeriodDiscriminator"]()
        with tempfile.TemporaryDirectory() as root:
            source = Path(root) / "weight_tts_epoch1.pt"
            torch.save({"step": 8000, "g": generator.state_dict(), "d": discriminator.state_dict()}, source)
            _, converted = convert(source, Path(root) / "converted.pt")
        self.assertEqual(converted["step"], 8000)
        self.assertEqual(converted["switches"], literal_assignment("TTS010_SWITCHES"))
        expected = convert_generator_state(generator.state_dict())
        self.assertTrue(all(torch.equal(expected[k], v) for k, v in converted["model"].items()))

    def test_kaggle_exports_one_notebook_per_arm(self):
        outputs = build()
        self.assertEqual(sorted(p.stem for p in outputs),
                         sorted(f"tts010_{arm}" for arm in literal_assignment("ARMS")))
        for path in outputs:
            notebook = json.loads(path.read_text(encoding="utf-8"))
            arm = path.stem.split("_", 1)[1]
            for cell in notebook["cells"]:
                if cell["cell_type"] == "code":
                    ast.parse(cell["source"])
                    self.assertEqual(cell["outputs"], [])
            tree = ast.parse(notebook["cells"][1]["source"])
            assignment = next(n for n in tree.body if isinstance(n, ast.Assign)
                              and isinstance(n.targets[0], ast.Name) and n.targets[0].id == "source_snapshots")
            snapshots = ast.literal_eval(assignment.value)
            self.assertEqual(snapshots["sts_eval/splits/ljspeech_heldout_100.txt"],
                             (ROOT / "sts_eval/splits/ljspeech_heldout_100.txt").read_text(encoding="utf-8"))
            exported = json.loads(snapshots["notebooks/vits-full-train.ipynb"])
            arm_lines = [line for cell in notebook["cells"] for line in cell["source"].splitlines()
                         if line.startswith("ARM = os.environ.get(")]
            self.assertEqual(arm_lines, [f'ARM = os.environ.get("STS_ARM", "{arm}")'])
            self.assertEqual(len(exported["cells"]), len(notebook["cells"]) - 2)


if __name__ == "__main__":
    unittest.main()
