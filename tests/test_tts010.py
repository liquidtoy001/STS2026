"""TTS_0.10 checks against benlu's archived source and the tts-v0.02 release.

Synthetic tones and random weights establish equivalence and wiring only, never speech quality.
"""

import ast
import contextlib
import csv
import importlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest

import numpy as np
import torch

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
        self.assertEqual(set(arms), {"R0", "R0b", "E0", "G1", "G2", "G3", "A1", "A2", "L1", "O1", "D1"})
        self.assertEqual(sorted(len(spec["revert"]) for name, spec in arms.items() if name not in ("R0", "R0b", "E0")),
                         [1] * 8)
        self.assertEqual({name for spec in arms.values() for name in spec["revert"]},
                         set(literal_assignment("TTS010_SWITCHES")))
        for arm in arms:
            if arm == "R0b":
                continue  # same switches as R0
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
