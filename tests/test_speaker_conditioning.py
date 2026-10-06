"""Exercise real notebook/trainer code. Test tones do not establish speech quality."""

import ast
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

import torch
import torchaudio

from sts_eval import ReconstructionEvaluator
from tools.build_kaggle_full_train import build


ROOT = Path(__file__).resolve().parents[1]


def definitions(source=None):
    if source is None:
        source = (ROOT / "notebooks/vits-full-train.ipynb").read_text(encoding="utf-8")
    namespace = {"json": json}
    for index, cell in enumerate(json.loads(source)["cells"]):
        if cell["cell_type"] != "code":
            continue
        tree = ast.parse("".join(cell["source"]))
        compile(tree, f"cell_{index}", "exec")
        nodes = tree.body if index == 1 else [
            node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))]
        exec(compile(ast.Module(body=nodes, type_ignores=[]), f"cell_{index}", "exec"), namespace)
    return namespace


class SpeakerConditioningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.ns = definitions()

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        torch.manual_seed(19)
        self.ns["PAD_ID"] = 0
        samples = []
        for speaker, frames in enumerate((36, 32)):
            time = torch.arange(frames * 256) / 22050
            wav = (0.2 * torch.sin(2 * torch.pi * (220 + speaker * 110) * time)).unsqueeze(0)
            padded = torch.nn.functional.pad(wav, (384, 384), mode="reflect")
            spec = torch.stft(padded, 1024, 256, 1024, torch.hann_window(1024),
                              center=False, return_complex=True).abs().squeeze(0)
            samples.append((torch.tensor([1, 2, 3]), spec, wav, speaker))
        self.samples = samples
        self.batch = self.ns["collate_tts_batch"](samples)
        self.model = self.make_model()

    def make_model(self, conditioned=True, namespace=None):
        return (namespace or self.ns)["VITSModel"](
            vocab_size=10, spec_channels=513, latent_channels=8, hidden_channels=8,
            feedforward_channels=16, **({"n_speakers": 2, "gin_channels": 4} if conditioned else {}))

    def trainer(self, gan_weight=0.0, mapping=None, name="run"):
        return self.ns["VITSTrainer"](
            self.model, self.ns["Discriminator"](base_channels=4),
            self.ns["VITSLoss"](gan_weight=gan_weight), [self.batch], device="cpu",
            speaker_to_id={"alice": 0, "bob": 1} if mapping is None else mapping,
            checkpoint_dir=self.root / name / "checkpoints", sample_dir=self.root / name / "samples")

    def test_manifest_dataset_collation_and_mapping(self):
        manifest = self.root / "speakers.csv"
        manifest.write_text("file_name|normalized_transcription|speaker_name\nb|hello|bob\na|hi|alice\n",
                            encoding="utf-8")
        frame, mapping = self.ns["load_training_metadata"](manifest, True)
        self.assertEqual(mapping, {"alice": 0, "bob": 1})
        self.assertEqual(frame.speaker_id.tolist(), [1, 0])
        notebook = json.loads((ROOT / "notebooks/vits-full-train.ipynb").read_text(encoding="utf-8"))
        exec("".join(notebook["cells"][7]["source"]), self.ns)
        self.ns.update(SAMPLE_RATE=22050, N_FFT=1024, HOP_LENGTH=256, WIN_LENGTH=1024,
                       hann_window=torch.hann_window(1024))
        torchaudio.save(str(self.root / "b.wav"), self.samples[0][2], 22050)
        item = self.ns["LJSpeechDataset"](frame, self.root)[0]
        self.assertEqual(item[3], 1)
        self.assertEqual(item[1].shape, self.samples[0][1].shape)
        self.assertTrue(torch.equal(item[0], self.ns["encode_text"]("hello")))
        self.assertEqual(self.batch[6].tolist(), [0, 1])
        for i, item in enumerate(self.samples):
            self.assertTrue(torch.equal(self.batch[4][i, :, :item[2].shape[-1]], item[2]))
        self.assertEqual(len(self.ns["collate_tts_batch"]([s[:3] for s in self.samples])), 6)
        with self.assertRaises(ValueError):
            self.ns["collate_tts_batch"]([self.samples[0], self.samples[1][:3]])
        for content in ("file_name|normalized_transcription|speaker_name\na|hi|\n",
                        "file_name|normalized_transcription|speaker_name\na|hi|alice\na|hello|bob\n"):
            manifest.write_text(content, encoding="utf-8")
            with self.assertRaises(ValueError):
                self.ns["load_training_metadata"](manifest, True)

    def test_reject_invalid_conditions(self):
        for ids in (None, torch.tensor([0]), torch.tensor([[0], [1]]), torch.tensor([0., 1.]),
                    torch.tensor([-1, 0]), torch.tensor([0, 2])):
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                self.model.infer(self.batch[0], self.batch[1], speaker_ids=ids, max_frames=8)
        with self.assertRaises(ValueError):
            self.make_model(False).speaker_condition(self.batch[6], 2)
        with self.assertRaises(ValueError):
            self.trainer(mapping={"alice": 1, "bob": 1})
        with self.assertRaises(ValueError):
            self.trainer()._training_step(self.batch[:6], 1.0)

    def test_actual_training_updates_both_speakers_with_and_without_gan(self):
        for weight in (0.0, 1.0):
            with self.subTest(gan_weight=weight):
                self.model = self.make_model()
                trainer = self.trainer(weight, name=f"gan{weight}")
                before = self.model.speaker_embedding.weight.detach().clone()
                _, _, losses = trainer._training_step(self.batch, 1.0)
                self.assertTrue(all(torch.isfinite(value).all() for value in losses.values()))
                gradient = self.model.speaker_embedding.weight.grad
                self.assertTrue(torch.isfinite(gradient).all())
                self.assertTrue((gradient.abs().sum(dim=1) > 0).all())
                self.assertTrue(((before - self.model.speaker_embedding.weight).abs().sum(dim=1) > 0).all())
                for module in (self.model.posterior_encoder, self.model.decoder,
                               self.model.duration_predictor):
                    self.assertGreater(module.condition_projection.weight.grad.abs().sum().item(), 0)
                self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in self.model.flow.parameters()))
                # The zero-initialized coupling output gates its condition gradients
                # on the first update; the second update must reach those projections.
                trainer._training_step(self.batch, 1.0)
                for layer in self.model.flow.coupling_layers:
                    projection = layer.coupling_network.condition_projection
                    self.assertGreater(projection.weight.grad.abs().sum().item(), 0)

    def test_same_text_changes_with_speaker_and_flow_is_invertible(self):
        self.model.eval()
        with torch.no_grad():
            first = self.model.infer(self.batch[0], self.batch[1], noise_scale=0,
                                    max_frames=16, speaker_ids=torch.tensor([0, 0]))
            second = self.model.infer(self.batch[0], self.batch[1], noise_scale=0,
                                     max_frames=16, speaker_ids=torch.tensor([1, 1]))
            self.assertFalse(torch.equal(first["generated_audio"], second["generated_audio"]))
            self.assertTrue(torch.isfinite(second["generated_audio"]).all())
            # Coupling starts at identity: perturb its output projections to exercise g.
            for layer in self.model.flow.coupling_layers:
                torch.nn.init.normal_(layer.coupling_network.output_projection.weight, std=0.01)
            z = torch.randn(2, 8, 12)
            mask = torch.ones(2, 1, 12)
            g = self.model.speaker_condition(self.batch[6], 2)
            flowed = self.model.flow(z, mask, g=g)
            self.assertFalse(torch.equal(flowed, self.model.flow(z, mask, g=g.flip(0))))
            recovered = self.model.flow(flowed, mask, g=g, reverse=True)
            torch.testing.assert_close(recovered, z, rtol=1e-5, atol=1e-6)

    def test_checkpoint_round_trip_and_mapping_mismatch(self):
        trainer = self.trainer()
        trainer._training_step(self.batch, 1.0)
        trainer.step = 1
        trainer.save_checkpoint()
        saved = {key: value.clone() for key, value in self.model.state_dict().items()}
        path = trainer.checkpoint_dir / "vits_step_1.pt"
        with torch.no_grad():
            self.model.speaker_embedding.weight.add_(10)
        trainer.load_checkpoint(path)
        self.assertTrue(all(torch.equal(value, self.model.state_dict()[key]) for key, value in saved.items()))
        self.assertEqual(trainer.step, 1)
        wrong = self.trainer(mapping={"alice": 1, "bob": 0}, name="wrong")
        with self.assertRaisesRegex(ValueError, "mapping"):
            wrong.load_checkpoint(path)
        trainer.save_samples(self.batch)
        self.assertEqual(len(list(trainer.sample_dir.glob("*.wav"))), 2)

    def test_conditioned_evaluator_preserves_rng_and_rejects_missing_ids(self):
        evaluator = ReconstructionEvaluator(self.batch, ["tone-a", "tone-b"], self.root / "eval")
        state = torch.random.get_rng_state().clone()
        first = evaluator.evaluate(self.model, 0)
        self.assertTrue(torch.equal(state, torch.random.get_rng_state()))
        second = evaluator.evaluate(self.model, 1)
        self.assertEqual([{k: v for k, v in row.items() if k != "step"} for row in first],
                         [{k: v for k, v in row.items() if k != "step"} for row in second])
        self.assertEqual(evaluator.manifest["speaker_ids"], [0, 1])
        missing = ReconstructionEvaluator(self.batch[:6], ["a", "b"], self.root / "missing")
        with self.assertRaises(ValueError):
            missing.evaluate(self.model, 0)
        swapped = (*self.batch[:6], self.batch[6].flip(0))
        other = ReconstructionEvaluator(swapped, ["tone-a", "tone-b"], self.root / "swap")
        self.assertNotEqual(first, other.evaluate(self.model, 0))

    def test_disabled_path_exactly_matches_previous_release(self):
        # A real historical implementation, not a rewritten expected-output mock.
        source = subprocess.run(["git", "show", "tts-v0.01:notebooks/vits-full-train.ipynb"],
                                cwd=ROOT, check=True, capture_output=True).stdout.decode("utf-8")
        old_ns = definitions(source)
        torch.manual_seed(51)
        previous = self.make_model(False, old_ns).eval()
        torch.manual_seed(51)
        current = self.make_model(False).eval()
        self.assertEqual(set(previous.state_dict()), set(current.state_dict()))
        for key, value in previous.state_dict().items():
            self.assertTrue(torch.equal(value, current.state_dict()[key]), key)
        current.load_state_dict(previous.state_dict(), strict=True)
        for infer in (False, True):
            outputs = []
            for model in (previous, current):
                torch.manual_seed(99)
                with torch.no_grad():
                    outputs.append(model.infer(*self.batch[:2], max_frames=16) if infer
                                   else model(*self.batch[:4]))
            for key in outputs[0]:
                self.assertTrue(torch.equal(outputs[0][key], outputs[1][key]), key)

    def test_kaggle_exports_preserve_conditioning_and_embedded_evaluator(self):
        for path in build():
            notebook = json.loads(path.read_text(encoding="utf-8"))
            for cell in notebook["cells"]:
                if cell["cell_type"] == "code":
                    ast.parse(cell["source"])
                    self.assertEqual(cell["outputs"], [])
                    self.assertIsNone(cell["execution_count"])
            tree = ast.parse(notebook["cells"][1]["source"])
            assignment = next(n for n in tree.body if isinstance(n, ast.Assign)
                              and isinstance(n.targets[0], ast.Name) and n.targets[0].id == "source_snapshots")
            snapshots = ast.literal_eval(assignment.value)
            self.assertEqual(snapshots["sts_eval/reconstruction.py"],
                             (ROOT / "sts_eval/reconstruction.py").read_text(encoding="utf-8"))
            exported = definitions(snapshots["notebooks/vits-full-train.ipynb"])
            model = self.make_model(namespace=exported).eval()
            result = model.infer(*self.batch[:2], speaker_ids=self.batch[6], max_frames=8)
            self.assertEqual(result["generated_audio"].shape[0], 2)


if __name__ == "__main__":
    unittest.main()
