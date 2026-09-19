"""Behavioral checks using the actual notebook classes, without dataset downloads.

Synthetic waveforms below are test inputs, never speech-quality evidence.
Run from repository root: python -m unittest discover -s tests -v
"""

import ast
import json
from pathlib import Path
import tempfile
import unittest

import torch
import torchaudio

from sts_eval import EvaluationConfig, ReconstructionEvaluator, plot_history


ROOT = Path(__file__).resolve().parents[1]


def notebook_definitions():
    """Execute imports and definitions only; never run data loading/training cells."""
    notebook = json.loads((ROOT / "notebooks/vits-light_v1.ipynb").read_text(encoding="utf-8"))
    namespace = {"json": json}
    for i, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        source = cell["source"]
        source = source if isinstance(source, str) else "".join(source)
        tree = ast.parse(source)
        compile(tree, f"cell_{i}", "exec")
        if i == 1:
            exec(compile(tree, f"cell_{i}", "exec"), namespace)
        else:
            definitions = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))]
            if definitions:
                exec(compile(ast.Module(body=definitions, type_ignores=[]), f"cell_{i}", "exec"), namespace)
    return namespace


class ReconstructionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.definitions = notebook_definitions()

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        torch.manual_seed(7)
        # The second utterance requires padding to the 32-frame decoder crop.
        lengths = torch.tensor([40 * 256, 8 * 256])
        time = torch.arange(int(lengths.max())) / 22050
        waveform = torch.stack([0.2 * torch.sin(2 * torch.pi * 220 * time),
                                0.2 * torch.sin(2 * torch.pi * 330 * time)]).unsqueeze(1)
        waveform[1, :, lengths[1]:] = 0
        # Match the notebook preprocessing, independently for each real waveform.
        specs = []
        for i, length in enumerate(lengths):
            x = torch.nn.functional.pad(waveform[i, :, :length], (384, 384), mode="reflect")
            specs.append(torch.stft(x, 1024, 256, 1024, torch.hann_window(1024),
                                    center=False, return_complex=True).abs().squeeze(0))
        spec = torch.zeros(2, 513, 40)
        for i, value in enumerate(specs):
            spec[i, :, :value.shape[-1]] = value
        self.batch = (torch.tensor([[1, 2, 3], [1, 2, 0]]), torch.tensor([3, 2]),
                      spec, torch.tensor([40, 8]), waveform, lengths)
        # Keep the real decoder's 512-channel interface; only shrink encoder widths.
        self.model = self.definitions["VITSModel"](
            vocab_size=10, spec_channels=513, latent_channels=8, hidden_channels=8,
            feedforward_channels=16, decoder_channels=512,
        )

    def evaluator(self, name="eval", batch=None):
        return ReconstructionEvaluator(self.batch if batch is None else batch,
                                       ["test-tone-a", "test-tone-b"], self.root / name)

    def test_identity_and_degradation_metrics(self):
        evaluator = self.evaluator()
        target = self.batch[4][0, 0, :8192]
        identical = evaluator.score(target, target)
        degraded = evaluator.score(torch.zeros_like(target), target)
        self.assertTrue(all(value == 0 for value in identical.values()))
        self.assertTrue(all(value > 0 for value in degraded.values()))
        with self.assertRaises(FloatingPointError):
            evaluator.score(target * float("nan"), target)
        with self.assertRaises(ValueError):
            evaluator.score(target[:-1], target)

    def test_repeated_evaluation_preserves_rng_modes_and_parameters(self):
        evaluator = self.evaluator()
        self.model.train()
        self.model.decoder.eval()  # Preserve mixed per-module modes as well.
        flags = [m.training for m in self.model.modules()]
        parameters = [p.detach().clone() for p in self.model.parameters()]
        state = torch.random.get_rng_state().clone()
        first = evaluator.evaluate(self.model, 0)
        self.assertTrue(torch.equal(state, torch.random.get_rng_state()))
        second = evaluator.evaluate(self.model, 1)
        self.assertEqual(flags, [m.training for m in self.model.modules()])
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(parameters, self.model.parameters())))
        self.assertTrue(all(p.grad is None for p in self.model.parameters()))
        self.assertEqual([{k: v for k, v in r.items() if k != "step"} for r in first],
                         [{k: v for k, v in r.items() if k != "step"} for r in second])
        self.assertNotEqual(first[0]["log_mel_l1"], first[1]["log_mel_l1"])

    def test_padding_is_not_scored(self):
        evaluator = self.evaluator()
        evaluator.evaluate(self.model, 0)
        rows = json.loads((evaluator.output_dir / "step_000000/per_sample.json").read_text())
        self.assertEqual([r["valid_samples"] for r in rows], [8192, 2048, 8192, 2048])
        changed = [t.clone() for t in self.batch]
        changed[4][1, :, 2048:] = 100  # Corrupt padded target only.
        other = self.evaluator("changed_padding", tuple(changed))
        self.assertEqual(evaluator.history, other.evaluate(self.model, 0))

    def test_failure_restores_state_and_writes_no_success(self):
        evaluator = self.evaluator()
        with torch.no_grad():
            next(self.model.decoder.parameters()).fill_(float("nan"))
        flags = [m.training for m in self.model.modules()]
        state = torch.random.get_rng_state().clone()
        with self.assertRaises(FloatingPointError):
            evaluator.evaluate(self.model, 0)
        self.assertEqual(flags, [m.training for m in self.model.modules()])
        self.assertTrue(torch.equal(state, torch.random.get_rng_state()))
        self.assertFalse((evaluator.output_dir / "step_000000").exists())

    def test_invalid_inputs_and_output_collision(self):
        self.evaluator()
        with self.assertRaises(FileExistsError):
            self.evaluator()
        short = [t.clone() for t in self.batch]
        short[5][1] = 256
        with self.assertRaises(ValueError):
            self.evaluator("short", tuple(short))
        self.model.hop_length = 128
        with self.assertRaises(ValueError):
            self.evaluator("mismatch").evaluate(self.model, 0)

    def test_real_training_update_logging_audio_and_plot(self):
        evaluator = self.evaluator()
        loss = self.definitions["VITSLoss"](duration_weight=0, kl_weight=0)
        trainer = self.definitions["VITSTrainer"](
            self.model, loss, [self.batch], device="cpu", use_amp=False, use_multi_gpu=False,
            checkpoint_dir=self.root / "checkpoints", sample_dir=self.root / "samples",
        )
        before = next(self.model.posterior_encoder.parameters()).detach().clone()
        trainer.overfit_train(1, fixed_batch=self.batch, evaluator=evaluator,
                              eval_every=5, log_every=5, save_every=5, run_dir=self.root)
        self.assertFalse(torch.equal(before, next(self.model.posterior_encoder.parameters())))
        self.assertEqual([r["step"] for r in evaluator.history], [0, 0, 1, 1])
        self.assertEqual(len(trainer.training_history), 1)
        self.assertEqual(trainer.optimizer_update_stats["updates"], 1)
        self.assertEqual(trainer.training_history[0]["optimizer_updates"], 1)
        self.assertTrue((self.root / "checkpoints/vits_step_1.pt").is_file())
        wav, rate = torchaudio.load(str(evaluator.output_dir / "step_000001/001_target.wav"))
        self.assertEqual(rate, 22050)
        self.assertEqual(wav.shape[-1], 2048)
        self.assertTrue(torch.allclose(wav[0], self.batch[4][1, 0, :2048]))
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig = plot_history(evaluator.history, trainer.training_history, self.root / "curves.png")
        self.assertTrue((self.root / "curves.png").is_file())
        plt.close(fig)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_cuda_preserves_cpu_and_cuda_rng(self):
        model = self.model.cuda()
        evaluator = self.evaluator()
        cpu = torch.random.get_rng_state().clone()
        cuda = torch.cuda.get_rng_state().clone()
        evaluator.evaluate(model, 0)
        self.assertTrue(torch.equal(cpu, torch.random.get_rng_state()))
        self.assertTrue(torch.equal(cuda, torch.cuda.get_rng_state()))


if __name__ == "__main__":
    unittest.main()
