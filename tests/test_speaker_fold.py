"""Folding a speaker of a conditioned checkpoint into a single-speaker model (pretrain -> fine-tune).

Random weights and tones establish the arithmetic only, never speech quality.
"""

from pathlib import Path
import tempfile
import unittest

import torch

from tools.notebook_defs import literal_assignment, load_definitions

SPEAKERS = {"a": 0, "b": 1, "c": 2}


class SpeakerFoldTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)
        cls.ns = load_definitions()
        cls.ns["PAD_ID"] = 0
        cls.switches = dict(literal_assignment("TTS010_SWITCHES"))

    def model(self, n_speakers, duration_predictor="deterministic"):
        return self.ns["VITSModel"](vocab_size=76, spec_channels=513, latent_channels=8, hidden_channels=8,
                                    feedforward_channels=16, n_speakers=n_speakers,
                                    gin_channels=4 if n_speakers else 0, duration_predictor=duration_predictor)

    def trainer(self, model, root, speaker_to_id=None, switches=None):
        return self.ns["VITSTrainer"](
            model, self.ns["MultiPeriodDiscriminator"](), self.ns["VITSLoss"](), [], device="cpu",
            speaker_to_id=speaker_to_id, switches=switches or self.switches,
            checkpoint_dir=Path(root) / "checkpoints", sample_dir=Path(root) / "samples")

    def conditioned_checkpoint(self, root, duration_predictor):
        torch.manual_seed(7)
        model = self.model(len(SPEAKERS), duration_predictor)
        with torch.no_grad():  # move every weight (and the zero-initialised flow outputs) off its init
            for parameter in model.parameters():
                parameter.add_(0.1 * torch.randn_like(parameter))
        trainer = self.trainer(model, root, SPEAKERS, {**self.switches, "duration_predictor": duration_predictor}
                               if duration_predictor != "deterministic" else None)
        trainer.step = 5
        trainer.save_checkpoint()
        return model.eval(), trainer.checkpoint_dir / "vits_step_5.pt"

    def fold(self, path, root, speaker, duration_predictor="deterministic"):
        model = self.model(0, duration_predictor)
        switches = {**self.switches, "duration_predictor": duration_predictor} \
            if duration_predictor != "deterministic" else None
        record = self.trainer(model, Path(root) / "ft", switches=switches).init_from_speaker_checkpoint(path, speaker)
        return model.eval(), record

    def check_components(self, conditioned, folded, g):
        torch.manual_seed(1)
        spec, lengths = torch.rand(2, 513, 30), torch.tensor([30, 22])
        mask = self.ns["create_sequence_mask"](lengths, 30).unsqueeze(1).float()
        with torch.no_grad():
            expected = conditioned.posterior_encoder(spec, lengths, g=g, sample=False)
            actual = folded.posterior_encoder(spec, lengths, sample=False)
            for e, a in zip(expected, actual):
                torch.testing.assert_close(a, e, atol=1e-5, rtol=1e-5)
            z = torch.randn(2, 8, 30) * mask
            for reverse in (False, True):
                torch.testing.assert_close(folded.flow(z, mask, reverse=reverse),
                                           conditioned.flow(z, mask, g=g, reverse=reverse), atol=1e-5, rtol=1e-5)
            torch.testing.assert_close(folded.decoder(z[:, :, :8]), conditioned.decoder(z[:, :, :8], g=g),
                                       atol=1e-5, rtol=1e-5)

    def test_mean_fold_matches_the_conditioned_model_and_the_duration_predictor_inside_the_text(self):
        with tempfile.TemporaryDirectory() as root:
            conditioned, path = self.conditioned_checkpoint(root, "deterministic")
            folded, record = self.fold(path, root, "mean")
            g = conditioned.speaker_embedding.weight.mean(dim=0).view(1, 4, 1).expand(2, 4, 1)
            self.check_components(conditioned, folded, g)
            text, text_mask = torch.randn(2, 8, 12), torch.ones(2, 1, 12)
            text_mask[1, :, 9:] = 0
            with torch.no_grad():
                expected = conditioned.duration_predictor(text, text_mask, g=g)
                actual = folded.duration_predictor(text, text_mask)
            torch.testing.assert_close(actual[0, :, 2:-2], expected[0, :, 2:-2], atol=1e-5, rtol=1e-5)
            torch.testing.assert_close(actual[1, :, 2:7], expected[1, :, 2:7], atol=1e-5, rtol=1e-5)
            self.assertGreater((actual[0, :, 0] - expected[0, :, 0]).abs().max().item(), 1e-6)  # the stated edge
            self.assertEqual(record["optimizer"], "fresh")
            self.assertEqual(record["speaker_fold"], {"speaker": "mean", "source_speakers": 3, "gin_channels": 4,
                                                      "folded_biases": 8 + 4 + 1 + 1})

    def test_named_speaker_and_stochastic_predictor_fold_exactly(self):
        with tempfile.TemporaryDirectory() as root:
            conditioned, path = self.conditioned_checkpoint(root, "stochastic")
            folded, record = self.fold(path, root, "b", "stochastic")
            g = conditioned.speaker_embedding.weight[1].view(1, 4, 1).expand(2, 4, 1)
            self.check_components(conditioned, folded, g)
            text, text_mask = torch.randn(2, 8, 12), torch.ones(2, 1, 12)
            with torch.no_grad():
                torch.manual_seed(3)
                expected = conditioned.duration_predictor(text, text_mask, g=g, reverse=True, noise_scale=0.8)
                torch.manual_seed(3)
                actual = folded.duration_predictor(text, text_mask, reverse=True, noise_scale=0.8)
            torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
            self.assertEqual(record["speaker_fold"]["speaker"], "b")

    def test_fold_refuses_mismatches(self):
        with tempfile.TemporaryDirectory() as root:
            _, path = self.conditioned_checkpoint(root, "deterministic")
            with self.assertRaisesRegex(ValueError, "not in the checkpoint"):
                self.fold(path, root, "nobody")
            with self.assertRaisesRegex(ValueError, "switches"):
                self.fold(path, root, "mean", "stochastic")
            with self.assertRaisesRegex(ValueError, "unconditioned model"):
                self.trainer(self.model(3), Path(root) / "x", SPEAKERS).init_from_speaker_checkpoint(path)
            plain = self.trainer(self.model(0), Path(root) / "plain")
            plain.step = 1
            plain.save_checkpoint()
            with self.assertRaisesRegex(ValueError, "not speaker-conditioned"):
                self.fold(plain.checkpoint_dir / "vits_step_1.pt", root, "mean")


if __name__ == "__main__":
    unittest.main()
