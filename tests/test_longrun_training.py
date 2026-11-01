"""Small real optimizer tests; no model downloads or long GPU jobs."""
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import torch
from tools import longrun_training as engine


class TinyDiffusion(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 3)
        self.dropout = torch.nn.Dropout(.1)
        self.register_buffer("counter", torch.zeros(1))

    def forward(self, z, mask, class_id=None):
        noise = torch.randn_like(z) * (1 + random.random())
        return (self.linear(self.dropout(z + noise)) - z).square().mean()


class LongRunTrainingTests(unittest.TestCase):
    def setUp(self):
        engine.STOP = False
        self.config = json.loads((Path(__file__).resolve().parents[1]/"configs/afrikaans_longrun.json").read_text())
        self.config.update(micro_batch=2, accumulation=3, validation_examples=4, validation_repeats=2,
                           train_probe_examples=4, checkpoint_every_steps=2, log_every_steps=1,
                           validate_every_epochs=1, early_stop_min_epochs=100, latent_length=2, latent_dim=3)
        self.spec = dict(name="tiny", model="small", seed=42, epochs=3)
        self.cache = dict(train=torch.arange(66).reshape(11,2,3).float()/66,
                          valid=torch.ones(5,2,3))

    def train(self, folder, steps=None):
        with patch.object(engine, "make_model", lambda *_: TinyDiffusion()):
            return engine.train_run(self.config, self.spec, self.cache, folder, {"fixture":True},
                                    max_new_steps=steps, device="cpu")

    def test_resume_equals_uninterrupted_weights_ema_optimizer_scheduler_rng(self):
        with tempfile.TemporaryDirectory() as temp:
            first, second = Path(temp)/"one", Path(temp)/"two"
            self.train(first)
            self.assertEqual(self.train(second, 3)["step"], 3)
            self.assertEqual(self.train(second)["step"], 6)
            a, b = engine.load_checkpoint(first/"latest.pt"), engine.load_checkpoint(second/"latest.pt")
            for group in ("model", "ema"):
                for key in a[group]:
                    self.assertTrue(torch.equal(a[group][key], b[group][key]), (group, key))
            for key in a["optimizer"]["state"]:
                for field in a["optimizer"]["state"][key]:
                    self.assertTrue(torch.equal(a["optimizer"]["state"][key][field], b["optimizer"]["state"][key][field]))
            self.assertEqual(a["scheduler"], b["scheduler"])
            self.assertTrue(torch.equal(a["rng"]["cpu"], b["rng"]["cpu"]))
            self.assertEqual(a["rng"]["python"], b["rng"]["python"])
            self.assertTrue((second/"best.pt").exists())

    def test_shuffled_epoch_visits_every_example_once_including_tail(self):
        ids = torch.cat([engine.epoch_batch(11, self.config, 42, step) for step in (0,1)])
        self.assertEqual(sorted(ids.tolist()), list(range(11)))
        next_ids = torch.cat([engine.epoch_batch(11, self.config, 42, step) for step in (2,3)])
        self.assertFalse(torch.equal(ids, next_ids))

    def test_validation_does_not_change_training_rng(self):
        engine.seed_all(77)
        model = TinyDiffusion()
        before = engine.rng_state()
        engine.validation(model, self.cache, self.config, device="cpu")
        after = engine.rng_state()
        self.assertEqual(before["python"], after["python"])
        self.assertTrue(torch.equal(before["cpu"], after["cpu"]))

    def test_changed_protocol_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            self.train(temp, 1)
            self.config["learning_rate"] *= 2
            with self.assertRaisesRegex(ValueError, "changed"):
                self.train(temp)

    def test_nonfinite_loss_preserves_last_valid_checkpoint(self):
        with tempfile.TemporaryDirectory() as temp:
            self.train(temp, 1)
            before = engine.digest(Path(temp)/"latest.pt")
            self.cache["train"][:] = float("nan")
            with self.assertRaises(FloatingPointError):
                self.train(temp)
            self.assertEqual(before, engine.digest(Path(temp)/"latest.pt"))
            self.assertTrue((Path(temp)/"failure.json").exists())

    def test_signal_flag_saves_resumable_boundary(self):
        with tempfile.TemporaryDirectory() as temp:
            engine.STOP = True
            result = self.train(temp)
            self.assertEqual(result["status"], "paused")
            self.assertEqual(result["step"], 0)
            self.assertTrue((Path(temp)/"latest.pt").exists())

    def test_interrupted_validation_is_replayed_before_next_update(self):
        with tempfile.TemporaryDirectory() as temp:
            with patch.object(engine, "validation", side_effect=InterruptedError("signal")):
                result = self.train(temp)
            self.assertEqual(result["step"], 2)
            self.assertEqual(result["last_validated_step"], -1)
            result = self.train(temp, 1)
            self.assertEqual(result["step"], 3)
            rows = engine.read_json(Path(temp)/"status.json")
            self.assertEqual(rows["last_validated_step"], 3)
            validation_steps = [json.loads(r)["step"] for r in (Path(temp)/"metrics.jsonl").read_text().splitlines()
                                if json.loads(r)["kind"] == "validation"]
            self.assertEqual(validation_steps, [2,3])


if __name__ == "__main__":
    unittest.main()
