"""Optional actual-model GPU check using synthetic latents; no AE/downloads."""
import json
from pathlib import Path
import tempfile
import unittest

import torch
from tools import longrun_training as engine


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required for actual-model micro check")
class LongRunActualModelTests(unittest.TestCase):
    def test_both_architectures_train_validate_save_resume_and_sample(self):
        config = json.loads((Path(__file__).resolve().parents[1]/"configs/afrikaans_longrun.json").read_text())
        config.update(micro_batch=2, accumulation=1, validation_examples=2, train_probe_examples=2,
                      validation_repeats=1, sampling_steps=2, checkpoint_every_steps=1, log_every_steps=1)
        engine.STOP = False
        engine.seed_all(17)
        z = torch.nn.functional.normalize(torch.randn(8,32,64), dim=-1)*8
        cache = dict(train=z[:6], valid=z[6:])
        report = dict(hardware=engine.hardware(), scope="synthetic latent code check, not cluster throughput or language quality", models={})
        with tempfile.TemporaryDirectory() as temp:
            for name in config["models"]:
                spec = dict(name=name, model=name, seed=42, epochs=2)
                folder = Path(temp)/name
                first = engine.train_run(config, spec, cache, folder, {}, max_new_steps=2)
                second = engine.train_run(config, spec, cache, folder, {}, max_new_steps=1)
                self.assertEqual((first["step"], second["step"]), (2,3))
                model = engine.make_model(config, name)
                state = engine.load_checkpoint(folder/"best.pt", "cuda")
                model.load_state_dict(state["ema"])
                model.eval()
                with torch.no_grad():
                    samples, _ = model.sample(1, torch.tensor([32], device="cuda"))
                self.assertEqual(tuple(samples.shape), (1,32,64))
                self.assertTrue(torch.isfinite(samples).all())
                report["models"][name] = dict(parameters=sum(p.numel() for p in model.parameters()),
                    resumed_step=second["step"], finite_sample=True,
                    peak_allocated_bytes=torch.cuda.max_memory_allocated())
                del model, state
                torch.cuda.empty_cache()
        destination = Path(__file__).resolve().parents[1]/"outputs/longrun-local-checks/model-check.json"
        engine.write_json(destination, report)


if __name__ == "__main__":
    unittest.main()
