"""Protocol safety tests runnable with Python's standard library alone."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from tools.longrun_common import audit_data, identity, lr_at_step, make_plan, steps_per_epoch, validate_config, write_json

ROOT = Path(__file__).resolve().parents[1]


class LongRunContractTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((ROOT / "configs/afrikaans_longrun.json").read_text())

    def dataset(self, folder):
        for split, text in zip(("train", "valid", "test"), ("Een voorbeeld", "Nog 'n sin", "Ander woorde")):
            row = dict(id=split, document_id=split, source_document=split+".txt", text=text, split=split, language="afr")
            (folder / f"{split}.jsonl").write_text(json.dumps(row)+"\n", encoding="utf8")

    def test_audit_accepts_disjoint_splits_and_rejects_document_leakage(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            self.dataset(folder)
            self.assertTrue(audit_data(folder)["passed"])
            path = folder / "valid.jsonl"
            row = json.loads(path.read_text())
            row["document_id"] = "train"
            path.write_text(json.dumps(row)+"\n")
            self.assertFalse(audit_data(folder)["passed"])

    def test_audit_rejects_normalized_duplicates_and_wrong_split(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            self.dataset(folder)
            path = folder / "valid.jsonl"
            row = json.loads(path.read_text())
            row.update(text=" EEN   VOORBEELD ", split="test")
            path.write_text(json.dumps(row)+"\n")
            report = audit_data(folder)
            self.assertFalse(report["passed"])
            self.assertEqual(report["overlap"]["train-valid"]["texts"], 1)

    def test_batch_accounting_covers_tail(self):
        self.assertEqual(steps_per_epoch(47767, self.config), 747)

    def test_lr_warmup_decay_floor(self):
        values = [lr_at_step(i, 10000, self.config) for i in (0, 999, 1000, 5000, 9999)]
        self.assertGreater(values[1], values[0])
        self.assertAlmostEqual(values[1], self.config["learning_rate"])
        self.assertGreater(values[2], values[3])
        self.assertAlmostEqual(values[-1], self.config["learning_rate"]*self.config["minimum_lr_ratio"])

    def timing(self, seconds=.2):
        timing = dict(step_seconds_median=seconds, validation_seconds=10., generation_seconds=20., checkpoint_seconds=.5)
        return dict(config_sha256=identity(self.config), hardware={"gpu":"fixture"},
                    models={name:copy.deepcopy(timing) for name in self.config["models"]})

    def test_planner_uses_real_measurements_and_locks_matched_epochs(self):
        plan = make_plan(self.config, self.timing(), 47767)
        self.assertGreaterEqual(plan["expected_training_and_validation_hours"], 48)
        self.assertLess(abs(plan["expected_training_and_validation_hours"]-72), 1)
        self.assertGreaterEqual(plan["initial_seed_pairs"], 3)
        self.assertEqual(len({r["epochs"] for r in plan["initial_runs"]}), 1)
        self.assertEqual(len(plan["initial_runs"])+len(plan["reserve_runs"]), 24)

    def test_planner_rejects_stale_config_and_impossible_budget(self):
        timing = self.timing()
        timing["config_sha256"] = "wrong"
        with self.assertRaisesRegex(ValueError, "differs"):
            make_plan(self.config, timing, 47767)
        timing = self.timing(.00001)
        for value in timing["models"].values():
            value.update(validation_seconds=0., generation_seconds=0., checkpoint_seconds=0.)
        with self.assertRaisesRegex(ValueError, "scope"):
            make_plan(self.config, timing, 47767)

    def test_atomic_json_rejects_nonfinite_and_keeps_previous(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/"state.json"
            write_json(path, {"step":2})
            with self.assertRaises(ValueError):
                write_json(path, {"loss":float("nan")})
            self.assertEqual(json.loads(path.read_text()), {"step":2})

    def test_bad_budget_and_batch_rejected(self):
        self.config["minimum_hours"] = 10
        with self.assertRaisesRegex(ValueError, "budget"):
            validate_config(self.config)
        self.config["minimum_hours"] = 48
        self.config["micro_batch"] = 0
        with self.assertRaisesRegex(ValueError, "positive"):
            validate_config(self.config)


if __name__ == "__main__":
    unittest.main()
