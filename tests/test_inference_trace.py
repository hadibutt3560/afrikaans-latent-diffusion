import unittest
from types import SimpleNamespace
import torch
from tools.inference_trace_utils import annotate, edit_distance, capture_predictions


class FakeSampler(torch.nn.Module):
    sampling_timesteps = 5

    def diffusion_model_predictions(self):
        return SimpleNamespace(pred_x_start=torch.randn(1, 2, 3))


class TraceTests(unittest.TestCase):
    def test_edits(self):
        self.assertEqual(edit_distance(['a','b'], ['a','c']), 1)
        self.assertEqual(edit_distance([], ['a']), 1)

    def test_cycle_not_same_as_stability(self):
        rows = annotate([dict(step=i+1, text=x) for i,x in enumerate(['a','a','b','a'])])
        self.assertIsNone(rows[0]['changed_from_previous'])
        self.assertFalse(rows[1]['revisits_earlier_text'])
        self.assertTrue(rows[3]['revisits_earlier_text'])

    def test_capture_does_not_change_rng_or_values(self):
        model = FakeSampler()
        torch.manual_seed(42)
        expected = [model.diffusion_model_predictions().pred_x_start for _ in range(5)]
        torch.manual_seed(42)
        with capture_predictions(model, 2) as trace:
            actual = [model.diffusion_model_predictions().pred_x_start for _ in range(5)]
        self.assertEqual([x[0] for x in trace], [1, 2, 4, 5])
        self.assertTrue(all(torch.equal(x,y) for x,y in zip(expected, actual)))
        self.assertNotIn('diffusion_model_predictions', model.__dict__)

    def test_restore_after_error(self):
        model = FakeSampler()
        with self.assertRaises(RuntimeError):
            with capture_predictions(model, 1):
                raise RuntimeError('interrupted')
        self.assertNotIn('diffusion_model_predictions', model.__dict__)

    def test_all_steps_when_interval_one(self):
        model = FakeSampler()
        with capture_predictions(model, 1) as rows:
            for _ in range(5):
                model.diffusion_model_predictions()
        self.assertEqual([r[0] for r in rows], [1, 2, 3, 4, 5])


if __name__ == '__main__':
    unittest.main()
