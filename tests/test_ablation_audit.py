import unittest
from tools.audit_inference_ablation import weighted, interval


class AuditTests(unittest.TestCase):
    def test_token_weighting(self):
        import math
        rows = [{'mgpt_score': {'nll': 2.0, 'tokens': 1}},
                {'mgpt_score': {'nll': 6.0, 'tokens': 3}}]
        self.assertAlmostEqual(weighted(rows), math.exp(2))

    def test_identical_pairs(self):
        rows = [{'mgpt_score': {'nll': float(i), 'tokens': i}} for i in (1, 2, 3)]
        self.assertEqual(interval(rows, rows), [1.0, 1.0])

    def test_uniform_improvement(self):
        import math
        control = [{'mgpt_score': {'nll': 2.0*i, 'tokens': i}} for i in (1, 2, 3)]
        better = [{'mgpt_score': {'nll': float(i), 'tokens': i}} for i in (1, 2, 3)]
        for bound in interval(better, control):
            self.assertAlmostEqual(bound, math.exp(-1))


if __name__ == '__main__':
    unittest.main()
