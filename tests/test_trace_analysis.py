import unittest
from tools.analyze_inference_trace import summarize_seed


class AnalysisTests(unittest.TestCase):
    def test_stability_and_revisits(self):
        rows = [dict(step=i+1, text=text, reached_length_cap=False)
                for i, text in enumerate(['a','a','b','a','a','a'])]
        result = summarize_seed(rows)
        self.assertEqual(result['transitions'], 5)
        self.assertEqual(result['changed'], 2)
        self.assertEqual(result['revisits'], 1)
        self.assertEqual(result['longest_unchanged_transition_run'], 2)
        self.assertEqual(result['final_unchanged_transition_run'], 2)

    def test_no_transitions(self):
        result = summarize_seed([dict(step=1, text='one', reached_length_cap=True)])
        self.assertEqual(result['transitions'], 0)
        self.assertIsNone(result['mean_word_edits'])
        self.assertEqual(result['capped_outputs'], 1)


if __name__ == '__main__':
    unittest.main()
