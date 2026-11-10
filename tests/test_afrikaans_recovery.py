import unittest
import tempfile
import json
from pathlib import Path
from train_latent_model import build_parser, merge_saved_args
from tools.afrikaans_recovery import command


class RecoveryTests(unittest.TestCase):
    def test_evaluation_does_not_reapply_original_initialization(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / 'args.json').write_text(json.dumps({'init_dir': 'old-baseline', 'validation_shuffle': False}))
            args = build_parser().parse_args(['--eval', '--resume_dir', tmp, '--validation_shuffle'])
            args = merge_saved_args(args)
            self.assertIsNone(args.init_dir)
            self.assertTrue(args.validation_shuffle)

    def test_selective_mode(self):
        args = build_parser().parse_args(['--lm_mode', 'decoder_last'])
        self.assertEqual(args.lm_mode, 'decoder_last')

    def test_diagnostic_is_explicit_and_reproducible(self):
        cmd = command(Path('out'), 'freeze', 250, Path('baseline'), tiny=True)
        args = build_parser().parse_args(cmd[3:])
        self.assertTrue(args.overfit_subset)
        self.assertEqual(args.train_subset_size, 32)
        self.assertEqual(args.init_dir, 'baseline')
        self.assertFalse(args.resume_training)
        self.assertTrue(args.fail_on_validation_error)

    def test_generalization_does_not_overfit_validation(self):
        cmd = command(Path('out'), 'decoder_last', 1000, Path('baseline'))
        args = build_parser().parse_args(cmd[3:])
        self.assertFalse(args.overfit_subset)
        self.assertTrue(args.validation_shuffle)
        self.assertEqual(args.validation_max_examples, 128)


if __name__ == '__main__':
    unittest.main()
