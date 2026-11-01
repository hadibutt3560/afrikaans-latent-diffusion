import hashlib
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from tools.apply_wits_blackwell_update import main, updated


class BlackwellUpdateTests(unittest.TestCase):
    def test_update_backup_checksums_and_repeat(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            originals = {
                'cluster/wits/afrikaans_24h.slurm': 'PY="$HOME/.conda/envs/ld4lg-reproduction/bin/python"\n',
                'cluster/wits/submit_afrikaans_24h.sh': 'PY="$HOME/.conda/envs/ld4lg-reproduction/bin/python"\n',
                'tools/wits_afrikaans_train.py': "def check():\n    expected = {'torch': '2.2.2'}\n    from latent_models.latent_finetuning import Trainer\n    print('CPU environment and required files checked.', flush=True)\n",
            }
            for name, content in originals.items():
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(content.encode())
            manifest = ''.join(f'{hashlib.sha256(content.encode()).hexdigest()}  {name}\n' for name, content in originals.items())
            (root / 'bundle-manifest.sha256').write_text(manifest)
            with patch.object(sys, 'argv', ['update', str(root)]):
                main()
                main()
            for line in (root / 'bundle-manifest.sha256').read_text().splitlines():
                digest, name = line.split('  ', 1)
                path = root / name
                self.assertEqual(digest, hashlib.sha256(path.read_bytes()).hexdigest())
                self.assertEqual(path.with_name(path.name + '.before-blackwell').read_text(), originals[name])

    def test_unexpected_content_rejected(self):
        with self.assertRaises(ValueError):
            updated('test.sh', 'unexpected code')

    def test_local_training_script_matches_updater(self):
        root = Path(__file__).resolve().parents[1]
        new = (root / 'tools/wits_afrikaans_train.py').read_text()
        old = new.replace("expected = {'torch': '2.7.1'", "expected = {'torch': '2.2.2'")
        old = old.replace("    import torch\n    if torch.version.cuda != '12.8':\n        raise RuntimeError(f'Expected CUDA 12.8 PyTorch build, found {torch.version.cuda}')\n", '')
        begin = old.index("    checkpoint = torch.load(ROOT / 'inputs/afrikaans-pilot/model.pt'")
        end = old.index('\n\n\ndef main():', begin)
        old = old[:begin] + "    print('CPU environment and required files checked.', flush=True)" + old[end:]
        self.assertEqual(updated('tools/wits_afrikaans_train.py', old), new)


if __name__ == '__main__':
    unittest.main()
