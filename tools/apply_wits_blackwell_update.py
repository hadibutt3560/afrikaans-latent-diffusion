"""Update only three verified files in the existing cluster bundle; preserve backups."""
import hashlib
from pathlib import Path
import shutil
import sys


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f'Expected exactly one occurrence of {old!r}; refusing to guess')
    return text.replace(old, new, 1)


def updated(relative, text):
    if relative.endswith(('.sh', '.slurm')):
        return replace_once(text, 'ld4lg-reproduction/bin/python', 'ld4lg-blackwell/bin/python')
    text = replace_once(text, "expected = {'torch': '2.2.2'", "expected = {'torch': '2.7.1'")
    text = replace_once(text, '    from latent_models.latent_finetuning import Trainer\n',
                        "    import torch\n"
                        "    if torch.version.cuda != '12.8':\n"
                        "        raise RuntimeError(f'Expected CUDA 12.8 PyTorch build, found {torch.version.cuda}')\n"
                        '    from latent_models.latent_finetuning import Trainer\n')
    return replace_once(text, "    print('CPU environment and required files checked.', flush=True)\n",
                        "    checkpoint = torch.load(ROOT / 'inputs/afrikaans-pilot/model.pt', map_location='cpu', weights_only=True, mmap=True)\n"
                        "    if checkpoint.get('step') != 500 or not {'model', 'opt', 'scheduler'}.issubset(checkpoint):\n"
                        "        raise RuntimeError('Unexpected pilot checkpoint; inspect before submitting')\n"
                        "    print('CPU environment, required files and checkpoint compatibility checked.', flush=True)\n")


def main():
    root = (Path(sys.argv[1]) if len(sys.argv) > 1 else Path.home() / 'research-afrikaans-24h').resolve()
    marker = root / 'blackwell-update-applied.txt'
    if marker.exists():
        print('Update already applied; no files changed.')
        return
    manifest = root / 'bundle-manifest.sha256'
    original = manifest.read_text()
    hashes = dict((line.split('  ', 1)[1], line.split('  ', 1)[0]) for line in original.splitlines())
    names = ['tools/wits_afrikaans_train.py', 'cluster/wits/submit_afrikaans_24h.sh', 'cluster/wits/afrikaans_24h.slurm']
    changes = []
    for name in names:
        path = root / name
        content = path.read_bytes()
        if hashlib.sha256(content).hexdigest() != hashes.get(name):
            raise RuntimeError(f'{name} differs from the bundle manifest; stopping without edits')
        new = updated(name, content.decode()).encode()
        if name.endswith('.py'):
            compile(new, name, 'exec')
        if path.with_name(path.name + '.before-blackwell').exists():
            raise RuntimeError('Backup exists; inspect a previous partial update before retrying')
        changes.append((path, new))
        hashes[name] = hashlib.sha256(new).hexdigest()
    shutil.copy2(manifest, root / 'bundle-manifest.before-blackwell.sha256')
    for path, new in changes:
        shutil.copy2(path, path.with_name(path.name + '.before-blackwell'))
        temporary = path.with_name(path.name + '.blackwell.tmp')
        temporary.write_bytes(new)
        temporary.replace(path)
    temporary = manifest.with_suffix('.tmp')
    temporary.write_text(''.join(f'{digest}  {name}\n' for name, digest in hashes.items()))
    temporary.replace(manifest)
    marker.write_text('PyTorch 2.7.1 CUDA 12.8; ld4lg-blackwell; original files backed up.\n')
    print('Blackwell launcher update applied. Data and checkpoints were not changed.')


if __name__ == '__main__':
    main()
