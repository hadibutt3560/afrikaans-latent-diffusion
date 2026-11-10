"""Package current working-tree code, prepared corpus, checkpoint and cached mT5."""
import hashlib
import io
from pathlib import Path
import tarfile
from contextlib import contextmanager

ROOT = Path(__file__).resolve().parents[1]
PREFIX = 'research-afrikaans-24h'


@contextmanager
def buffered_archive(path):
    # Large buffers avoid small cross-filesystem writes on WSL's Windows mount.
    with path.open('wb', buffering=8 * 1024 * 1024) as output:
        with tarfile.open(fileobj=output, mode='w:gz', compresslevel=1, dereference=True) as archive:
            yield archive


def main():
    destination = ROOT / 'outputs/cluster-transfer'
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / f'{PREFIX}.tar.gz'
    if target.exists():
        raise FileExistsError(target)
    sources = [(path, path.relative_to(ROOT)) for path in ROOT.glob('*.py')]
    for directory in ('dataset_utils', 'diffusion', 'evaluation', 'latent_models',
                      'model', 'utils', 'tools', 'tests', 'cluster', 'scripts'):
        sources += [(path, path.relative_to(ROOT)) for path in (ROOT / directory).rglob('*')
                    if path.is_file() and path.suffix in {'.py', '.sh', '.slurm'}]
    for name in ('environment.reproduction.yml', 'README.md', 'LICENSE'):
        if (ROOT / name).exists():
            sources.append((ROOT / name, Path(name)))
    data = Path('data/nchlt_afrikaans/processed/stage_5_7')
    for split in ('train', 'valid', 'test'):
        sources.append((ROOT / data / f'{split}.jsonl', data / f'{split}.jsonl'))
    checkpoint = ROOT / 'outputs/nchlt-afrikaans/recovery/pilot-decoder_last-no-backbone-dropout-500/best'
    for name in ('model.pt', 'args.json', 'selection.json'):
        sources.append((checkpoint / name, Path('inputs/afrikaans-pilot') / name))
    snapshots = list(Path('/home/scientist/.cache/huggingface/hub/models--google--mt5-base/snapshots').iterdir())
    if len(snapshots) != 1:
        raise RuntimeError('Expected one cached mT5 revision; select explicitly')
    for name in ('config.json', 'generation_config.json', 'pytorch_model.bin',
                 'spiece.model', 'special_tokens_map.json', 'tokenizer_config.json'):
        sources.append((snapshots[0] / name, Path('models/mt5-base') / name))
    for path, _ in sources:
        if not path.is_file():
            raise FileNotFoundError(path)
    manifest = []
    temporary = target.with_suffix('.tmp')
    with buffered_archive(temporary) as archive:
        for source, relative in sorted(sources, key=lambda pair: str(pair[1])):
            digest = hashlib.sha256()
            with source.open('rb') as stream:
                for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                    digest.update(block)
            info = archive.gettarinfo(str(source), arcname=f'{PREFIX}/{relative.as_posix()}')
            if source.suffix in {'.sh', '.slurm'}:
                content = source.read_bytes().replace(b'\r\n', b'\n')
                digest = hashlib.sha256(content)
                info.size = len(content)
                archive.addfile(info, io.BytesIO(content))
            else:
                with source.open('rb', buffering=8 * 1024 * 1024) as stream:
                    archive.addfile(info, stream)
            manifest.append(f'{digest.hexdigest()}  {relative.as_posix()}\n')
            print(f'Packed {relative} ({info.size} bytes)', flush=True)
        content = ''.join(manifest).encode()
        info = tarfile.TarInfo(f'{PREFIX}/bundle-manifest.sha256')
        info.size = len(content)
        archive.addfile(info, io.BytesIO(content))
    temporary.replace(target)
    print(f'Bundle ready: {target} ({target.stat().st_size} bytes)', flush=True)


if __name__ == '__main__':
    main()
