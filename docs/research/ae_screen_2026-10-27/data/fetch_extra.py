"""Download the Afrikaans parts of Gov-ZA and Vuk'uzenzele at pinned revisions.

Writes data/extra_afrikaans/raw/ with the original files, a documents.jsonl per corpus and a
manifest of revisions, licences and SHA-256 hashes. Re-running refuses to overwrite.
"""
import hashlib, json, zipfile
from pathlib import Path
from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / 'data/extra_afrikaans/raw'
SOURCES = {
    'gov-za': dict(repo='dsfsi/gov-za-monolingual', revision='12fc7cccf0dd24b628462b49ec7bfe98509e958b',
                   files=['govza-cabinet-statements.json.zip', 'LICENSE', 'LICENSE_data.md', 'README.md'],
                   licence='MIT (dataset card); text from gov.za cabinet statements, GCIS'),
    'vukuzenzele': dict(repo='dsfsi/vukuzenzele-monolingual', revision='b689fe84fdc2ceb01793dfeb9835327825e7a987',
                        files=['afr/train-00000-of-00001.parquet', 'afr/eval-00000-of-00001.parquet',
                               'afr/test-00000-of-00001.parquet', 'LICENSE', 'LICENSE_data.md', 'README.md'],
                        licence='CC BY 4.0 (data); Vuk\'uzenzele newspaper, GCIS'),
}


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def gov_za_documents(folder):
    with zipfile.ZipFile(folder / 'govza-cabinet-statements.json.zip') as archive:
        names = [n for n in archive.namelist() if n.endswith('.json') and not n.startswith('__MACOSX/')]
        assert len(names) == 1, names
        rows = json.loads(archive.read(names[0]).decode('utf8'))
    documents = []
    for i, row in enumerate(rows):
        text = row['af']['text'] if isinstance(row.get('af'), dict) else None
        if text and text.strip():
            documents.append(dict(source='gov-za', source_index=i, title=row.get('title'), date=row.get('date'),
                                  url=row.get('url'), text=text))
    return documents


def vukuzenzele_documents(folder):
    import pandas as pd
    documents = []
    for split in ('train', 'eval', 'test'):
        frame = pd.read_parquet(folder / f'afr/{split}-00000-of-00001.parquet')
        for i, row in frame.iterrows():
            if isinstance(row['text'], str) and row['text'].strip():
                documents.append(dict(source='vukuzenzele', source_index=f'{split}-{i}', title=str(row['title']).strip(),
                                      edition=row['edition'], author=str(row['author']).strip(), text=row['text']))
    return documents


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    manifest = {}
    for name, spec in SOURCES.items():
        folder = OUT / name
        for file in spec['files']:
            path = Path(hf_hub_download(spec['repo'], file, repo_type='dataset', revision=spec['revision'],
                                        local_dir=folder))
            assert path.exists(), file
        documents = gov_za_documents(folder) if name == 'gov-za' else vukuzenzele_documents(folder)
        with (folder / 'documents.jsonl').open('x', encoding='utf8') as f:
            for d in documents:
                f.write(json.dumps(d, ensure_ascii=False) + '\n')
        manifest[name] = dict(repo=spec['repo'], revision=spec['revision'], licence=spec['licence'],
                              documents=len(documents), words=sum(len(d['text'].split()) for d in documents),
                              files={f: digest(folder / f) for f in spec['files'] + ['documents.jsonl']})
    with (OUT / 'manifest.json').open('x', encoding='utf8') as f:
        json.dump(manifest, f, indent=2)
    print(json.dumps({k: {x: v[x] for x in ('documents', 'words')} for k, v in manifest.items()}, indent=2))


if __name__ == '__main__':
    main()
