"""Download the pinned public EPI files, or verify existing files without network."""
import argparse
import hashlib
import json
import shutil
import tempfile
import urllib.request
from pathlib import Path


REVISION = 'f1290ccd49d4c7f8014cac47c93779aed06a294b'
# SHA256 and compressed size read from the pinned repository's LFS metadata.
FILES = {
    'train': ('6de1f9d0d8d6c543ab346bbfd8956a1face4230f9da6d4f437d8945d85d78631', 13735810),
    'val': ('ffda22308238babbc475d9f15e21d9abb58de6c9f21260f58211279d0afe74a0', 2750898),
    'test': ('c02eed9adc4853bc9698c504a25e3a252bceaafc069a4a834057a4b4daf4f712', 2752314),
}


def verify(path, expected, size):
    assert path.stat().st_size == size, f'Size mismatch: {path}'
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert digest == expected, f'Hash mismatch; existing file left unchanged: {path}'
    return digest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--check-only', action='store_true', help='Read-only, no network; all files must exist')
    args = parser.parse_args()
    root = Path('data/GUE_v2_source')
    if not args.check_only:
        root.mkdir(parents=True, exist_ok=True)
    manifest = []
    for split, (expected, size) in FILES.items():
        path = root/f'EPI_GM12878_{split}.csv.gz'
        relative = f'EPI_GM12878/{path.name}'
        url = f'https://huggingface.co/datasets/genomic-benchmarks/GUE_v2/resolve/{REVISION}/{relative}'
        if not path.exists():
            assert not args.check_only, f'Missing file: {path}'
            with tempfile.NamedTemporaryFile(dir=root, suffix='.part', delete=False) as output:
                temporary = Path(output.name)
                try:
                    with urllib.request.urlopen(url, timeout=120) as response:
                        shutil.copyfileobj(response, output)
                except BaseException:
                    output.close()
                    temporary.unlink(missing_ok=True)
                    raise
            try:
                verify(temporary, expected, size)
                # Rename refuses to overwrite an existing destination on Windows.
                temporary.rename(path)
            except BaseException:
                temporary.unlink(missing_ok=True)
                raise
        digest = verify(path, expected, size)
        manifest.append({'path':relative, 'revision':REVISION,
                         'lfs':{'oid':expected,'size':size}, 'download_sha256':digest})
        print(f'Verified {path.name}: {digest}', flush=True)
    metadata = root/'lfs_verification.json'
    if not args.check_only and not metadata.exists():
        metadata.write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf-8')


if __name__ == '__main__':
    main()
