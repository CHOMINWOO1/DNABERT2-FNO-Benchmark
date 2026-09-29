"""Download pinned public LRB metadata/labels; never execute its dataset loader."""
import hashlib
import json
import time
import urllib.request
from pathlib import Path

REPO='InstaDeepAI/genomics-long-range-benchmark'
REVISION='b56f68ee30c01daf8f80a8d0b8a47f84199de08e'
ROOT=Path('data/continuous_rna_source')


def fetch(url):
    with urllib.request.urlopen(url,timeout=90) as response:return response.read()


def main():
    ROOT.mkdir(parents=True,exist_ok=True)
    api=f'https://huggingface.co/api/datasets/{REPO}/tree/{REVISION}/bulk_rna_expression?expand=true'
    tree=json.loads(fetch(api));(ROOT/'pinned_tree.json').write_text(json.dumps(tree,indent=2),encoding='utf-8')
    files=[('README.md',None),('genomics-long-range-benchmark.py',None)]+[(r['path'],r) for r in tree if r['type']=='file']
    manifest=[]
    for name,meta in files:
        dest=ROOT/Path(name).name;url=f'https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/{name}'
        if not dest.exists():
            data=fetch(url);part=dest.with_suffix(dest.suffix+'.part');part.write_bytes(data);part.replace(dest)
        data=dest.read_bytes();digest=hashlib.sha256(data).hexdigest()
        if meta:
            assert len(data)==meta['size']
            if 'lfs' in meta:assert digest==meta['lfs']['oid']
            else:assert hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()==meta['oid']
        manifest.append({'path':str(dest),'url':url,'sha256':digest,'bytes':len(data),'repository_entry':meta})
        print(f'Downloaded and verified {dest.name}: {len(data)} bytes',flush=True)
    sizes=ROOT/'hg19.chrom.sizes';url='https://hgdownload.soe.ucsc.edu/goldenPath/hg19/bigZips/hg19.chrom.sizes'
    if not sizes.exists():sizes.write_bytes(fetch(url))
    manifest.append({'path':str(sizes),'url':url,'sha256':hashlib.sha256(sizes.read_bytes()).hexdigest(),'bytes':sizes.stat().st_size})
    (ROOT/'download_manifest.json').write_text(json.dumps({'revision':REVISION,'files':manifest,'download_unix':time.time()},indent=2),encoding='utf-8')


if __name__=='__main__':main()
