"""Download the official GUE archive; extract only one promoter task."""
import argparse
import hashlib
import json
import shutil
import tarfile
import zipfile
from pathlib import Path, PurePosixPath

import gdown

GUE_ID = "1uOrwlf07qGQuruXqGXWMpPn8avBoW7T-"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=["prom_300_tata", "prom_300_all", "prom_300_notata"], default="prom_300_tata")
    p.add_argument("--root", type=Path, default=Path("data"))
    p.add_argument("--archive", type=Path)
    p.add_argument("--source", choices=["official", "mirror"], default="official")
    args = p.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    if args.source == "mirror":
        prepare_mirror(args)
        return
    archive = args.archive or args.root / "GUE.archive"
    if not archive.exists():
        partial = archive.with_suffix(".partial")
        result = gdown.download(id=GUE_ID, output=str(partial), quiet=False)
        if result is None:
            raise RuntimeError("Official GUE download failed; pass --archive PATH to an existing GUE archive")
        partial.replace(archive)
    dest = args.root / "GUE" / "prom" / args.task
    dest.mkdir(parents=True, exist_ok=True)
    found = []

    def extract_selected(name, opener):
        parts = PurePosixPath(name.replace("\\", "/")).parts
        if len(parts) >= 3 and parts[-3:-1] == ("prom", args.task) and parts[-1] in {"train.csv", "dev.csv", "test.csv"}:
            if parts[-1] in found:
                raise ValueError("Archive contains duplicate split entries")
            target = dest / parts[-1]  # fixed allowlist, never extract arbitrary paths
            with opener() as source, target.open("wb") as output:
                shutil.copyfileobj(source, output)
            found.append(parts[-1])

    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as z:
            for entry in z.infolist():
                if not entry.is_dir():
                    extract_selected(entry.filename, lambda e=entry: z.open(e))
    else:
        with tarfile.open(archive) as t:
            for entry in t:
                if entry.isfile():
                    extract_selected(entry.name, lambda e=entry: t.extractfile(e))
    if set(found) != {"train.csv", "dev.csv", "test.csv"}:
        raise RuntimeError(f"Missing official split files for {args.task}: found {found}")
    def digest(path):
        h = hashlib.sha256()
        with path.open("rb") as f:
            for block in iter(lambda: f.read(1024 * 1024), b""):
                h.update(block)
        return h.hexdigest()
    manifest = {
        "source": f"https://drive.google.com/file/d/{GUE_ID}/view",
        "reference": "https://github.com/MAGICS-LAB/DNABERT_2",
        "archive_sha256": digest(archive), "task": args.task,
        "split_sha256": {name: digest(dest / name) for name in sorted(found)},
    }
    (dest / "source.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


def prepare_mirror(args):
    """Explicit fallback; require two independently hosted mirrors to agree."""
    import csv
    import io
    import urllib.request
    mirrors = [
        ("dnagpt/GUE", "e0bc5eec01b46aca338e34ed4ab8ba97e096fe3e", "prom"),
        ("leannmlindsey/GUE", "240df51a2cd2c0fb1d88efffaf86e8d1e477f932", "GUE"),
    ]
    dest = args.root / "GUE" / "prom" / args.task
    dest.mkdir(parents=True, exist_ok=True)
    report = {"source_type": "two_cross_checked_community_mirrors", "task": args.task,
              "official_source": f"https://drive.google.com/file/d/{GUE_ID}/view",
              "official_archive_verified": False, "splits": {}}
    expected = {"prom_300_tata": [4904, 613, 613], "prom_300_all": [47356, 5920, 5920],
                "prom_300_notata": [42452, 5307, 5307]}[args.task]
    for split, count in zip(["train", "dev", "test"], expected):
        copies, metadata = [], []
        for repo, rev, prefix in mirrors:
            url = f"https://huggingface.co/datasets/{repo}/resolve/{rev}/{prefix}/{args.task}/{split}.csv"
            with urllib.request.urlopen(url, timeout=60) as response:
                payload = response.read()
            rows = list(csv.DictReader(io.StringIO(payload.decode("utf-8-sig"))))
            normalized = [(r["sequence"].strip().upper(), int(r["label"])) for r in rows]
            copies.append((payload, normalized))
            metadata.append({"url": url, "sha256": hashlib.sha256(payload).hexdigest()})
        if copies[0][1] != copies[1][1] or len(copies[0][1]) != count:
            raise ValueError(f"{split}: mirrors disagree or row count differs from GUE paper")
        (dest / f"{split}.csv").write_bytes(copies[0][0])
        report["splits"][split] = {"rows": count, "ordered_rows_equal": True, "mirrors": metadata}
    (dest / "source.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
