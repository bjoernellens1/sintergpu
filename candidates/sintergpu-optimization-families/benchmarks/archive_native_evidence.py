# SPDX-License-Identifier: Apache-2.0
"""Losslessly archive native samples and deduplicate exact fixture/output bytes.

The manifest preserves each original logical path, byte count and SHA256.
restore recreates all original files; no sample or failure is removed.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path


def sha(data):return hashlib.sha256(data).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('archive','restore','verify'))
    parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args();root=args.root.resolve();evidence=root/'evidence'
    manifest=evidence/'native-archive.json';records=json.loads(manifest.read_text()) if manifest.exists() else {}
    if args.action=='archive':
        for iteration in ('iteration4','iteration5','iteration6','iteration7'):
            directory=evidence/iteration
            for path in sorted(directory.rglob('*')):
                if not path.is_file():continue
                binary=path.suffix=='.bin'
                raw=path.suffix=='.json' and '-run' in path.name
                if not binary and not raw:continue
                data=path.read_bytes();digest=sha(data)
                target=evidence/'native-blobs'/(digest+'.bin.gz') if binary else path.with_suffix('.json.gz')
                target.parent.mkdir(parents=True,exist_ok=True)
                compressed=gzip.compress(data,mtime=0)
                if target.exists():assert gzip.decompress(target.read_bytes())==data
                else:target.write_bytes(compressed)
                records[str(path.relative_to(root))]=dict(bytes=len(data),sha256=digest,
                    archive=str(target.relative_to(root)),archive_sha256=sha(target.read_bytes()))
                # Persist recovery metadata before removing the original.
                manifest.write_text(json.dumps(records,indent=2,sort_keys=True)+'\n')
                path.unlink()
    for logical,entry in records.items():
        archive=root/entry['archive'];compressed=archive.read_bytes()
        assert sha(compressed)==entry['archive_sha256']
        data=gzip.decompress(compressed)
        assert sha(data)==entry['sha256'] and len(data)==entry['bytes']
        if args.action=='restore':
            target=root/logical;target.parent.mkdir(parents=True,exist_ok=True)
            if target.exists():assert target.read_bytes()==data
            else:target.write_bytes(data)
    print(args.action,len(records),'logical files; every original SHA256 verified')


if __name__=='__main__':main()
