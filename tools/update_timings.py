#!/usr/bin/env python3
"""Build the source-bound timing registry and searchable table from raw runs."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from sintergpu import __version__
from sintergpu.catalog import list_operations
from sintergpu.timings import build_registry,render_markdown


def serialize_registry(database):
    """Deterministic compact JSON preserves the public timing schema and samples."""
    return json.dumps(database, separators=(',', ':'), allow_nan=False) + '\n'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,action='append',default=[])
    p.add_argument('--database',type=Path,default=ROOT/'src/sintergpu/data/timings.json')
    p.add_argument('--markdown',type=Path,default=ROOT/'docs/kernel-timings.md')
    p.add_argument('--check',action='store_true');args=p.parse_args()
    records=[]
    if args.database.is_file():records.extend(json.loads(args.database.read_text())['measurements'])
    for path in args.input:records.extend(json.loads(path.read_text())['measurements'])
    unique={}
    for item in records:
        item=dict(item);item.pop('current_source',None)
        identity=hashlib.sha256(json.dumps(item,sort_keys=True,allow_nan=False).encode()).hexdigest()
        unique.setdefault(identity,item)
    database=build_registry(list_operations(),list(unique.values()));database['library_version']=__version__
    database['additional_pending_targets']=[]
    for name in ('upsweep','spine','downsweep','downsweep_key_value'):
        source=ROOT/'third_party/vrdx/src/shader'/('downsweep.slang' if name=='downsweep_key_value' else name+'.slang')
        database['additional_pending_targets'].append({'operation':'vrdx.'+name,'variant':'optional-native-key-value' if name=='downsweep_key_value' else 'optional-native','source_path':source.relative_to(ROOT).as_posix(),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'status':'pending','reason':'Optional native Vulkan runtime/standalone CPU-GPU timing recipe pending; C++ build only is not a timing measurement.'})
    payload=serialize_registry(database);markdown=render_markdown(database)
    if args.check:
        if not args.database.is_file() or args.database.read_text()!=payload or not args.markdown.is_file() or args.markdown.read_text()!=markdown:
            raise SystemExit('timing registry/table is stale; run tools/update_timings.py')
    else:
        for path,text in ((args.database,payload),(args.markdown,markdown)):
            path.parent.mkdir(parents=True,exist_ok=True);path.write_text(text)
    print(json.dumps(database['coverage']))


if __name__=='__main__':main()
