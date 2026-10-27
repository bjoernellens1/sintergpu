#!/usr/bin/env python3
"""Check the shared reflection ABI under both pinned Slang compilers."""
import argparse
import hashlib
import json
from pathlib import Path
import re

ROOT=Path(__file__).resolve().parents[1]
TYPE_KINDS={'scalar','vector','matrix','struct','array','resource','constantBuffer'}


def compare_records(left,right,catalog):
    """Fail closed on layout/source changes; permit only known schema additions."""
    for record,pin in ((left,'2026.12'),(right,'2026.19')):
        if record.get('status')!='passed' or record.get('pinned_release')!=pin or record.get('compiler_version')!=pin or not re.fullmatch('[a-f0-9]{64}',record.get('compiler_sha256','')):
            raise ValueError('pinned compiler identity or compilation failure')
        entries=record.get('entries',[])
        actual={e['operation']:e['source_sha256'] for e in entries}
        if not catalog or actual!=catalog or len(actual)!=len(entries) or any(e.get('returncode')!=0 or not e.get('reflection') for e in entries):
            raise ValueError('catalog source coverage or reflection failure')
    if left.get('version')!=right.get('version') or left.get('target')!=right.get('target') or left.get('target')!='spirv_1_5':
        raise ValueError('package version or target mismatch')
    added={}
    def compare(a,b,path=()):
        location='/'.join(map(str,path)) or 'root'
        if type(a) is not type(b):raise ValueError('reflection type mismatch: '+location)
        if isinstance(a,dict):
            for key in a:
                if key not in b:raise ValueError('missing reflected field: '+location+'/'+key)
                compare(a[key],b[key],path+(key,))
            for key in b.keys()-a.keys():
                known=(not path and ((key=='version' and b[key]=='1.1') or (key=='globalScope' and isinstance(b[key],dict))))
                known|=(len(path)==2 and path[0]=='entryPoints' and key=='scope' and isinstance(b[key],dict))
                known|=(key=='sizes' and a.get('kind') in TYPE_KINDS and isinstance(b[key],list))
                if not known:raise ValueError('unreviewed reflection schema addition: '+location+'/'+key)
                added[key]=added.get(key,0)+1
        elif isinstance(a,list):
            if len(a)!=len(b):raise ValueError('reflection array length mismatch: '+location)
            for i,(x,y) in enumerate(zip(a,b)):compare(x,y,path+(i,))
        elif a!=b:raise ValueError('reflection value mismatch: '+location)
    newer={e['operation']:e for e in right['entries']}
    for entry in left['entries']:
        try:
            compare(entry['reflection'],newer[entry['operation']]['reflection'])
        except ValueError as error:
            raise ValueError(entry['operation']+': '+str(error)) from error
    return {'status':'passed','entries':len(catalog),'scope':'All 2026.12 reflection fields match 2026.19 exactly, including types, bindings, offsets, sizes, strides, ordering and workgroups. New-only metadata is retained in compiler records but has no cross-version counterpart.','known_schema_additions':added,'compiler_sha256':{r['pinned_release']:r['compiler_sha256'] for r in (left,right)}}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--left',type=Path,required=True);p.add_argument('--right',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    left=json.loads(args.left.read_text());right=json.loads(args.right.read_text())
    catalog={o['id']:o['sha256'] for o in json.loads((ROOT/'src/sintergpu/data/catalog.json').read_text())['operations']}
    result=compare_records(left,right,catalog)
    result['inputs_sha256']={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in (args.left,args.right)}
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'status':result['status'],'entries':result['entries']}))


if __name__=='__main__':main()
