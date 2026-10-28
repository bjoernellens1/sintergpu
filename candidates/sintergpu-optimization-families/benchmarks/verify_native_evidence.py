# SPDX-License-Identifier: Apache-2.0
"""Verify archived native provenance, independent outputs and invocation identity.

This CPU-only audit does not recompute bootstrap decisions or qualify GPU timing.
Run from the candidate directory after restoring or archiving the native files.
"""
import gzip
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

from native_fixture import check_point_oracle, point_case

ROOT=Path(__file__).resolve().parents[1]


def digest(data):return hashlib.sha256(data).hexdigest()


def fnv(data):
    value=14695981039346656037
    for byte in data:value=((value^byte)*1099511628211)&((1<<64)-1)
    return format(value,'x')


def main():
    archive_path=ROOT/'evidence/native-archive.json'
    archive=json.loads(archive_path.read_text()) if archive_path.exists() else {}
    def read(path):
        if path.exists():return path.read_bytes()
        logical=str(path.relative_to(ROOT))
        if logical in archive:
            entry=archive[logical];packed=(ROOT/entry['archive']).read_bytes()
            assert digest(packed)==entry['archive_sha256']
            data=gzip.decompress(packed)
            assert digest(data)==entry['sha256'] and len(data)==entry['bytes']
            return data
        return gzip.decompress(path.with_suffix(path.suffix+'.gz').read_bytes())
    runtimes=json.loads((ROOT/'evidence/native-runtime.json').read_text())
    runtime_fnv={}
    for expected,entry in runtimes.items():
        data=read(ROOT/entry['path']);assert digest(data)==expected
        runtime_fnv[expected]=fnv(data)
    runs=0
    for path in sorted((ROOT/'evidence').glob('iteration*/native-*/suite-plan.json')):
        plan=json.loads(path.read_text());pids={}
        assert len(plan['results'])==len(plan['counts'])*len(plan['profiles'])*3
        for name,expected in plan['sources'].items():
            source=subprocess.check_output(['git','show',plan['source_commit']+':candidates/sintergpu-optimization-families/'+name],cwd=ROOT)
            assert digest(source)==expected,(path,name)
        for result in plan['results']:
            raw_bytes=read(path.parent/result['output']);assert digest(raw_bytes)==result['sha256']
            raw=json.loads(raw_bytes)
            assert raw['status']=='passed'
            assert raw['count']==result['count'] and raw['blocks']==plan['blocks']
            assert raw['host']['cpu_affinity']==[plan['affinity_cpu']]
            assert raw['hardware']['vendor_id']==int(plan['vendor'],0)
            assert raw['hardware']['device_id']==int(plan['device'],0)
            assert len(raw['samples'])==12*plan['blocks']
            for boundary in ('gpu_timestamp','resident_completion','transfer_inclusive'):
                samples=[s for s in raw['samples'] if s['boundary']==boundary]
                assert len(samples)==4*plan['blocks']
                for i,sample in enumerate(samples):
                    assert sample['block']==i//4 and sample['position']==i%4
                    assert sample['variant']==('baseline','candidate','candidate','baseline')[i%4]
                    assert sample['wall_ns']>0
            if plan['native_binary_sha256'] in runtime_fnv:
                assert raw['identity']['native_executable']==runtime_fnv[plan['native_binary_sha256']]
            if 'candidate_local_size' in plan:
                assert raw['completion_mode']==plan['completion_mode']
                for variant,size in [('baseline',1024),('candidate',plan['candidate_local_size'])]:
                    assert raw['launch'][variant+'_local_size']==[size,1,1]
                    assert raw['launch'][variant+'_groups_x']==(raw['count']+size-1)//size
            key=(result['profile'],result['count']);pids.setdefault(key,set())
            assert raw['host']['pid'] not in pids[key];pids[key].add(raw['host']['pid'])
            command=result['command'];compiled=ROOT/command[command.index('--baseline')+1]
            for name,expected in plan['artifacts'].items():assert digest(read(compiled.parent/name))==expected
            fixture=ROOT/command[command.index('--fixture-dir')+1]
            for name,expected in result['fixtures'].items():assert digest(read(fixture/name))==expected
            for name,expected in result['dumps'].items():assert digest(read(path.parent/name))==expected
            case=point_case(None,result['count'],result['profile'])
            for phase in ('before','after'):
                outputs={}
                for variant in ('baseline','candidate'):
                    outputs[variant]={name:np.frombuffer(read(path.parent/(result['output']+'.'+phase+'.'+variant+'.'+name+'.bin')),dtype='<f4') for name in case['outputs']}
                    check_point_oracle(case,outputs[variant])
                for name in case['outputs']:assert outputs['baseline'][name].tobytes()==outputs['candidate'][name].tobytes()
            runs+=1
        assert all(len(p)==3 for p in pids.values())
        print(path.relative_to(ROOT),'source/artifacts/fixtures/outputs/oracles/independent processes PASS',flush=True)
    print(runs,'completed native processes independently audited; bootstrap decisions remain separate')


if __name__=='__main__':main()
