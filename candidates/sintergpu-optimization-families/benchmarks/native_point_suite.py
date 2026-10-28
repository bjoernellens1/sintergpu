# SPDX-License-Identifier: Apache-2.0
"""Run and report the separate native caller-owned-buffer point profile.

Each configuration has three fresh native processes. Reports apply the existing
unchanged uncertainty gates independently to all three measured boundaries.
"""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess

from family_timings import METHOD, compare
from native_fixture import COUNTS, check_native_outputs, export_fixture


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(args):
    args.output.mkdir(parents=True, exist_ok=True)
    identity = dict(schema='sintergpu.native.suite.v1', blocks=args.blocks, warmups=10,
        runs=3, profiles=['aosoa', 'scalar'], counts=list(COUNTS), method=METHOD,
        candidate_local_size=args.candidate_local_size,completion_mode=args.completion_mode,
        compiler_lane='2026.19', hostname=platform.node(), vendor=args.vendor,
        device=args.device, affinity_cpu=args.cpu,
        source_commit=args.source_commit or subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        source_commit_origin='explicit archive identity' if args.source_commit else 'local git HEAD',
        native_binary_sha256=sha(args.binary), native_build_flags='-std=c++17 -O2 -Wall -Wextra -Werror -Wno-missing-field-initializers -lvulkan',
        sources={str(p.relative_to(Path(__file__).resolve().parents[1])):sha(p)
                 for p in [Path(__file__).resolve(), Path(__file__).with_name('native_fixture.py'),
                           Path(__file__).with_name('inlined_point_support.py'),
                           Path(__file__).resolve().parents[1]/'native/point_plane_ab.cpp',
                           Path(__file__).resolve().parents[1]/'kernels/optimization/point_plane_inlined.slang']},
        artifacts={p.name:sha(p) for p in args.compiled.glob('*') if p.is_file()}, results=[])
    (args.output/'suite-plan.json').write_text(json.dumps(identity, indent=2)+'\n')
    for profile in identity['profiles']:
        for count in COUNTS:
            fixture=args.output/f'fixture-{profile}-{count}'
            export_fixture(fixture, count, profile,args.candidate_local_size)
            for run_id in range(1,4):
                output=args.output/f'{profile}-{count}-run{run_id}.json'
                if output.exists() or output.with_suffix('.json.gz').exists():
                    raise ValueError('refusing to overwrite an existing run: '+str(output))
                command=['taskset','-c',str(args.cpu),str(args.binary),
                    '--baseline',str(args.compiled/f'{profile}-baseline.spv'),
                    '--candidate',str(args.compiled/f'{profile}-candidate.spv'),
                    '--fixture-dir',str(fixture),'--count',str(count),
                    '--vendor',args.vendor,'--device',args.device,'--output',str(output),
                    '--blocks',str(args.blocks),'--warmups','10','--completion-mode',args.completion_mode]
                subprocess.run(command, check=True)
                oracle=check_native_outputs(output, count, profile)
                result=dict(profile=profile,count=count,run=run_id,command=command,
                    output=output.name,sha256=sha(output),oracle=oracle,
                    fixtures={p.name:sha(p) for p in fixture.glob('*')},
                    dumps={p.name:sha(p) for p in args.output.glob(output.name+'.*.bin')})
                identity['results'].append(result)
                (args.output/'suite-plan.json').write_text(json.dumps(identity, indent=2)+'\n')
                print(profile,count,run_id,'byte-exact and independent oracle passed',flush=True)


def report(args):
    reports=[]
    for directory in args.inputs:
        plan=json.loads((directory/'suite-plan.json').read_text())
        assert len(plan['results'])==42 and plan['blocks']==1000 and plan['runs']==3
        device=dict(host=plan['hostname'],vendor=plan['vendor'],device=plan['device'],results=[])
        for result in plan['results']:
            path=directory/result['output']
            data=path.read_bytes() if path.exists() else gzip.decompress(path.with_suffix('.json.gz').read_bytes())
            assert hashlib.sha256(data).hexdigest()==result['sha256']
            raw=json.loads(data)
            assert raw['status']=='passed' and raw['blocks']==1000 and raw['warmups_per_boundary_per_variant']==10
            assert result['oracle']['independent_float64_oracle'] and result['oracle']['full_byte_differential']
            row={key:result[key] for key in ('profile','count','run')};row['boundaries']={}
            for boundary in ('gpu_timestamp','resident_completion','transfer_inclusive'):
                samples=[s for s in raw['samples'] if s['boundary']==boundary]
                assert len(samples)==4000
                blocks=[]
                for index in range(1000):
                    group=samples[4*index:4*index+4]
                    assert [s['variant'] for s in group]==METHOD['order']
                    assert [s['position'] for s in group]==[0,1,2,3]
                    assert all(s['block']==index for s in group)
                    key='gpu_ns' if boundary=='gpu_timestamp' else 'wall_ns'
                    blocks.append([s[key]/1e6 for s in group])
                row['boundaries'][boundary]=compare(blocks)
            row['all_boundaries_pass']=all(b['decision']!='retain-baseline' for b in row['boundaries'].values())
            device['results'].append(row)
            print(plan['hostname'],row['profile'],row['count'],row['run'],row['all_boundaries_pass'],flush=True)
        device['all_configurations_pass']=all(r['all_boundaries_pass'] for r in device['results'])
        reports.append(device)
    output=dict(schema='sintergpu.native.suite.report.v1',method=METHOD,devices=reports,
        all_configurations_pass=all(d['all_configurations_pass'] for d in reports),
        scope='Separate native caller-owned Vulkan profile only; existing Python verdicts remain unchanged.')
    args.output.write_text(json.dumps(output,indent=2)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='action',required=True)
    p=sub.add_parser('run');p.add_argument('--binary',type=Path,required=True);p.add_argument('--compiled',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--vendor',required=True);p.add_argument('--device',required=True)
    p.add_argument('--cpu',type=int,required=True);p.add_argument('--blocks',type=int,default=1000)
    p.add_argument('--source-commit',help='Full source commit for an exported archive without .git; source hashes remain mandatory')
    p.add_argument('--candidate-local-size',type=int,choices=(256,1024),default=1024)
    p.add_argument('--completion-mode',choices=('wait','spin'),default='wait')
    p=sub.add_parser('report');p.add_argument('--inputs',type=Path,nargs='+',required=True);p.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();globals()[args.action](args)


if __name__=='__main__':main()
