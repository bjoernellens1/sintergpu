# SPDX-License-Identifier: Apache-2.0
"""Validate retained run identities and render conservative per-config gate tables."""
import hashlib
import gzip
import json
from pathlib import Path

CANDIDATE=Path(__file__).resolve().parents[1]
ROOT=CANDIDATE.parents[1]


def read_json(path):
    if path.is_file():return json.loads(path.read_text())
    return json.loads(gzip.decompress(path.with_suffix(path.suffix+'.gz').read_bytes()))


def key(row):
    return (row['family'],row['profile'],row['count'] or row['dimension'])


def main():
    directory=CANDIDATE/'evidence/iteration3'
    by_device={}
    for name in ('radeon8060s','rtx4070ti'):
        runs=[read_json(directory/f'{name}-run-{i}.json') for i in (1,2,3)]
        for run in runs:
            assert run.get('completed_utc') and len(run['measurements'])==32
            assert run['blocks']==100 and run['warmups']==10
            assert run['candidate_revision']=='bebbbe235836eb944f36861609d4c47ed3995336'
            for path,digest in run['candidate_sources'].items():
                assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==digest,path
            for field,path in [('runner_sha256',CANDIDATE/'benchmarks/family_timings.py'),('support_sha256',CANDIDATE/'benchmarks/family_support.py')]:
                assert hashlib.sha256(path.read_bytes()).hexdigest()==run[field],field
        assert len({r['environment']['compiler_sha256'] for r in runs})==1
        keyed=[{key(row):row for row in run['measurements']} for run in runs]
        assert all(set(rows)==set(keyed[0]) for rows in keyed)
        for k in keyed[0]:
            assert len({rows[k]['workload_sha256'] for rows in keyed})==1
            assert all(rows[k]['source_sha256']==keyed[0][k]['source_sha256'] for rows in keyed)
        by_device[name]=dict(runs=runs,keyed=keyed)
    lines=['# Candidate acceptance: retained incumbents', '',
        'Source revision: `bebbbe235836eb944f36861609d4c47ed3995336`. These results',
        'cover the exact seeded workloads and configurations below. They do not qualify',
        'a range of sizes, production tracking, gradients beyond the stated Jacobian,',
        'installed APIs or consumer migration. No family passes every applicable boundary;',
        '**all incumbents remain available and no promotion/default change is requested.**', '',
        'Each cell is candidate/baseline median ratio for runs1/2/3, followed by p95',
        'ratios in parentheses. Below1 is faster. Decisions use the declared paired-block',
        'bootstrap bounds, including credible-slowdown rejection, not these point estimates.',
        'Fusion additionally requires an upper median ratio <=0.95 at each relevant',
        'pipeline boundary. A resident-only win never qualifies a CPU-export route.', '',
        'Final numerical validation: 64 tests on each device. Both Slang2026.12 and',
        '2026.19 compile15 profiles each; cross-compiler reflection and baseline/candidate',
        'resource layouts match. Library CPU suite on the recorded v0.1.9 base:',
        '259 passed,290 skipped,142 subtests, one existing pytest fixture deprecation warning.', '',
        'Raw samples retain CPU-reference wall, GPU kernel/producer/full-pipeline',
        'timestamps, resident host wall and transfer-inclusive wall separately. The JSON',
        'below additionally records raw baseline/candidate medians and p95s in milliseconds,',
        'each bound, source/compiler/device identity and exact workload hashes.', '',
        '- [Machine-readable per-configuration acceptance](evidence/iteration3/acceptance.json)',
        '- [Numerical tests on Radeon](evidence/iteration3/numerical.log)',
        '- [Numerical tests on selected RTX4070Ti](evidence/iteration3/rtx4070ti-numerical.log)',
        '- [Pinned compiler/reflection compatibility](evidence/iteration3/compilers/compatibility.json)',
        '- [Baseline/candidate resource layouts](evidence/iteration3/baseline-candidate-reflection.json)', '',
        'Historical iteration1 and iteration2 rejections are retained with their original',
        'method and source identities. No historical result is relabeled as final.', '']
    report=dict(schema_version=1,source_revision='bebbbe235836eb944f36861609d4c47ed3995336',
        promotion='none; retain all incumbents',configurations=[])
    for device,data in by_device.items():
        env=data['runs'][0]['environment']
        lines += ['## '+device,'',f"{env['gpu']}; vendor0x{env['vendor_id']:04x}/device0x{env['device_id']:04x}; SlangPy{env['slangpy']}; precise FP mode.",
            '', 'Raw runs (lossless gzip JSON): '+', '.join(f'[run{i}](evidence/iteration3/{device}-run-{i}.json.gz)' for i in (1,2,3))+'.', '']
        if device=='rtx4070ti':
            lines += ['Canonical host `cps-wkstn-nvidia4070`, GPU UUID',
                '`GPU-96d2dba0-95fd-5c00-e849-2e0610f24e06`, driver595.99.02.',
                'The earlier unreachable RTX3090 is historical and was replaced by explicit user steering.', '']
        for family in ('point_plane','small_cholesky','point_plane_gn'):
            lines += ['### '+family,'',
                '| Profile / exact size | GPU ratio median (p95) | Resident wall ratio median (p95) | Transfer ratio median (p95) | GPU+resident gate / all-boundary gate |',
                '| --- | --- | --- | --- | --- |']
            for k in data['keyed'][0]:
                if k[0]!=family: continue
                rows=[r[k] for r in data['keyed']]
                metric='gpu_resident_pipeline_ms' if family=='point_plane_gn' else 'gpu_kernel_ms'
                boundaries=(metric,'resident_complete_family_ms','transfer_inclusive_ms')
                def values(name):
                    return '/'.join(f"{r['acceptance'][name]['median_ratio']:.3f}" for r in rows)+' ('+'/'.join(f"{r['acceptance'][name]['p95_ratio']:.3f}" for r in rows)+')'
                resident=all(r['acceptance'][m]['decision']!='retain-baseline' for r in rows for m in boundaries[:2])
                complete=all(a['decision']!='retain-baseline' for r in rows for a in r['acceptance'].values())
                decision=('PASS' if resident else 'retain')+' / '+('PASS' if complete else 'retain')
                lines.append('| '+k[1]+' / '+str(k[2])+' | '+values(metric)+' | '+values(boundaries[1])+' | '+values(boundaries[2])+' | '+decision+' |')
                report['configurations'].append(dict(device=device,family=family,profile=k[1],size=k[2],
                    resident_gate=resident,all_boundary_gate=complete,decision='retain-baseline' if not complete else 'candidate-only-pass',
                    runs=[dict(run=i+1,acceptance=r['acceptance'],cpu_reference_summary=r['cpu_reference_summary']) for i,r in enumerate(rows)]))
            lines.append('')
    lines += ['## Interpretation and pending work','',
        'The fused producer and full GN step produce real resident GPU wins in measured',
        'large configurations on both targets while preserving exact full outputs. Those',
        'wins do not overcome transfer cost or noise in every supported configuration.',
        'The generic point-plane and solver helpers therefore remain candidate implementations.',
        'The checked solver convenience profile has direct numerical tests but no timing',
        'qualification. Preserve specialized incumbents for every failed/noisy/unmeasured',
        'configuration; do not infer a size threshold or choose a default from this table.', '',
        'Further work: stabilize/understand host/transfer tails, optimize remaining regressing',
        'solver cases without changing loop semantics, independently qualify any new profile',
        'or workload, and request maintainer review before a separate promotion PR.', '']
    (directory/'acceptance.json').write_text(json.dumps(report,indent=2)+'\n')
    (CANDIDATE/'ACCEPTANCE.md').write_text('\n'.join(lines))
    print(json.dumps(dict(configurations=len(report['configurations']),resident_passes=sum(c['resident_gate'] for c in report['configurations']),
        all_boundary_passes=sum(c['all_boundary_gate'] for c in report['configurations']))))


if __name__=='__main__':main()
