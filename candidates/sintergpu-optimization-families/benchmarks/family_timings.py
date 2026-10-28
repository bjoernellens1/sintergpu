# SPDX-License-Identifier: Apache-2.0
"""Controlled, candidate-only ABBA benchmark. Never promotes a library route."""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import time

import numpy as np

from family_support import ROOT, CANDIDATE, point_case, solve_case, fusion_case, point_oracle, solve_oracle, fusion_oracle, check_point_oracle, check_solve_oracle, check_fusion_oracle

METHOD = dict(order=['baseline', 'candidate', 'candidate', 'baseline'],
    bootstrap='paired whole ABBA blocks; ratio of raw-sample quantiles',
    bootstrap_resamples=10000, bootstrap_seed=74511, upper_confidence=.95,
    median_upper_limit=1.03, p95_upper_limit=1.05,
    credible_slowdown_rejection='one-sided lower95 ratio >1 for median or p95',
    fusion_upper_median_limit=.95, fusion_upper_p95_limit=1.05,
    rule='both bounds must pass in every run, workload and applicable boundary; no aggregate wins')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def summary(samples):
    return dict(median_ms=float(np.median(samples)), p95_ms=float(np.quantile(samples, .95)),
        min_ms=float(np.min(samples)), max_ms=float(np.max(samples)))


def compare(blocks):
    blocks = np.asarray(blocks, np.float64)
    a, b = blocks[:, [0, 3]], blocks[:, [1, 2]]
    rng = np.random.default_rng(METHOD['bootstrap_seed'])
    # Chunk resampling to bound host allocation at large retained sample counts.
    median_ratios, tail_ratios = [], []
    for _ in range(100):
        index = rng.integers(0, len(blocks), (100, len(blocks)))
        old, new = a[index].reshape(100, -1), b[index].reshape(100, -1)
        median_ratios.extend(np.median(new, axis=1) / np.median(old, axis=1))
        tail_ratios.extend(np.quantile(new, .95, axis=1) / np.quantile(old, .95, axis=1))
    median_upper = float(np.quantile(median_ratios, .95))
    tail_upper = float(np.quantile(tail_ratios, .95))
    median_lower = float(np.quantile(median_ratios, .05))
    tail_lower = float(np.quantile(tail_ratios, .05))
    passed = median_upper <= METHOD['median_upper_limit'] and tail_upper <= METHOD['p95_upper_limit'] and median_lower<=1 and tail_lower<=1
    return dict(baseline=summary(a), candidate=summary(b),
        median_ratio=float(np.median(b)/np.median(a)), p95_ratio=float(np.quantile(b,.95)/np.quantile(a,.95)),
        median_ratio_upper95=median_upper, p95_ratio_upper95=tail_upper,
        median_ratio_lower95=median_lower, p95_ratio_lower95=tail_lower,
        decision=('faster' if median_upper < 1 else 'on-par') if passed else 'retain-baseline')


def verify(case):
    actual = {name: case['read'](name) for name in ('baseline', 'candidate')}
    for key in actual['baseline']:
        np.testing.assert_array_equal(actual['candidate'][key].view(np.uint32), actual['baseline'][key].view(np.uint32))
    dict(point_plane=check_point_oracle, small_cholesky=check_solve_oracle,
        point_plane_gn=check_fusion_oracle)[case['family']](case, actual['candidate'])


def measure(case, blocks, warmups):
    import slangpy as spy
    device = case['device']
    verify(case)
    for _ in range(warmups):
        for variant in METHOD['order']:
            case['dispatch'](variant)
            device.wait()
    oracle = dict(point_plane=point_oracle, small_cholesky=solve_oracle, point_plane_gn=fusion_oracle)[case['family']]
    for _ in range(warmups): oracle(case)
    cpu = []
    for _ in range(blocks):
        start = time.perf_counter_ns(); oracle(case); cpu.append((time.perf_counter_ns()-start)/1e6)
    pool = device.create_query_pool(spy.QueryType.timestamp, 2)
    gpu_metrics = ('gpu_producer_pipeline_ms','gpu_resident_pipeline_ms') if case['family']=='point_plane_gn' else ('gpu_kernel_ms',)
    metrics = {name: [] for name in (*gpu_metrics, 'resident_complete_family_ms', 'transfer_inclusive_ms')}
    for metric in metrics:
        for _ in range(warmups):
            for variant in METHOD['order']:
                if metric=='transfer_inclusive_ms': case['upload'](variant)
                case['dispatch'](variant); device.wait()
                if metric=='transfer_inclusive_ms': case['download'](variant)
        for _ in range(blocks):
            samples = []
            for variant in METHOD['order']:
                if metric in gpu_metrics:
                    pool.reset()
                    kwargs=dict(region='producer') if metric=='gpu_producer_pipeline_ms' else {}
                    case['dispatch'](variant, query_pool=pool, query_index_before=0, query_index_after=1, **kwargs)
                    device.wait()
                    stamps = pool.get_timestamp_results(0, 2)
                    value = (stamps[1]-stamps[0])*1000
                else:
                    start = time.perf_counter_ns()
                    if metric=='transfer_inclusive_ms': case['upload'](variant)
                    case['dispatch'](variant); device.wait()
                    if metric=='transfer_inclusive_ms': case['download'](variant)
                    value = (time.perf_counter_ns()-start)/1e6
                if not np.isfinite(value) or value <= 0: raise RuntimeError('invalid timing')
                samples.append(value)
            metrics[metric].append(samples)
    verify(case)
    identities = {name: hashlib.sha256(source.encode()).hexdigest() for name, source in case['sources'].items()}
    workload_hash = hashlib.sha256()
    for key, array in sorted(case['arrays'].items()):
        workload_hash.update(key.encode()); workload_hash.update(array.tobytes())
    acceptance={metric:compare(values) for metric,values in metrics.items()}
    if case['family']=='point_plane_gn':
        for metric,result in acceptance.items():
            result['fusion_five_percent_gate']=result['median_ratio_upper95']<=.95 and result['p95_ratio_upper95']<=1.05
            if not result['fusion_five_percent_gate']:result['decision']='retain-baseline'
    return dict(family=case['family'], count=case.get('count'), dimension=case.get('dimension'),
        profile=case.get('profile', 'schur' if case.get('schur') else 'caspar6'), kind=case.get('kind'),
        source_sha256=identities, workload_sha256=workload_hash.hexdigest(),
        uploaded_bytes=case['uploaded_bytes'], readback_bytes=case['readback_bytes'],
        allocation='identical preallocated input/output buffers per variant; excluded from timing',
        dispatch_counts=case.get('dispatch_counts', dict(baseline=1,candidate=1)),
        synchronization='one submission and completion wait per variant per sample; readback only in transfer boundary',
        cpu_reference_wall_ms=cpu, cpu_reference_summary=summary(cpu), samples=metrics,
        acceptance=acceptance,
        correctness='exact frozen-baseline full-output/padding differential plus independent float64 oracle before and after sampling')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--blocks', type=int, default=100)
    parser.add_argument('--warmups', type=int, default=10)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--fusion-only', action='store_true')
    parser.add_argument('--target',choices=('radeon8060s','rtx4070'),default='radeon8060s')
    args = parser.parse_args()
    if os.getenv('SINTERGPU_GPU_TESTS') != '1': raise RuntimeError('SINTERGPU_GPU_TESTS=1 required')
    if args.blocks < 1 or args.warmups < 1: raise ValueError('positive block and warmup counts required')
    manifest = json.loads((CANDIDATE/'candidate.json').read_text())
    for source in manifest['baseline_sources']:
        if sha(ROOT/source['path']) != source['sha256']: raise RuntimeError('frozen baseline source changed')
    lock = open('/tmp/sintergpu-gpu.lock', 'a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    owner = dict(pid=os.getpid(), host=platform.node(), started_utc=datetime.now(timezone.utc).isoformat(),
        worktree=str(ROOT), command='family_timings.py', output=str(args.output.resolve()))
    owner_path = Path('/tmp/sintergpu-gpu-owner.json')
    owner_path.write_text(json.dumps(owner, indent=2)+'\n')
    device = None
    try:
        import slangpy as spy
        adapters = spy.Device.enumerate_adapters(spy.DeviceType.vulkan)
        if args.target=='radeon8060s':
            adapter=next(a for a in adapters if a.vendor_id==0x1002 and a.device_id==0x1586)
        else:
            adapter=next(a for a in adapters if a.vendor_id==0x10de and a.name.endswith('GeForce RTX 4070 Ti'))
        device = spy.Device(type=spy.DeviceType.vulkan, adapter_luid=adapter.luid,
            compiler_options={'floating_point_mode': spy.SlangFloatingPointMode.precise})
        compilers = list(Path(spy.__file__).parent.glob('libslang-compiler*'))
        if len(compilers)!=1: raise RuntimeError('ambiguous compiler binary identity')
        environment = dict(host=platform.node(), gpu=device.info.adapter_name, vendor_id=adapter.vendor_id,
            device_id=adapter.device_id, api=device.info.api_name, compiler_sha256=sha(compilers[0]),
            slangpy=importlib.metadata.version('slangpy'), numpy=np.__version__, floating_point_mode='precise',
            timestamp_frequency=int(device.info.timestamp_frequency), python=platform.python_version(),
            cpu_affinity=sorted(os.sched_getaffinity(0)), uname=list(platform.uname()), ownership=owner,
            cpuinfo=Path('/proc/cpuinfo').read_text().split('\n\n')[0])
        environment['target']=args.target
        environment['gpu_busy_before']={str(p):p.read_text().strip() for p in Path('/sys/class/drm').glob('card*/device/gpu_busy_percent')}
        users=subprocess.run(['fuser','/dev/dri/renderD128'],capture_output=True,text=True)
        pids=[value for value in users.stdout.split() if value.isdigit()]
        environment['render_node_users']=[]
        for pid in pids:
            path=Path('/proc')/pid/'comm'
            try:environment['render_node_users'].append(dict(pid=int(pid),command=path.read_text().strip()))
            except OSError:pass
        try:
            result = subprocess.run(['vulkaninfo','--summary'], capture_output=True,text=True,timeout=20)
            environment['vulkaninfo_summary'] = result.stdout or result.stderr
        except (OSError,subprocess.TimeoutExpired) as error: environment['vulkaninfo_summary'] = str(error)
        record = dict(schema_version=1, base_revision=manifest['base_revision'], method=METHOD,
            environment=environment, blocks=args.blocks, warmups=args.warmups,
            runner_sha256=sha(__file__), support_sha256=sha(CANDIDATE/'benchmarks/family_support.py'),
            candidate_sources={str(path.relative_to(ROOT)):sha(path) for path in sorted((CANDIDATE/'kernels').rglob('*.slang'))},
            scope='complete residual/Jacobian, reduced-system solve, or point-plane GN step including all outputs; no tracking/application pipeline claim',
            measurements=[])
        record['candidate_revision']=os.getenv('SINTERGPU_CANDIDATE_REVISION') or subprocess.check_output(
            ['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()
        jobs = [('point', count, profile) for count in ([1025] if args.smoke else [1,63,1023,1024,1025,4099,65539]) for profile in ('aosoa','scalar')]
        jobs += [('solve', n, schur) for n,schur in ([(6,False)] if args.smoke else [(6,False),(6,True),(12,True),(18,True)])]
        if args.fusion_only: jobs=[]
        jobs += [('fusion', count, tree) for count in ([1025] if args.smoke else [1,63,64,65,1025,4099,65539]) for tree in ('workgroup','subgroup')]
        args.output.parent.mkdir(parents=True, exist_ok=True)
        for family, size, profile in jobs:
            case = dict(point=lambda:point_case(device,size,profile),solve=lambda:solve_case(device,size,schur=profile),
                fusion=lambda:fusion_case(device,size,profile))[family]()
            row = measure(case,args.blocks,args.warmups)
            record['measurements'].append(row)
            args.output.write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
            print(json.dumps({k:row[k] for k in ('family','count','dimension','profile')}), flush=True)
            del case
        record['completed_utc']=datetime.now(timezone.utc).isoformat()
        record['gpu_busy_after']={str(p):p.read_text().strip() for p in Path('/sys/class/drm').glob('card*/device/gpu_busy_percent')}
        args.output.write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
    finally:
        if device is not None: device.close()
        owner['finished_utc'] = datetime.now(timezone.utc).isoformat()
        owner_path.write_text(json.dumps(owner,indent=2)+'\n')
        lock.close()


if __name__ == '__main__': main()
