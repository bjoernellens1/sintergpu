# SPDX-License-Identifier: Apache-2.0
"""Unqualified paired single-dispatch measurements on caller-prepared fixtures.

The caller owns GPU locks, device selection, identity validation and output
oracle. No resident pyramid or tracking measurement is implemented here.
"""
from __future__ import annotations
import copy
import hashlib
import time
from pathlib import Path
import numpy as np
from benchmarks.depth_pyramid_gpu_fixture import PADDING, ROOT

LABELS = (('old_before','old'), ('candidate','candidate'), ('old_after','old'))
BOUNDARIES = ('isolated_gpu_ms','host_complete_ms','transfer_inclusive_ms')


def _same(a,b):
    if set(a) != set(b): return False
    return all(a[k].shape == b[k].shape and a[k].dtype == b[k].dtype and
               np.array_equal(a[k].view(np.uint8),b[k].view(np.uint8)) for k in a)


def _workload(spec, initial):
    fields = ('profile', 'params', 'shape', 'groups', 'entry', 'parameter', 'threads', 'count')
    if any(k not in spec for k in fields): raise ValueError('incomplete dispatch contract')
    if set(initial) != set(spec['inputs']) | set(spec['outputs']): raise ValueError('incomplete buffer contract')
    inputs = {}
    for name, value in spec['inputs'].items():
        a = np.asarray(value)
        if not a.flags.c_contiguous: raise ValueError('contiguous inputs required')
        initial_array=np.asarray(initial[name])
        if initial_array.dtype != a.dtype or initial_array.size != a.size + PADDING or not np.array_equal(initial_array[:a.size].view(np.uint8),a.ravel().view(np.uint8)):
            raise ValueError('input initial contents mismatch')
        inputs[name]={'shape':list(a.shape),'dtype':a.dtype.str,'sha256':hashlib.sha256(a.tobytes()).hexdigest(), 'initial_sha256':hashlib.sha256(initial_array.tobytes()).hexdigest(), 'buffer_elements':initial_array.size}
    outputs={}
    for name, dtype in spec['outputs'].items():
        array=np.asarray(initial[name]); dtype=np.dtype(dtype)
        if array.dtype != dtype or array.size != spec['count'] + PADDING: raise ValueError('invalid output capacity or dtype')
        outputs[name]={'dtype':dtype.str,'live_elements':spec['count'],'buffer_elements':array.size,'initial_sha256':hashlib.sha256(array.tobytes()).hexdigest()}
    return {**{k:copy.deepcopy(spec[k]) for k in fields}, 'inputs':inputs, 'outputs':outputs}


def _check_identity(identity, frequency, profile):
    if not isinstance(identity,dict): raise ValueError('identity required')
    required={'sources','compiler','compiler_sha256','runtime','device','driver','device_uuid','timestamp_frequency','control_receipt'}
    if set(identity) != required: raise ValueError('complete identity and control receipt required')
    if not isinstance(identity['sources'],dict) or not identity['sources']: raise ValueError('source closure required')
    required_paths={'src/sintergpu/kernels/common/depth_pyramid_core.slang', 'benchmarks/shaders/depth_pyramid_'+profile+'_candidate.slang', ('src/sintergpu/kernels/vkl/depth_pyramid.slang' if profile=='vkl' else 'src/sintergpu/kernels/slang_rgbd_slam/tracker.slang')}
    if profile == 'slam': required_paths.add('src/sintergpu/kernels/generated/primitives.generated.slang')
    if not required_paths <= set(identity['sources']): raise ValueError('incomplete source closure')
    for path,digest in identity['sources'].items():
        if not isinstance(path,str) or not isinstance(digest,str) or len(digest)!=64 or any(c not in '0123456789abcdef' for c in digest): raise ValueError('source SHA256 required')
        source=ROOT/path
        if source.is_symlink() or not source.is_file() or hashlib.sha256(source.read_bytes()).hexdigest()!=digest: raise ValueError('source identity mismatch')
    digest=identity['compiler_sha256']
    if not isinstance(digest,str) or len(digest)!=64 or any(c not in '0123456789abcdef' for c in digest): raise ValueError('compiler SHA256 required')
    for key in ('compiler','runtime','device','driver','device_uuid'):
        if not isinstance(identity[key],str) or not identity[key].strip(): raise ValueError('complete observed environment required')
    if type(identity['timestamp_frequency']) is not int or identity['timestamp_frequency'] != frequency or frequency <= 0:
        raise ValueError('timestamp frequency mismatch')
    receipt=identity['control_receipt']
    if not isinstance(receipt,dict) or not isinstance(receipt.get('sha256'),str) or len(receipt['sha256'])!=64 or any(c not in '0123456789abcdef' for c in receipt['sha256']):
        raise ValueError('control receipt hash required')


def _read_raw(f):
    # This is the timed readback only; comparisons and extra output copies occur later.
    return {name:f.buffers[name].to_numpy() for name in f.initial}


def _check_raw(f, raw):
    result={}
    for name,before in f.initial.items():
        actual=np.asarray(raw[name]).view(before.dtype).reshape(-1)
        if actual.shape != before.shape: raise AssertionError('buffer length changed: '+name)
        live=before.size-PADDING
        if not np.array_equal(actual[live:].view(np.uint32),before[live:].view(np.uint32)):
            raise AssertionError('tail changed: '+name)
        if name in f.spec['inputs']:
            if not np.array_equal(actual.view(np.uint32),before.view(np.uint32)): raise AssertionError('input mutated: '+name)
        else:
            result[name]=actual[:live].reshape(f.spec['shape']).copy()
    return result


def collect(device, timestamp_type, fixtures, validate, preflight, *, samples=50, warmups=5):
    """Three bracketed runs; caller validates reference outputs after each dispatch.

    read_checked includes all output and input/guard readbacks; transfer time
    includes those readbacks. Allocation/compilation and output validation are
    outside the measured intervals. No control/qualification claim is made.
    """
    if type(samples) is not int or samples < 50 or type(warmups) is not int or warmups < 0:
        raise ValueError('at least 50 samples and nonnegative warmups required')
    if set(fixtures) != {'old','candidate'}: raise ValueError('two variants required')
    old,candidate=fixtures['old'],fixtures['candidate']
    workload=_workload(old.spec,old.initial)
    if _workload(candidate.spec,candidate.initial) != workload: raise ValueError('input or dispatch contract mismatch')
    if old.kernel is candidate.kernel or any(old.buffers[k] is candidate.buffers[k] for k in old.buffers):
        raise ValueError('independent variant resources required')
    identity=copy.deepcopy(preflight())
    frequency=int(device.info.timestamp_frequency)
    _check_identity(identity,frequency,workload['profile'])
    pool=device.create_query_pool(timestamp_type,2)

    def reset(f):
        for name in f.spec['outputs']:
            f.buffers[name].copy_from_numpy(f.initial[name])
        device.wait_for_idle()

    def measure(f):
        reset(f); pool.reset()
        start=time.perf_counter_ns()
        f.kernel.dispatch(thread_count=f.spec['threads'],vars=f.bindings,query_pool=pool,query_index_before=0,query_index_after=1)
        device.wait_for_idle()
        host=(time.perf_counter_ns()-start)/1e6
        stamps=pool.get_timestamp_results(0,2)
        gpu=(float(stamps[1])-float(stamps[0]))*1000
        if not np.isfinite(gpu) or gpu <= 0: raise ValueError('invalid GPU timestamp')
        output=f.read_checked(); validate(output)
        start=time.perf_counter_ns()
        for name in f.spec['inputs']: f.buffers[name].copy_from_numpy(f.initial[name])
        for name in f.spec['outputs']: f.buffers[name].copy_from_numpy(f.initial[name])
        f.kernel.dispatch(thread_count=f.spec['threads'],vars=f.bindings)
        device.wait_for_idle()
        raw=_read_raw(f)
        transfer=(time.perf_counter_ns()-start)/1e6
        transfer_output=_check_raw(f,raw)
        validate(transfer_output)
        if not _same(output,transfer_output): raise AssertionError('measured and transfer outputs differ')
        if host <= 0 or transfer <= 0: raise ValueError('invalid host timestamp')
        return (gpu,host,transfer), output

    runs=[]
    for _ in range(3):
        for _ in range(warmups):
            for _,variant in LABELS: measure(fixtures[variant])
        run={'order':[],**{b:{label:[] for label,_ in LABELS} for b in BOUNDARIES}}
        for _ in range(samples):
            paired=[]
            for label,variant in LABELS:
                values,output=measure(fixtures[variant]); paired.append(output)
                run['order'].append(label)
                for boundary,value in zip(BOUNDARIES,values): run[boundary][label].append(value)
            if not _same(paired[0],paired[1]) or not _same(paired[0],paired[2]):
                raise AssertionError('candidate and bracketed incumbent outputs differ')
        observed=preflight()
        _check_identity(observed,frequency,workload['profile'])
        if observed != identity: raise ValueError('source identity changed')
        runs.append(run)
    return {'samples':runs,'observed_identity_unverified':identity,'workload':workload,'warmups':warmups,
            'runner_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'boundaries':{'isolated_gpu_ms':'query timestamps around one dispatch', 'host_complete_ms':'host query-instrumented dispatch and wait', 'transfer_inclusive_ms':'host input and output reset uploads, dispatch, wait, and all input/output/guard readbacks; validation excluded'},
            'sequence':'Each old/candidate/old element runs an instrumented dispatch followed by a separate transfer dispatch and readback; not uninterrupted GPU-only ABA.',
            'scope':'Single dispatch only; supplied identity and controls unverified, resident pyramid, tracking and hardware qualification not established.'}


def evaluate(report, *, bootstrap_samples=10000, seed=613):
    """Statistical screen only; this does not validate independent run controls."""
    if bootstrap_samples < 1000 or len(report.get('samples',[])) != 3:
        return {'status':'inconclusive','reason':'three runs and sufficient bootstrap required'}
    rng=np.random.default_rng(seed); results=[]
    for run in report['samples']:
        summary={}
        for boundary in BOUNDARIES:
            try:
                data=run[boundary]
                a,b,c=(np.asarray(data[k],dtype=float) for k in ('old_before','candidate','old_after'))
            except (KeyError,TypeError,ValueError): return {'status':'inconclusive','reason':'missing samples'}
            if (a.ndim != 1 or len(a)<50 or len(a)!=len(b) or len(a)!=len(c) or
                not all(x.ndim==1 and np.isfinite(x).all() and (x>0).all() for x in (a,b,c)) or
                run.get('order') != ['old_before','candidate','old_after']*len(a)):
                return {'status':'inconclusive','reason':'invalid samples or order'}
            baseline=(a+c)/2; diff=b-baseline
            boots=np.array([np.median(rng.choice(diff,len(diff),replace=True)) for _ in range(bootstrap_samples)])
            ci=np.percentile(boots,[2.5,97.5]); old95=float(np.percentile(baseline,95)); new95=float(np.percentile(b,95))
            summary[boundary]={'old_median_ms':float(np.median(baseline)),'candidate_median_ms':float(np.median(b)),
                               'old_p95_ms':old95,'candidate_p95_ms':new95,'median_difference_95ci_ms':ci.tolist()}
            if ci[1]>0 or new95>old95:
                return {'status':'inconclusive','reason':'per-run no-regression gate failed','runs':results+[summary]}
        results.append(summary)
    return {'status':'statistical_pass','runs':results,'scope':'Statistical screen only; controlled conditions and hardware qualification are unverified.'}
