# SPDX-License-Identifier: Apache-2.0
"""CPU-testable paired timing contract; no native timing backend or GPU claims.

A caller supplies a future independently reviewed backend and retains its GPU
lease. This module cannot authenticate its caller, device, clocks, controls,
Vulkan timestamp semantics or run independence; it is not a qualification runner.
"""
import copy
import math
import statistics

LABELS=(('old_before','old'),('candidate','candidate'),('old_after','old'))
METRICS=('kernel_ns','resident_pipeline_ns','transfer_host_ns')


def ticks_to_ns(pair,period_ns,valid_bits,upper_bound_ns):
    if not isinstance(pair,(list,tuple)) or len(pair)!=2 or type(valid_bits)!=int or not 1<=valid_bits<=64:
        raise ValueError('timestamp pair/valid bits')
    if type(period_ns) not in (int,float) or not math.isfinite(period_ns) or period_ns<=0: raise ValueError('timestamp period')
    limit=1<<valid_bits
    # The caller must supply a separately measured host completion upper bound.
    # It must be strictly below one counter wrap or modulo subtraction is ambiguous.
    if (type(upper_bound_ns)!=int or upper_bound_ns<=0 or upper_bound_ns>=limit*period_ns): raise ValueError('timestamp wrap ambiguity')
    if any(type(v)!=int or not 0<=v<limit for v in pair): raise ValueError('timestamp range')
    delta=(pair[1]-pair[0])%limit
    if not delta: raise ValueError('zero/ambiguous timestamp interval')
    result=delta*period_ns
    if result>upper_bound_ns: raise ValueError('timestamp exceeds host completion bound')
    return result


def _p95(values):
    ordered=sorted(values);x=(len(ordered)-1)*0.95;i=int(x)
    return ordered[i]+(ordered[min(i+1,len(ordered)-1)]-ordered[i])*(x-i)


def assess(runs):
    """Conservative descriptive thresholds only; no statistical or adoption claim."""
    if len(runs)!=3: raise ValueError('three runs required')
    result=[]
    for run in runs:
        metrics={}
        for metric in METRICS:
            data=run[metric];a,b,c=(data[label] for label,_ in LABELS)
            if not len(a)==len(b)==len(c) or len(a)<50: raise ValueError('sample counts')
            if any(not isinstance(x,(float,int)) or isinstance(x,bool) or not math.isfinite(x) or x<=0 for seq in (a,b,c) for x in seq): raise ValueError('invalid sample')
            baseline=[(x+y)/2 for x,y in zip(a,c)]
            mbase=statistics.median(baseline);mcandidate=statistics.median(b)
            pbase=min(_p95(a),_p95(c));pcandidate=_p95(b)
            metrics[metric]={'old_before_median_ns':statistics.median(a),'old_after_median_ns':statistics.median(c),
                'paired_baseline_median_ns':mbase,'candidate_median_ns':mcandidate,
                'old_p95_min_ns':pbase,'candidate_p95_ns':pcandidate,
                'median_candidate_le_baseline':mcandidate<=mbase,'candidate_p95_le_105_percent_reference':pcandidate<=1.05*pbase}
        result.append(metrics)
    return result


def _outputs(value):
    if not isinstance(value,dict) or not value or any(not isinstance(k,str) or not isinstance(v,bytes) for k,v in value.items()):
        raise ValueError('complete immutable byte readbacks required')
    return value


def collect(backend,identity,validate,*,samples=50,warmups=5):
    """Use reset/measure/read/validate order; same-process runs are not independent.

    Backend supplies contract(), reset(variant), resident(variant), transfer(variant)
    and validate_outputs(variant, raw). Resident returns raw tick pairs and complete
    readbacks; transfer includes uploads and all raw readbacks. Allocation and
    validation must occur outside reported intervals. Each backend call is a trust
    boundary until real Vulkan instrumentation is independently reviewed.
    """
    if type(samples)!=int or samples<50 or type(warmups)!=int or warmups<0: raise ValueError('sample counts')
    contract=copy.deepcopy(backend.contract())
    if contract.get('old')!=contract.get('candidate') or not contract.get('old'): raise ValueError('workload contract mismatch')
    period=contract.get('timestamp_period_ns');bits=contract.get('timestamp_valid_bits')
    if type(bits)!=int or not 1<=bits<=64 or type(period) not in (int,float) or not math.isfinite(period) or period<=0: raise ValueError('timestamp capabilities')
    observed=copy.deepcopy(identity());
    if not isinstance(observed,dict) or not observed: raise ValueError('identity unavailable')
    validate('preflight',None)
    runs=[]
    for run_index in range(3):
        run={'order':[],'raw_ticks':{label:[] for label,_ in LABELS},**{metric:{label:[] for label,_ in LABELS} for metric in METRICS}}
        for iteration in range(warmups+samples):
            paired=[]
            for label,variant in LABELS:
                backend.reset(variant)
                resident=backend.resident(variant)
                kernel=ticks_to_ns(resident['kernel_ticks'],period,bits,resident['completion_bound_ns'])
                pipeline=ticks_to_ns(resident['pipeline_ticks'],period,bits,resident['completion_bound_ns'])
                _outputs(resident['outputs']);backend.validate_outputs(variant,resident['outputs']);validate('resident',resident['outputs'])
                # Checked resident output must precede resetting outputs for transfer.
                backend.reset(variant)
                transfer=backend.transfer(variant)
                ns=transfer['host_ns']
                if type(ns)!=int or ns<=0: raise ValueError('transfer timing')
                _outputs(transfer['outputs']);backend.validate_outputs(variant,transfer['outputs']);validate('transfer',transfer['outputs'])
                if resident['outputs']!=transfer['outputs']: raise AssertionError('resident/transfer mismatch')
                paired.append(resident['outputs'])
                if iteration>=warmups:
                    run['order'].append(label);run['raw_ticks'][label].append({'kernel':list(resident['kernel_ticks']),'pipeline':list(resident['pipeline_ticks']),'completion_bound_ns':resident['completion_bound_ns']})
                    for metric,value in zip(METRICS,(kernel,pipeline,ns)):run[metric][label].append(value)
            if paired[0]!=paired[1] or paired[0]!=paired[2]: raise AssertionError('paired output mismatch')
        if identity()!=observed: raise ValueError('identity changed')
        runs.append(run)
    return {'runs':runs,'descriptive_evaluation':assess(runs),'contract_unverified':contract,'identity_unverified':observed,
        'sequence':'For each old/candidate/old: reset, resident timestamps and complete readback/validation; reset, transfer-inclusive upload/dispatch/readback and validation.',
        'scope':'CPU protocol only; no verified backend, independent controls, statistical confidence, performance qualification or optimized selection.'}
