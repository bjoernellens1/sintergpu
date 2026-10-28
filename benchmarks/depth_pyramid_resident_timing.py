# SPDX-License-Identifier: Apache-2.0
"""CPU-testable resident timing core, not an executable qualification runner.

Caller must verify the exact source/compiled/device/driver/lock/limits context;
this module cannot independently verify them. The timestamp interval contains
explicit barriers, all stages, and no uploads or readbacks.
"""
import copy
import time
import numpy as np
from benchmarks.depth_pyramid_resident_graph import PADDING

LABELS=(('old_before','old'),('candidate','candidate'),('old_after','old'))
BOUNDARIES=('gpu_resident_ms','host_complete_ms','transfer_inclusive_ms')


def _equal(a,b):
    if a.keys()!=b.keys(): return False
    return all(a[k].keys()==b[k].keys() and all(a[k][n].shape==b[k][n].shape and a[k][n].dtype==b[k][n].dtype and np.array_equal(a[k][n].view(np.uint32),b[k][n].view(np.uint32)) for n in a[k]) for k in a)


def _checked(graph,validator):
    result={}
    def per_level(level,values):
        validator(level,values); result[level]={k:v.copy() for k,v in values.items()}
    graph.read_checked(per_level)
    if set(result)!={p['level'] for p in graph.plans}: raise ValueError('missing levels')
    return result


def _raw(graph):
    return ({name:buf.to_numpy() for name,buf in graph.inputs.items()},
            [{name:buf.to_numpy() for name,buf in level.items()} for level in graph.levels])


def _decode(graph,raw,validator):
    inputs,outputs=raw
    if inputs.keys()!=graph.original.keys() or len(outputs)!=len(graph.plans): raise ValueError('missing buffers')
    for name,before in graph.original.items():
        a=np.asarray(inputs[name]).view(before.dtype).reshape(-1)
        if a.shape!=before.shape or not np.array_equal(a.view(np.uint32),before.view(np.uint32)):
            raise AssertionError('input changed: '+name)
    result={}
    for plan,buffers,before in zip(graph.plans,outputs,graph.output_initial):
        if buffers.keys()!=before.keys(): raise ValueError('missing level outputs')
        values={}
        for name,initial in before.items():
            a=np.asarray(buffers[name]).view(initial.dtype).reshape(-1)
            if a.shape!=initial.shape or not np.array_equal(a[-PADDING:].view(np.uint32),initial[-PADDING:].view(np.uint32)):
                raise AssertionError('tail changed: '+name)
            values[name]=a[:-PADDING].reshape((plan['shape'][0],*plan['shape'][-2:])).copy()
        validator(plan['level'],values); result[plan['level']]=values
    return result


def collect(device,timestamp_type,old,candidate,validator,identity,*,samples=50,warmups=5):
    if type(samples) is not int or samples<50 or type(warmups) is not int or warmups<0: raise ValueError('at least 50 samples required')
    if old.plans!=candidate.plans or old.original.keys()!=candidate.original.keys(): raise ValueError('graph contract mismatch')
    for name,a in old.original.items():
        b=candidate.original[name]
        if a.dtype!=b.dtype or a.shape!=b.shape or not np.array_equal(a.view(np.uint32),b.view(np.uint32)):
            raise ValueError('input buffer mismatch')
    if len(old.output_initial)!=len(candidate.output_initial): raise ValueError('output contract mismatch')
    for a,b in zip(old.output_initial,candidate.output_initial):
        if a.keys()!=b.keys() or any(a[k].dtype!=b[k].dtype or a[k].shape!=b[k].shape or not np.array_equal(a[k].view(np.uint32),b[k].view(np.uint32)) for k in a):
            raise ValueError('output initialization mismatch')
    observed=copy.deepcopy(identity())
    if not isinstance(observed,dict) or not observed: raise ValueError('caller identity required')
    frequency=int(device.info.timestamp_frequency)
    if frequency<=0: raise ValueError('GPU timestamp frequency unavailable')
    pool=device.create_query_pool(timestamp_type,2)
    graphs={'old':old,'candidate':candidate}
    def reset_graph(graph):
        for buffers,initial in zip(graph.levels,graph.output_initial):
            for name in buffers: buffers[name].copy_from_numpy(initial[name])
        device.wait_for_idle()
    def resident(graph):
        pool.reset(); start=time.perf_counter_ns(); encoder=device.create_command_encoder()
        encoder.write_timestamp(pool,0); graph.record(encoder); encoder.write_timestamp(pool,1)
        device.submit_command_buffer(encoder.finish()); device.wait_for_idle(); host=(time.perf_counter_ns()-start)/1e6
        ticks=pool.get_results(0,2); gpu=(int(ticks[1])-int(ticks[0]))*1000/frequency
        if not np.isfinite(gpu) or gpu<=0 or host<=0: raise ValueError('invalid timestamp')
        return gpu,host,_checked(graph,validator)
    # Establish independent and paired parity before any output reset.
    initial=[resident(graphs[k])[2] for k in ('old','candidate')]
    if not _equal(*initial): raise AssertionError('initial paired output differs')
    def execute(graph):
        reset_graph(graph)
        gpu,host,resident_output=resident(graph)
        if not _equal(initial[0],resident_output): raise AssertionError('resident differs from validated paired baseline')
        # Reset only after complete resident validation, outside transfer interval.
        reset_graph(graph)
        start=time.perf_counter_ns()
        for name,buf in graph.inputs.items(): buf.copy_from_numpy(graph.original[name])
        encoder=device.create_command_encoder(); graph.record(encoder); device.submit_command_buffer(encoder.finish()); device.wait_for_idle()
        raw=_raw(graph); transfer=(time.perf_counter_ns()-start)/1e6
        transferred=_decode(graph,raw,validator)
        if not _equal(resident_output,transferred): raise AssertionError('resident and transfer outputs differ')
        if transfer<=0: raise ValueError('invalid host interval')
        return (gpu,host,transfer),resident_output
    runs=[]
    for _ in range(3):
        run={'order':[],**{b:{k:[] for k,_ in LABELS} for b in BOUNDARIES}}
        for n in range(warmups+samples):
            outputs=[]
            for label,variant in LABELS:
                values,result=execute(graphs[variant]); outputs.append(result)
                if n>=warmups:
                    run['order'].append(label)
                    for boundary,value in zip(BOUNDARIES,values): run[boundary][label].append(value)
            if not _equal(outputs[0],outputs[1]) or not _equal(outputs[0],outputs[2]):
                raise AssertionError('paired output differs')
        if identity()!=observed: raise ValueError('caller identity changed')
        runs.append(run)
    return {'runs':runs,'identity_unverified':observed,'timestamp_frequency':frequency,'warmups':warmups,
        'sequence':'Each old/candidate/old sample resets all output levels outside the clock, runs and validates the timestamped graph against a previously paired baseline, resets outputs again outside the clock, then uploads inputs and runs the separate transfer graph and validates all raw readbacks.',
        'boundaries':{'gpu_resident_ms':'GPU timestamps around full explicitly-barriered graph','host_complete_ms':'host recording of timestamped graph plus submission and wait','transfer_inclusive_ms':'input uploads, full graph submission and wait, all input and level output raw readbacks; validation outside'},
        'scope':'Pilot collector only; same-process loops are not independent controlled runs. Caller identity, lock, device limits and controls are unverified here. No performance qualification or adoption claim.'}
