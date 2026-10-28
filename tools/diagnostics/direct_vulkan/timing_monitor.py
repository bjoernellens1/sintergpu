# SPDX-License-Identifier: Apache-2.0
"""Sample target telemetry during a caller-controlled diagnostic, never qualifies it.

Snapshots are discontinuous: nvidia-smi is not an event stream, can miss short
peers and does not authenticate Vulkan/graphics process ownership. Its polling
also perturbs the measured system. No hidden locking, device or GPU submission.
"""
import hashlib
import math
import subprocess
import threading
import time
from pathlib import Path

GPU='GPU-96d2dba0-95fd-5c00-e849-2e0610f24e06'
QUERIES=(('--query-gpu=uuid,name,driver_version,clocks.current.sm,clocks.current.memory,temperature.gpu,power.draw,utilization.gpu,utilization.memory','--format=csv,noheader,nounits'),
         ('--query-compute-apps=gpu_uuid,pid,used_gpu_memory','--format=csv,noheader,nounits'),())


def make_nvidia_collector(binary,expected_sha,*,timeout=2):
    binary=Path(binary)
    if not binary.is_absolute() or binary.is_symlink() or not binary.is_file():raise ValueError('monitor binary path')
    if hashlib.sha256(binary.read_bytes()).hexdigest()!=expected_sha:raise ValueError('monitor binary hash')
    def capture():
        output=[]
        for query in QUERIES:
            p=subprocess.run([str(binary),*query],capture_output=True,text=True,timeout=timeout,check=False)
            if p.returncode:raise RuntimeError('nvidia-smi query failed')
            output.append(p.stdout)
        if not any(line.split(',')[0].strip()==GPU for line in output[0].splitlines()):raise ValueError('selected GPU missing')
        return {'gpu_query':output[0],'compute_query':output[1],'full_snapshot':output[2]}
    return capture


class SampledMonitor:
    """Explicit thread; start/stop outside measured region and serialize one owner."""
    def __init__(self,collector,*,interval_s=.25,max_gap_s=1.,shutdown_timeout_s=7.,sample_hook=None):
        if not callable(collector) or not math.isfinite(interval_s) or not math.isfinite(max_gap_s) or not math.isfinite(shutdown_timeout_s) or shutdown_timeout_s<=0 or interval_s<=0 or max_gap_s<interval_s:raise ValueError('monitor configuration')
        self.collector=collector;self.interval_s=interval_s;self.max_gap_s=max_gap_s;self.shutdown_timeout_s=shutdown_timeout_s;self.sample_hook=sample_hook
        self._stop=threading.Event();self._ready=threading.Event();self.failed=threading.Event();self._thread=None;self.samples=[];self.error=None
    def _loop(self):
        try:
            stopping=False
            while True:
                start=time.monotonic_ns();data=self.collector();end=time.monotonic_ns()
                if not isinstance(data,dict) or set(data)!={'gpu_query','compute_query','full_snapshot'} or any(not isinstance(v,str) for v in data.values()):raise ValueError('telemetry sample')
                sample={'start_ns':start,'end_ns':end,**data};self.samples.append(sample)
                if end-start>self.max_gap_s*1e9 or (len(self.samples)>1 and start-self.samples[-2]['start_ns']>self.max_gap_s*1e9):raise ValueError('monitor gap')
                if self.sample_hook:self.sample_hook(sample)
                self._ready.set()
                if stopping:break
                stopping=self._stop.wait(self.interval_s)
        except Exception as exc:self.error=exc;self.failed.set();self._ready.set()
    def start(self):
        if self._thread is not None:raise RuntimeError('monitor already started')
        self._thread=threading.Thread(target=self._loop,name='sinter-nvidia-telemetry',daemon=True);self._thread.start()
        if not self._ready.wait(self.max_gap_s):
            self._stop.set();self._thread.join(timeout=self.shutdown_timeout_s)
            if self._thread.is_alive():raise RuntimeError('monitor startup timeout; sampler remains alive')
            raise RuntimeError('monitor startup timeout')
        if self.error:raise RuntimeError('monitor failed') from self.error
    def stop(self,work_start_ns,work_end_ns):
        if self._thread is None or work_start_ns>=work_end_ns:raise ValueError('monitor window')
        self._stop.set();self._thread.join(timeout=self.shutdown_timeout_s)
        if self._thread.is_alive():raise RuntimeError('monitor did not stop')
        if self.error:raise RuntimeError('monitor failed') from self.error
        if len(self.samples)<2 or self.samples[0]['end_ns']>work_start_ns or self.samples[-1]['start_ns']<work_end_ns:raise ValueError('monitor window incomplete')
        if any(s['end_ns']-s['start_ns']>self.max_gap_s*1e9 for s in self.samples) or any(b['start_ns']-a['start_ns']>self.max_gap_s*1e9 for a,b in zip(self.samples,self.samples[1:])):raise ValueError('monitor gap')
        return {'samples':self.samples,'work_start_ns':work_start_ns,'work_end_ns':work_end_ns,
                'max_gap_ns':int(self.max_gap_s*1e9),
                'scope':'sampled telemetry only; no continuous peer proof, source attestation, isolation or performance qualification'}


def evaluate_sampled_controls(receipt,policy):
    """Check recorded snapshots against explicit controls; not peer isolation proof.

    Policy must be frozen and independently bound to the run by its caller.
    Compute PID visibility excludes unreported APIs; no claim is made about
    Xorg/graphics attribution or activity between snapshots.
    """
    import csv
    validate_policy(policy)
    samples=receipt.get('samples')
    if not isinstance(samples,list) or len(samples)<2:raise ValueError('missing telemetry samples')
    start=receipt.get('work_start_ns');end=receipt.get('work_end_ns');gap=receipt.get('max_gap_ns')
    if any(type(x)!=int for x in (start,end,gap)) or gap<=0 or start>=end:raise ValueError('telemetry window')
    if samples[0]['end_ns']>start or samples[-1]['start_ns']<end:raise ValueError('telemetry window incomplete')
    for i,sample in enumerate(samples):
        a=sample.get('start_ns');b=sample.get('end_ns')
        if type(a)!=int or type(b)!=int or a>b or b-a>gap or (i and (a<=samples[i-1]['start_ns'] or a-samples[i-1]['start_ns']>gap)):raise ValueError('telemetry coverage gap')
    for sample in samples:check_sample(sample,policy)
    return {'result':'sampled_controls_within_policy','sample_count':len(samples),
            'scope':'UNQUALIFIED caller-supplied telemetry and policy; unreported peers and between-snapshot events remain unknown'}


def check_sample(sample,policy):
    """Fail on a reported control violation; caller must prevalidate policy."""
    import csv
    rows=list(csv.reader(sample['gpu_query'].splitlines(),skipinitialspace=True));selected=[r for r in rows if r and r[0].strip()==GPU]
    if len(selected)!=1 or len(selected[0])!=9:raise ValueError('GPU telemetry shape')
    row=selected[0]
    if row[2].strip()!=policy['driver']:raise ValueError('driver mismatch')
    try:sm,mem,temp,power,util,memory_util=(float(v.strip()) for v in row[3:])
    except ValueError as exc:raise ValueError('non-numeric telemetry') from exc
    if any(not math.isfinite(x) or x<0 for x in (sm,mem,temp,power,util,memory_util)) or util>100 or memory_util>100:raise ValueError('invalid telemetry')
    if not policy['min_sm_mhz']<=sm<=policy['max_sm_mhz'] or mem<policy['min_memory_mhz'] or temp>policy['max_temperature_c'] or power>policy['max_power_w']:raise ValueError('sampled environment outside policy')
    for row in csv.reader(sample['compute_query'].splitlines(),skipinitialspace=True):
        if not row:continue
        if len(row)!=3:raise ValueError('compute telemetry shape')
        if row[0].strip()==GPU:
            try:pid=int(row[1].strip())
            except ValueError as exc:raise ValueError('compute pid') from exc
            if pid not in policy['allowed_compute_pids']:raise ValueError('unexpected sampled compute peer')
    if not sample['full_snapshot'].strip():raise ValueError('missing full snapshot')



def validate_policy(policy):
    required={'driver','min_sm_mhz','max_sm_mhz','min_memory_mhz','max_temperature_c','max_power_w','allowed_compute_pids'}
    if set(policy)!=required or not isinstance(policy['driver'],str) or not policy['driver'] or not isinstance(policy['allowed_compute_pids'],(set,frozenset)):
        raise ValueError('explicit control policy required')
    for name in required-{'driver','allowed_compute_pids'}:
        if type(policy[name]) not in (int,float) or not math.isfinite(policy[name]) or policy[name]<0:raise ValueError('invalid policy bound')
    if policy['min_sm_mhz']>policy['max_sm_mhz'] or any(type(pid)!=int or pid<=0 for pid in policy['allowed_compute_pids']):raise ValueError('invalid control policy')


def make_policy_monitor(collector,policy,**options):
    validate_policy(policy)
    frozen=dict(policy);frozen['allowed_compute_pids']=frozenset(policy['allowed_compute_pids'])
    return SampledMonitor(collector,sample_hook=lambda sample:check_sample(sample,frozen),**options)
