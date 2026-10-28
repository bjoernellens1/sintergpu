# SPDX-License-Identifier: Apache-2.0
import time
import hashlib
from types import SimpleNamespace
import pytest
from timing_monitor import SampledMonitor,make_nvidia_collector,GPU

def valid():return {'gpu_query':'target','compute_query':'','full_snapshot':'contexts'}

def test_monitor_brackets_window_with_snapshots():
    m=SampledMonitor(valid,interval_s=.001,max_gap_s=.25);m.start();begin=time.monotonic_ns();time.sleep(.005);end=time.monotonic_ns();receipt=m.stop(begin,end)
    assert receipt['samples'][0]['end_ns']<=begin and receipt['samples'][-1]['start_ns']>=end
    assert 'no continuous peer proof' in receipt['scope']

def test_monitor_rejects_failure_and_bad_shape():
    for collector in (lambda:None,lambda:(_ for _ in ()).throw(OSError('failure'))):
        with pytest.raises(RuntimeError,match='monitor failed'):SampledMonitor(collector,interval_s=.001,max_gap_s=.25).start()
    with pytest.raises(ValueError):SampledMonitor(valid,interval_s=0)
    with pytest.raises(ValueError):make_nvidia_collector('relative', 'f'*64)

def test_monitor_detects_missing_coverage():
    m=SampledMonitor(valid,interval_s=.001,max_gap_s=.25);m.start();now=time.monotonic_ns()
    with pytest.raises(ValueError,match='window incomplete'):m.stop(now-1_000_000_000,now)


def test_nvidia_collector_pins_binary_and_keeps_all_raw_output(tmp_path,monkeypatch):
    import timing_monitor
    binary=tmp_path/'nvidia-smi';binary.write_bytes(b'mock-binary');digest=hashlib.sha256(binary.read_bytes()).hexdigest();calls=[]
    def invoke(cmd,**kwargs):
        calls.append(cmd);return SimpleNamespace(returncode=0,stdout=(GPU+', 4070' if len(calls)==1 else 'raw-'+str(len(calls))))
    monkeypatch.setattr(timing_monitor.subprocess,'run',invoke)
    sample=make_nvidia_collector(binary,digest)();assert len(calls)==3 and sample['gpu_query'].startswith(GPU) and sample['compute_query']=='raw-2' and sample['full_snapshot']=='raw-3'
    binary.write_bytes(b'changed')
    with pytest.raises(ValueError,match='binary hash'):make_nvidia_collector(binary,digest)

def test_sampled_policy_checks_driver_clocks_temperature_and_compute_peers():
    from timing_monitor import evaluate_sampled_controls
    row=f'{GPU}, RTX 4070 Ti, 595.99.02, 2000, 10000, 50, 100, 10, 5'
    sample={'gpu_query':row,'compute_query':'','full_snapshot':'graphics table','start_ns':0,'end_ns':1}
    second={**sample,'start_ns':2,'end_ns':3}
    receipt={'samples':[sample.copy(),second],'work_start_ns':1,'work_end_ns':2,'max_gap_ns':10}
    policy={'driver':'595.99.02','min_sm_mhz':1900,'max_sm_mhz':2100,'min_memory_mhz':9000,'max_temperature_c':70,'max_power_w':200,'allowed_compute_pids':set()}
    result=evaluate_sampled_controls(receipt,policy);assert result['result']=='sampled_controls_within_policy' and result['scope'].startswith('UNQUALIFIED')
    receipt['samples'][1]['compute_query']=f'{GPU}, 4321, 1'
    with pytest.raises(ValueError,match='unexpected sampled compute peer'):evaluate_sampled_controls(receipt,policy)
    receipt['samples'][1]['compute_query']='';receipt['samples'][1]['gpu_query']=row.replace('2000, 10000','1000, 10000')
    with pytest.raises(ValueError,match='environment outside'):evaluate_sampled_controls(receipt,policy)
    receipt['samples'][1]['gpu_query']=row.replace('595.99.02','wrong')
    with pytest.raises(ValueError,match='driver mismatch'):evaluate_sampled_controls(receipt,policy)


def test_startup_timeout_joins_bounded_collector():
    def slow():time.sleep(.04);return valid()
    m=SampledMonitor(slow,interval_s=.001,max_gap_s=.01,shutdown_timeout_s=.5)
    with pytest.raises(RuntimeError,match='startup timeout'):m.start()
    assert not m._thread.is_alive()

def test_policy_rejects_coverage_gap():
    from timing_monitor import evaluate_sampled_controls
    row=f'{GPU}, RTX, 595.99.02, 2000, 10000, 50, 100, 10, 5'
    sample={'gpu_query':row,'compute_query':'','full_snapshot':'graphics','start_ns':0,'end_ns':1}
    receipt={'samples':[sample,{**sample,'start_ns':20,'end_ns':21}],'work_start_ns':1,'work_end_ns':20,'max_gap_ns':5}
    policy={'driver':'595.99.02','min_sm_mhz':0,'max_sm_mhz':3000,'min_memory_mhz':0,'max_temperature_c':80,'max_power_w':300,'allowed_compute_pids':set()}
    with pytest.raises(ValueError,match='coverage gap'):evaluate_sampled_controls(receipt,policy)

def test_policy_hook_detects_reported_peer_during_run():
    from timing_monitor import make_policy_monitor
    row=f'{GPU}, RTX, 595.99.02, 2000, 10000, 50, 100, 10, 5';calls=[]
    def collect():
        calls.append(1);return {'gpu_query':row,'compute_query':f'{GPU}, 77, 10' if len(calls)>1 else '', 'full_snapshot':'graphics'}
    policy={'driver':'595.99.02','min_sm_mhz':0,'max_sm_mhz':3000,'min_memory_mhz':0,'max_temperature_c':80,'max_power_w':300,'allowed_compute_pids':set()}
    m=make_policy_monitor(collect,policy,interval_s=.001,max_gap_s=.2);m.start()
    assert m.failed.wait(.5) and 'unexpected sampled compute peer' in str(m.error) and len(m.samples)>=2
    with pytest.raises(RuntimeError,match='monitor failed'):m.stop(m.samples[0]['end_ns'],time.monotonic_ns())

def test_bound_child_pid_is_allowed_and_other_peer_still_aborts():
    from timing_monitor import make_policy_monitor
    row=f'{GPU}, RTX, 595.99.02, 2000, 10000, 50, 100, 10, 5';pids=[]
    def collect():return {'gpu_query':row,'compute_query':f'{GPU}, {pids[-1]}, 10' if pids else '', 'full_snapshot':'graphics'}
    policy={'driver':'595.99.02','min_sm_mhz':0,'max_sm_mhz':3000,'min_memory_mhz':0,'max_temperature_c':80,'max_power_w':300,'allowed_compute_pids':set()}
    m=make_policy_monitor(collect,policy,interval_s=.001,max_gap_s=.2);m.start();m.bind_child_pid(1234);pids.append(1234);time.sleep(.005)
    assert not m.failed.is_set()
    with pytest.raises(ValueError,match='already bound'):m.bind_child_pid(9999)
    pids.append(9999);assert m.failed.wait(.5) and 'unexpected sampled compute peer' in str(m.error)
