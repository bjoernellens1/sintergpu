# SPDX-License-Identifier: Apache-2.0
import os
import signal
import threading
import subprocess
import pytest
from timing_supervisor import supervise,QuarantineRequired

class Journal:
    def __init__(self):self.files={}
    def put(self,name,data):assert name not in self.files;self.files[name]=data
class Monitor:
    def __init__(self,violate=False):self.failed=threading.Event();self.samples=[];self.error=None;self.violate=violate
    def start(self):self.samples.append({'telemetry':'initial'})
    def stop(self,a,b):return {'scope':'unqualified'}
class Child:
    def __init__(self,monitor,fd,*,ready=True,rc=0,cleanup=True):
        self.returncode=None;self.monitor=monitor;self.fd=os.dup(fd);self.terminated=False;self.rc=rc;self.cleanup=cleanup
        if ready:os.write(self.fd,b'R')
    def finish(self):
        if self.cleanup:os.write(self.fd,b'C')
        os.close(self.fd);self.fd=-1;self.returncode=self.rc
    def poll(self):
        if self.returncode is None and self.monitor.violate:self.monitor.error=ValueError('peer');self.monitor.failed.set()
        elif self.returncode is None:self.finish()
        return self.returncode
    def send_signal(self,sig):assert sig==signal.SIGUSR1;self.terminated=True;self.rc=1;self.finish()
    def wait(self,timeout):return self.returncode
    def kill(self):raise AssertionError('must not kill')

def fake_process(m,**opts):
    holder=[]
    def popen(*a,**k):
        c=Child(m,int(k['env']['SINTERGPU_COOPERATIVE_FD']),**opts);holder.append(c);return c
    return holder,popen

def test_success_has_lease_and_retained_monitor_evidence():
    monitor=Monitor();journal=Journal();leases=[];children,popen=fake_process(monitor)
    result=supervise(['fake'],monitor,journal,lambda:leases.append(1),lambda *a:pytest.fail('quarantine'),label='run',popen=popen)
    assert result['returncode']==0 and len(leases)==2 and b'unqualified' in journal.files['run.monitor.json']

def test_detected_peer_requests_cooperative_stop_and_quarantines():
    monitor=Monitor(True);journal=Journal();leases=[];quarantined=[];children,popen=fake_process(monitor)
    with pytest.raises(ExceptionGroup) as errors:
        supervise(['fake'],monitor,journal,lambda:leases.append(1),lambda c,e:quarantined.append(c),label='run',popen=popen)
    assert children[0].terminated and quarantined==children and len(leases)==2 and b'peer' in journal.files['run.monitor.json']
    assert any(isinstance(e,QuarantineRequired) for e in errors.value.exceptions)

def test_missing_cleanup_quarantines_even_on_success():
    monitor=Monitor();children,popen=fake_process(monitor,cleanup=False);quarantined=[]
    with pytest.raises(ExceptionGroup):supervise(['fake'],monitor,Journal(),lambda:None,lambda c,e:quarantined.append(c),label='run',popen=popen)
    assert quarantined==children

def test_real_sampled_policy_stops_fake_child_on_reported_peer():
    from timing_monitor import make_policy_monitor,GPU
    row=f'{GPU}, RTX, 595.99.02, 2000, 10000, 50, 100, 10, 5';calls=[]
    def collect():
        calls.append(1);return {'gpu_query':row,'compute_query':f'{GPU}, 77, 10' if len(calls)>1 else '', 'full_snapshot':'graphics'}
    policy={'driver':'595.99.02','min_sm_mhz':0,'max_sm_mhz':3000,'min_memory_mhz':0,'max_temperature_c':80,'max_power_w':300,'allowed_compute_pids':set()}
    m=make_policy_monitor(collect,policy,interval_s=.001,max_gap_s=.2);journal=Journal();quarantined=[]
    class WaitingChild(Child):
        def poll(self):return self.returncode
    def popen(*a,**k):return WaitingChild(m,int(k['env']['SINTERGPU_COOPERATIVE_FD']))
    with pytest.raises(ExceptionGroup):supervise(['fake'],m,journal,lambda:None,lambda c,e:quarantined.append(c),label='run',poll_s=.01,popen=popen)
    assert quarantined and quarantined[0].terminated and b'unexpected sampled compute peer' in journal.files['run.monitor.json']

def test_violation_before_ready_never_signals_and_quarantines():
    monitor=Monitor();journal=Journal();quarantined=[];children=[]
    class NotReady(Child):
        def poll(self):return self.returncode
    def popen(*a,**k):
        c=NotReady(monitor,int(k['env']['SINTERGPU_COOPERATIVE_FD']),ready=False);children.append(c);monitor.failed.set();return c
    with pytest.raises(ExceptionGroup):supervise(['fake'],monitor,journal,lambda:None,lambda c,e:quarantined.append(c),label='run',popen=popen)
    assert children and not children[0].terminated and quarantined==children
    os.close(children[0].fd)

def test_abnormal_exit_with_cleanup_still_quarantines():
    monitor=Monitor();children,popen=fake_process(monitor,rc=2);quarantined=[]
    with pytest.raises(ExceptionGroup):supervise(['fake'],monitor,Journal(),lambda:None,lambda c,e:quarantined.append(c),label='run',popen=popen)
    assert quarantined==children and children[0].returncode==2
