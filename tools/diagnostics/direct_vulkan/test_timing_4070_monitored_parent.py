# SPDX-License-Identifier: Apache-2.0
import pytest
import timing_4070_monitored_parent as candidate

def test_production_gate_precedes_any_device_or_source_access(monkeypatch):
    monkeypatch.setattr(candidate,'committed_sources',lambda *_:pytest.fail('unexpected source access'))
    with pytest.raises(RuntimeError,match='blocked pending independent review'):candidate.run(None)

def test_experimental_closure_includes_all_new_dependencies():
    required={'timing_cooperative.cpp','timing_handshake.py','timing_monitor.py','timing_supervisor.py','timing_quarantine_guardian.py','timing_quarantine_lease.py','timing_trial_manifest.py','timing_preregistered_analysis.py','timing_4070_monitored_parent.py'}
    assert required.issubset(set(candidate.LOCAL)) and candidate.NATIVE_SHA is None

def exercise_parent(tmp_path,monkeypatch,mode):
    import inspect,json,hashlib
    from pathlib import Path
    from types import SimpleNamespace
    import timing_monitor,timing_trial_manifest
    # Test-only in-memory copy; production file retains its unconditional block.
    source=inspect.getsource(candidate.run).replace("    raise RuntimeError('monitored trial blocked pending independent review')\n",'',1)
    scope=dict(candidate.__dict__);exec(compile(source,'<fake-parent>','exec'),scope)
    scope['NATIVE_SHA']='a'*64
    compiler=tmp_path/'slang';compiler.write_bytes(b'compiler')
    policy={'driver':'595.99.02','min_sm_mhz':0,'max_sm_mhz':3000,'min_memory_mhz':0,'max_temperature_c':80,'max_power_w':300,'allowed_compute_pids':[]}
    policy_file=tmp_path/'policy';policy_file.write_text(json.dumps(policy));events=[];leases=[]
    args=SimpleNamespace(execute=True,output=tmp_path/'receipt.json',commit='a'*40,icd='icd',icd_sha='hash',slangc=compiler,policy=policy_file,monitor_binary='/fake',samples=50)
    sources={name:(name+'-bytes').encode() for name in candidate.LOCAL}
    class Journal:
        def __init__(self,path):self.root=Path(str(path)+'.artifacts');self.root.mkdir();self.directory=1
        def event(self,name,**kwargs):events.append(name)
        def put(self,name,data):(self.root/name).write_bytes(data);events.append('put:'+name)
        def close(self):events.append('journal_close')
    class Lease:
        def __init__(self,*a):self.fd=42;self.quarantine_required=False;self.process=SimpleNamespace(poll=lambda:None);leases.append(self)
        def start_guardian(self):events.append('guardian');return {'guardian_pid':123}
        def check(self):events.append('lease_check')
        def quarantine(self,child,reason):self.quarantine_required=True;events.append('quarantine');return {'guardian_pid':123}
        def close(self):
            events.append('lease_close')
            if mode=='guardian-death':raise RuntimeError('guardian unavailable')
        def release_normal(self):events.append('release_normal')
    def captured(cmd,journal,label):
        Path(cmd[-1]).write_bytes(b'donor' if label.startswith('donor') else b'candidate');return SimpleNamespace(returncode=0)
    helper={'ArtifactJournal':Journal,'checked_icd':lambda *a:'icdhash','checked_sources':lambda c:{'fixture':b'f','core':b'c','donor':b'd','candidate':b'n'},
            'fixture':lambda f:({},b'input'),'COMPILER_SHA':hashlib.sha256(b'compiler').hexdigest(),
            'SPV_SHA':{'donor':hashlib.sha256(b'donor').hexdigest(),'candidate':hashlib.sha256(b'candidate').hexdigest()},
            'captured_process':captured,'lease_event':lambda *a:events.append('lease_event')}
    class MonitorModule:
        validate_policy=staticmethod(timing_monitor.validate_policy)
        make_nvidia_collector=staticmethod(lambda *a:lambda:None)
        make_policy_monitor=staticmethod(lambda *a:SimpleNamespace(sample_hook=lambda sample:None))
    class Supervisor:
        @staticmethod
        def supervise(cmd,monitor,journal,check,quarantine,**kwargs):
            label=kwargs['label'];journal.put(label+'.stdout',b'');journal.put(label+'.stderr',b'')
            if mode!='success':
                journal.put('partial.samples',b'partial')
                quarantine(object(),ValueError('peer'));raise RuntimeError('sampled control violation')
    modules={'timing_monitor':MonitorModule,'timing_trial_manifest':timing_trial_manifest,
             'timing_quarantine_lease':SimpleNamespace(QuarantineLease=Lease),'timing_supervisor':Supervisor,
             'timing_preregistered_analysis':SimpleNamespace(analyze=lambda r:{'scope':'UNQUALIFIED'})}
    scope.update(committed_sources=lambda c:sources,load_checked_modules=lambda s:modules,require_capacity=lambda p:200000000,
                 load_checked=lambda src,name,path:helper if name=='sinter_timing_helper' else ({'parse':lambda *a:None} if name=='sinter_timing_report' else {'validate':lambda *a:{}}),
                 build_native=lambda s:b'binary',check_peer=lambda *a:None,
                 hold_parent_lease=lambda *a:(_ for _ in ()).throw(RuntimeError('parent hold invoked')))
    def runs(**kw):
        for _ in range(3):kw['process'](['fake'],pass_fds=(42,))
        return [1,2,3]
    scope['controlled_runs']=runs
    if mode=='success':
        payload=scope['run'](args);assert payload['status']=='unqualified_trusted_host_timing_only'
        assert events.index('put:preregistration.json')<events.index('guardian')<events.index('put:watch-0.stdout')
        assert events.count('put:watch-0.stdout')==1 and 'release_normal' in events
    elif mode=='guardian-death':
        with pytest.raises(RuntimeError,match='parent hold invoked'):scope['run'](args)
        assert 'quarantine' in events and 'lease_close' in events and 'release_normal' not in events
    else:
        with pytest.raises(RuntimeError,match='sampled control violation'):scope['run'](args)
        assert 'quarantine' in events and 'lease_close' in events and 'release_normal' not in events
        assert (tmp_path/'receipt.json.artifacts/partial.samples').read_bytes()==b'partial'
        assert json.loads(args.output.read_text())['status']=='failed'

def test_fake_parent_success(tmp_path,monkeypatch):exercise_parent(tmp_path,monkeypatch,'success')
def test_fake_parent_quarantine_retains_partial(tmp_path,monkeypatch):exercise_parent(tmp_path,monkeypatch,'quarantine')
def test_fake_parent_guardian_death_enters_hold(tmp_path,monkeypatch):exercise_parent(tmp_path,monkeypatch,'guardian-death')
