# SPDX-License-Identifier: Apache-2.0
import hashlib
from pathlib import Path
import os
from types import SimpleNamespace
import pytest
from timing_parent import controlled_runs, run, receipt_once
from timing_report import parse,PARAMS,SAMPLE_BYTES

class Journal:
    def __init__(self,path): self.root=path;self.directory=os.open(path,os.O_DIRECTORY);self.events=[]
    def event(self,name,**kwargs): self.events.append((name,kwargs))
    def put(self,name,data): (self.root/name).write_bytes(data)
    def close(self): os.close(self.directory)

def test_three_processes_and_failure(tmp_path):
    rawvalue=b'x'*SAMPLE_BYTES;dig=hashlib.sha256(rawvalue).hexdigest()
    header=['# unqualified engineering diagnostics; ticks and host nanoseconds','# timestamp_valid_bits=32 period_ns=1 queue_family=0']
    rows=[f'0 {i} {s} 1 2 3 4 100 50 {dig}' for i in range(50) for s in range(3)]
    raw=(rawvalue+PARAMS)*(50*3*2);calls=[];validations=[]
    journal=Journal(tmp_path)
    helpers={'lease_event':lambda fd,j,stage:calls.append(stage),'decode':lambda b:b,'validate':lambda a,b,c:validations.append(b==rawvalue and c==rawvalue)}
    def process(cmd,**kwargs):
        calls.append(('process',cmd[-1],kwargs['pass_fds']))
        output=Path(cmd[6]);output.write_text('\n'.join(header+rows)+'\n');Path(str(output)+'.raw').write_bytes(raw)
        return SimpleNamespace(returncode=0,stdout=b'ok',stderr=b'')
    try:
        result=controlled_runs(lease_fd=42,executable='probe',shader_paths=['old','new'],fixture_path='fixture',journal=journal,helpers=helpers,parse_report=parse,arrays={},controls=lambda j,stage:calls.append("control-"+stage),process=process)
        assert len(result)==3 and len(validations)==450 and all(validations)
        assert [v[0] for v in calls if isinstance(v,tuple)]==['process']*3
        assert calls==['run-0-before','control-run-0-before',('process','50',(42,)),'run-0-after','control-run-0-after','run-1-before','control-run-1-before',('process','50',(42,)),'run-1-after','control-run-1-after','run-2-before','control-run-2-before',('process','50',(42,)),'run-2-after','control-run-2-after']
    finally: journal.close()

def test_receipt_atomic_and_write_failure(tmp_path,monkeypatch):
    import timing_parent
    receipt_once(tmp_path/'ok.json',{'status':'ok'})
    assert (tmp_path/'ok.json').read_text().strip()=='{"status": "ok"}'
    def failed(_fd): raise OSError('injected fsync failure')
    monkeypatch.setattr(timing_parent.os,'fsync',failed)
    with pytest.raises(OSError,match='injected fsync failure'): receipt_once(tmp_path/'failed.json',{'status':'ok'})
    assert not (tmp_path/'failed.json').exists()
    assert not list(tmp_path.glob('.sinter-timing-receipt-*'))

def test_production_gate_before_input_access():
    with pytest.raises(RuntimeError,match="blocked pending independent review"):
        run(None)

def test_failed_child_fails_closed(tmp_path):
    journal=Journal(tmp_path);events=[]
    helpers={'lease_event':lambda fd,j,stage:events.append(stage)}
    try:
        with pytest.raises(RuntimeError,match='run 0 failed'):
            controlled_runs(lease_fd=42,executable='probe',shader_paths=['old','new'],fixture_path='fixture',journal=journal,helpers=helpers,parse_report=parse,arrays={},controls=lambda j,stage:None,process=lambda *a,**k:SimpleNamespace(returncode=2,stdout=b'',stderr=b'failed'))
        assert events==['run-0-before','run-0-after']
        assert (tmp_path/'run-0.stderr').read_bytes()==b'failed'
    finally:journal.close()

def test_committed_source_binding_rejects_mutation(tmp_path,monkeypatch):
    import timing_parent as parent
    root=tmp_path/'repo';path=root/parent.BASE/'unit.h';path.parent.mkdir(parents=True);path.write_bytes(b'expected')
    monkeypatch.setattr(parent,'ROOT',root);monkeypatch.setattr(parent,'LOCAL',('unit.h',))
    monkeypatch.setattr(parent.subprocess,'check_output',lambda *args,**kwargs:b'expected')
    assert parent.committed_sources('a'*40)=={'unit.h':b'expected'}
    path.write_bytes(b'tampered')
    with pytest.raises(ValueError,match='source mismatch'):parent.committed_sources('a'*40)
    path.unlink();target=tmp_path/'target';target.write_bytes(b'expected');path.symlink_to(target)
    with pytest.raises(ValueError,match='symlink'):parent.committed_sources('a'*40)

def test_double_build_uses_staged_source(tmp_path,monkeypatch):
    import timing_parent as parent
    compiler=tmp_path/'compiler';compiler.write_bytes(b'compiler');monkeypatch.setattr(parent,'CPP_SHA',hashlib.sha256(b'compiler').hexdigest())
    calls=[]
    def compile_fake(cmd,**_):
        source=Path(cmd[cmd.index('-o')-1]);assert source.read_bytes()==b'native-source'
        assert (source.parent/'dependency.h').read_bytes()==b'verified-dependency'
        Path(cmd[cmd.index('-o')+1]).write_bytes(b'identical-binary');calls.append(cmd)
    monkeypatch.setattr(parent.subprocess,'run',compile_fake)
    assert parent.build_native({'timing_executable.cpp':b'native-source','dependency.h':b'verified-dependency'},compiler)==b'identical-binary'
    assert len(calls)==2
