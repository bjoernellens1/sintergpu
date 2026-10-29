# SPDX-License-Identifier: Apache-2.0
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
import timing_4070_transfer_components_parent as parent
from timing_transfer_components_report import validate_components

class Journal:
    def __init__(self, root): self.root=root; self.events=[]
    def event(self, name, **kw): self.events.append((name,kw))
    def put(self, name, data): (self.root/name).write_bytes(data)
    @property
    def directory(self):
        import os
        # Supplied as a borrowed fd for the duration of controlled_runs.
        return self.fd

def test_blocked_before_io(tmp_path):
    with pytest.raises(RuntimeError,match='pin unavailable'):
        parent.run(SimpleNamespace(execute=True,output=tmp_path/'receipt'))
    assert not list(tmp_path.iterdir())

def test_controlled_runs_validates_component_journal(tmp_path):
    import os
    journal=Journal(tmp_path);journal.fd=os.open(tmp_path,os.O_RDONLY|os.O_DIRECTORY)
    def process(cmd,**kwargs):
        output=Path(cmd[6]); rows=[];parts=[]
        for i in range(150):
            prefix=f'0 {i//3} {i%3}';rows.append(f'{prefix} 1 2 3 4 9 10 '+ 'a'*64);parts.append(f'{prefix} 10 1 2 3 4')
        output.write_text('\n'.join(rows)+'\n');Path(str(output)+'.raw').write_bytes(b'raw');Path(str(output)+'.components').write_text('\n'.join(parts)+'\n')
        return SimpleNamespace(returncode=0,stdout=b'',stderr=b'')
    helpers={'lease_event':lambda *a:None,'decode':lambda x:x,'validate':lambda *a:None,'component_validate':validate_components}
    try:
        runs=parent.controlled_runs(lease_fd=1,executable='fake',shader_paths=['a','b'],fixture_path='in',journal=journal,helpers=helpers,parse_report=lambda *a,**kw:{},arrays=None,controls=lambda *a:None,process=process)
        assert len(runs)==3
        assert sum(name=='components_validated' for name,_ in journal.events)==3
        assert len(list(tmp_path.glob('*.components')))==3
    finally:os.close(journal.fd)

def test_incomplete_components_fail_and_evidence_retained(tmp_path):
    import os
    journal=Journal(tmp_path);journal.fd=os.open(tmp_path,os.O_RDONLY|os.O_DIRECTORY)
    def process(cmd,**kwargs):
        output=Path(cmd[6]);output.write_text('0 0 0 1 2 3 4 9 10 '+ 'a'*64+'\n')
        Path(str(output)+'.raw').write_bytes(b'partial-raw')
        Path(str(output)+'.components').write_text('0 0 0 10 1 2 3 4\n')
        return SimpleNamespace(returncode=0,stdout=b'',stderr=b'')
    helpers={'lease_event':lambda *a:None,'decode':lambda x:x,'validate':lambda *a:None,'component_validate':validate_components}
    try:
        with pytest.raises(ValueError,match='incomplete'):
            parent.controlled_runs(lease_fd=1,executable='fake',shader_paths=['a','b'],fixture_path='in',journal=journal,helpers=helpers,parse_report=lambda *a,**kw:{},arrays=None,controls=lambda *a:None,process=process)
        assert (tmp_path/'run-0.samples.raw').read_bytes()==b'partial-raw'
        assert (tmp_path/'run-0.samples.components').exists()
        assert not any(name=='components_validated' for name,_ in journal.events)
    finally: os.close(journal.fd)
