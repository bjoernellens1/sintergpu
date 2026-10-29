# SPDX-License-Identifier: Apache-2.0
"""CPU fake full orchestration only: fake shader copies retained donor bytes."""
import hashlib
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import pytest
from timing_4070_monitored_parent import controlled_runs
from timing_monitor import make_policy_monitor,GPU
from timing_quarantine_lease import QuarantineLease
from timing_report import parse
from timing_supervisor import supervise
from timing_trial_manifest import make_manifest,write_preregistration,verify_preregistration

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
FIXTURE=ROOT/'diagnostics/direct-point-plane-4070-1e4af42/retained-receipt.json.artifacts'
GUARDIAN=HERE/'timing_quarantine_guardian.py'

def compile_fake(tmp_path,lock):
    fake=tmp_path/'fake.o';native=tmp_path/'native.o';binary=tmp_path/'cooperative'
    subprocess.run(['cc','-std=c11','-Wall','-Wextra','-Werror','-c',str(HERE/'fake_vulkan_timing.c'),'-o',str(fake)],check=True)
    subprocess.run(['c++','-std=c++17','-Wall','-Wextra','-Werror','-Wno-missing-field-initializers','-DSINTERGPU_CPU_TEST_ONLY',f'-DSINTERGPU_LOCK_PATH="{lock}"','-c',str(HERE/'timing_cooperative.cpp'),'-o',str(native)],check=True)
    subprocess.run(['c++',str(native),str(fake),'-lcrypto','-o',str(binary)],check=True)
    assert 'libvulkan' not in subprocess.check_output(['ldd',str(binary)],text=True)
    return binary

@pytest.mark.parametrize('failure',[False,True])
def test_real_lease_supervision_and_fake_native(tmp_path,monkeypatch,failure):
    spec=importlib.util.spec_from_file_location('fixture_oracle',HERE/'engineering_attempt/runner.py');helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    source=helper.checked_sources('6dde8dc965cd37364b9b22c5fca1f1d2ef2870f2');arrays,packed=helper.fixture(source['fixture'])
    shader=tmp_path/'fake.spv';shader.write_bytes(bytes.fromhex('0302230701000000'));fixture=tmp_path/'input';fixture.write_bytes(packed)
    lock=tmp_path/'gpu.lock';binary=compile_fake(tmp_path,lock);journal=helper.ArtifactJournal(tmp_path/'receipt')
    lease=QuarantineLease(lock,GUARDIAN,hashlib.sha256(GUARDIAN.read_bytes()).hexdigest());lease.start_guardian()
    row=f'{GPU}, RTX, 595.99.02, 2000, 10000, 50, 100, 10, 5'
    collector=lambda:{'gpu_query':row,'compute_query':'','full_snapshot':'fake graphics'}
    policy={'driver':'595.99.02','min_sm_mhz':0,'max_sm_mhz':3000,'min_memory_mhz':0,'max_temperature_c':80,'max_power_w':300,'allowed_compute_pids':set()}
    identities={'cooperative_executable':hashlib.sha256(binary.read_bytes()).hexdigest(),'monitor_binary':hashlib.sha256(b'fake-monitor').hexdigest(),
                'old_spirv':hashlib.sha256(shader.read_bytes()).hexdigest(),'candidate_spirv':hashlib.sha256(shader.read_bytes()).hexdigest(),
                'fixture':hashlib.sha256(packed).hexdigest(),**{role:hashlib.sha256((HERE/name).read_bytes()).hexdigest() for role,name in
                (('analysis_source','timing_preregistered_analysis.py'),('monitor_source','timing_monitor.py'),('supervisor_source','timing_supervisor.py'))}}
    manifest=make_manifest('a'*40,identities,policy);prereg=tmp_path/'prereg.json';prereg_sha=write_preregistration(prereg,manifest)
    assert verify_preregistration(prereg,prereg_sha)==manifest
    monkeypatch.setenv('SINTERGPU_FAKE_POSITIVE','1');monkeypatch.setenv('SINTERGPU_FAKE_4070','1');monkeypatch.setenv('SINTERGPU_FAKE_EXPECTED_OUTPUT',str(FIXTURE/'donor.bin'))
    if failure:monkeypatch.setenv('SINTERGPU_FAKE_FAIL_WAIT','2')
    helpers={'lease_event':lambda fd,j,stage:lease.check(),'decode':helper.decode,'validate':helper.validate};counter=[]
    def launch(cmd,**kwargs):
        label='watch-'+str(len(counter));counter.append(label)
        monitor=make_policy_monitor(collector,policy,interval_s=.005,max_gap_s=3)
        supervise(cmd,monitor,journal,lease.check,lease.quarantine,label=label,pass_fds=kwargs['pass_fds'])
        return SimpleNamespace(returncode=0,stdout=(journal.root/(label+'.stdout')).read_bytes(),stderr=(journal.root/(label+'.stderr')).read_bytes())
    try:
        if failure:
            with pytest.raises(ExceptionGroup):controlled_runs(lease_fd=lease.fd,executable=binary,shader_paths=[shader,shader],fixture_path=fixture,journal=journal,helpers=helpers,parse_report=parse,arrays=arrays,controls=lambda *a:None,process=launch)
            assert lease.quarantine_required and (journal.root/'run-0.samples').exists()
            process=lease.process;lease.close()
            script='import fcntl,os,sys; f=os.open(sys.argv[1],os.O_RDWR);\ntry: fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)\nexcept BlockingIOError: sys.exit(11)\nsys.exit(0)'
            assert subprocess.run([sys.executable,'-c',script,str(lock)]).returncode==11
            process.terminate();process.wait(timeout=5)
        else:
            runs=controlled_runs(lease_fd=lease.fd,executable=binary,shader_paths=[shader,shader],fixture_path=fixture,journal=journal,helpers=helpers,parse_report=parse,arrays=arrays,controls=lambda *a:None,process=launch)
            assert len(runs)==3 and all(len(r['samples'])==150 for r in runs) and len(counter)==3
            assert verify_preregistration(prereg,prereg_sha)==manifest
            lease.release_normal()
    finally:
        journal.close()
        if lease.fd>=0:
            if lease.process is not None and lease.process.poll() is None:lease.process.terminate();lease.process.wait(timeout=5)
            os.close(lease.fd);lease.fd=-1
