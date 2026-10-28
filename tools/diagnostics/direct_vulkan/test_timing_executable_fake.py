# SPDX-License-Identifier: Apache-2.0
"""CPU fake dispatch copies retained expected bytes; it does not run shaders."""
import fcntl
import os
from pathlib import Path
import subprocess
import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
ARTIFACTS = ROOT / 'diagnostics/direct-point-plane-4070-1e4af42/retained-receipt.json.artifacts'
UUID = '96d2dba095fd5c00e8492e0610f24e06'

@pytest.mark.parametrize('source',['timing_executable.cpp','timing_4070_trusted.cpp','timing_cooperative.cpp'])
def test_fake_end_to_end(tmp_path,source):
    lock=tmp_path/'lock'; lock.touch()
    fake=tmp_path/'fake.o'; obj=tmp_path/'main.o'; binary=tmp_path/'timing'
    subprocess.run(['cc','-std=c11','-Wall','-Wextra','-Werror','-c',str(HERE/'fake_vulkan_timing.c'),'-o',str(fake)],check=True)
    subprocess.run(['c++','-std=c++17','-Wall','-Wextra','-Werror','-Wno-missing-field-initializers','-DSINTERGPU_CPU_TEST_ONLY',f'-DSINTERGPU_LOCK_PATH="{lock}"','-c',str(HERE/source),'-o',str(obj)],check=True)
    subprocess.run(['c++',str(obj),str(fake),'-lcrypto','-o',str(binary)],check=True)
    assert 'libvulkan' not in subprocess.check_output(['ldd',str(binary)],text=True)
    shader=tmp_path/'test.spv';shader.write_bytes(bytes.fromhex('0302230701000000'))
    events=tmp_path/'events';out=tmp_path/'samples'
    env=dict(os.environ,SINTERGPU_FAKE_POSITIVE='1',SINTERGPU_FAKE_4070='1',SINTERGPU_FAKE_EVENTS=str(events),SINTERGPU_FAKE_EXPECTED_OUTPUT=str(ARTIFACTS/'donor.bin'))
    with lock.open('r+') as lease:
        fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
        args=[str(binary),UUID,'2497233024',str(shader),str(shader),str(ARTIFACTS/'input.bin'),str(out),str(lease.fileno()),'50']
        def launch(args,env):
            if source!='timing_cooperative.cpp':return subprocess.run(args,env=env,pass_fds=(lease.fileno(),),capture_output=True,text=True),None
            r,w=os.pipe();child_env=dict(env,SINTERGPU_COOPERATIVE_FD=str(w))
            try:p=subprocess.run(args,env=child_env,pass_fds=(lease.fileno(),w),capture_output=True,text=True)
            finally:os.close(w)
            handshake=os.read(r,8);os.close(r);return p,handshake
        p,handshake=launch(args,env)
        assert p.returncode==0,p.stderr
        if handshake is not None:assert handshake==b'RC'
        lines=out.read_text().splitlines();assert len(lines)==2+50*3
        assert (tmp_path/'samples.raw').stat().st_size==50*3*2*(74268+32)
        assert sum('dispatch_fake_cpu_oracle_copy' in line for line in events.read_text().splitlines())==55*3*2
        failout=tmp_path/'failure';args[6]=str(failout)
        env['SINTERGPU_FAKE_FAIL_WAIT']='2'
        before=len(events.read_text().splitlines())
        p,handshake=launch(args,env)
        if handshake is not None:assert handshake==b'RC'
        assert p.returncode!=0 and failout.exists() and len(failout.read_text().splitlines())==2
        log=events.read_text().splitlines()[before:]
        assert log.count('queue_wait')==2 and 'timestamp_write' in log and 'dispatch_fake_cpu_oracle_copy' in log
        failed_wait=max(i for i,event in enumerate(log) if event=='queue_wait')
        assert log.index('timestamp_write') < log.index('dispatch_fake_cpu_oracle_copy') < failed_wait < log.index('device_wait') < log.index('destroy_query') < log.index('destroy_device')
        failout=tmp_path/'failure-drain';args[6]=str(failout)
        env['SINTERGPU_FAKE_FAIL_TEARDOWN']='1';before=len(events.read_text().splitlines())
        p,handshake=launch(args,env)
        if handshake is not None:assert handshake==b'R'
        assert p.returncode==2 and failout.exists() and len(failout.read_text().splitlines())==2
        log=events.read_text().splitlines()[before:]
        assert log.count('queue_wait')==2 and 'timestamp_write' in log and 'dispatch_fake_cpu_oracle_copy' in log
        assert 'device_wait' in log and 'destroy_query' not in log and 'destroy_device' not in log

        env.pop('SINTERGPU_FAKE_FAIL_TEARDOWN');env['SINTERGPU_FAKE_FAIL_WAIT']='82'
        partial=tmp_path/'partial';args[6]=str(partial)
        p,handshake=launch(args,env)
        if handshake is not None:assert handshake==b'RC'
        assert p.returncode!=0 and len(partial.read_text().splitlines())>2

        if source=='timing_cooperative.cpp':
            env.pop('SINTERGPU_FAKE_FAIL_WAIT');env['SINTERGPU_FAKE_INTERRUPT_WAIT']='2'
            cooperative=tmp_path/'cooperative';args[6]=str(cooperative);before=len(events.read_text().splitlines())
            p,handshake=launch(args,env)
            assert handshake==b'RC'
            assert p.returncode==1 and 'cooperative sampled-control abort' in p.stderr
            log=events.read_text().splitlines()[before:];assert log.count('queue_wait')==2
            assert log.index('device_wait')<log.index('destroy_query')<log.index('destroy_device')
