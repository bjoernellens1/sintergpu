# SPDX-License-Identifier: Apache-2.0
"""Synthetic loader interception; no GPU or real driver calls."""
import fcntl
import hashlib
import os
from pathlib import Path
import subprocess
import pytest

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
COMPILER=Path(os.environ.get('SINTERGPU_SLANGC','/home/bjoern/.local/opt/slang-2026.19/bin/slangc'))
SHA='b34c971681eba7c1c7d9a885a13223fc3d481ab72fa39f2b9bf0ddbda2f00f76'

def test_cpu_fake_preflight_and_rejections(tmp_path):
    import numpy as np
    if not COMPILER.is_file(): pytest.skip('pinned Slang compiler unavailable')
    if hashlib.sha256(COMPILER.read_bytes()).hexdigest()!='19c0217f848537db2a87120eb1b3f882ab2e3eb54fdec4cf5300035943704145': pytest.fail('compiler identity mismatch')
    donor=tmp_path/'donor.spv'
    subprocess.run([str(COMPILER),str(ROOT/'src/sintergpu/kernels/caspar_slang/composed/point_plane_residual.slang'),'-entry','computeMain','-target','spirv','-o',str(donor)],check=True)
    assert hashlib.sha256(donor.read_bytes()).hexdigest()==SHA
    lock=tmp_path/'lock';exe=tmp_path/'probe';fake=tmp_path/'fake.o';events=tmp_path/'events';output=tmp_path/'out'
    subprocess.run(['cc','-c',str(HERE.parent/'fake_vulkan_preflight.c'),'-o',str(fake)],check=True)
    subprocess.run(['c++','-std=c++17','-DSINTERGPU_CPU_TEST_ONLY',f'-DSINTERGPU_LOCK_PATH="{lock}"',str(HERE/'point_plane.cpp'),str(fake),'-o',str(exe),'-lcrypto'],check=True)
    links=subprocess.run(['ldd',str(exe)],check=True,capture_output=True,text=True).stdout
    assert 'libvulkan' not in links
    with np.load(ROOT/'src/sintergpu/fixtures/point_plane_aosoa/point_plane_aosoa_1025.npz') as z:
        packed=b''.join(z[k].tobytes() for k in ('points','targets','normals','out_0','out_1','points_indices'))
    inp=tmp_path/'input';inp.write_bytes(packed)
    env=dict(os.environ, SINTERGPU_FAKE_EVENTS=str(events),VK_DRIVER_FILES='/nonexistent/no-icd.json',VK_ICD_FILENAMES='/nonexistent/no-icd.json')
    fd=os.open(lock,os.O_CREAT|os.O_RDWR,0o600)
    try:
        fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
        base=[str(exe),'96d2dba095fd5c00e8492e0610f24e06','2497233024',str(donor),SHA,'main','2',str(inp),str(output),str(fd)]
        def run(args):return subprocess.run(args,env=env,pass_fds=(fd,),capture_output=True,text=True)
        bad=base.copy();bad[4]='0'*64;assert run(bad).returncode!=0;assert not events.exists()
        bad=base.copy();bad[2]='4294967296';assert run(bad).returncode!=0;assert not events.exists()
        bad=base.copy();bad[1]='00'*16;assert run(bad).returncode!=0;assert not events.exists()
        bad=base.copy();bad[9]='4294967296';assert run(bad).returncode!=0;assert not events.exists()
        corrupt=tmp_path/'corrupt';corrupt.write_bytes(packed[:-4]+(512).to_bytes(4,'little'))
        bad=base.copy();bad[7]=str(corrupt);assert run(bad).returncode!=0;assert not events.exists()
        p=run(base);assert p.returncode!=0 and 'UUID/driver/vendor/device mismatch' in p.stderr
        assert events.read_text().splitlines()==['create_instance','enumerate','enumerate','properties','destroy_instance']
        assert not output.exists()
        # Fake backend copies an independently computed CPU oracle; validates only
        # native buffer/descriptor/barrier/readback orchestration, never GPU math.
        with np.load(ROOT/'src/sintergpu/fixtures/point_plane_aosoa/point_plane_aosoa_1025.npz') as z:
            a={k:z[k].copy() for k in z.files}
        pnt=a['points'][a['points_indices'],:3].astype(np.float64)
        target=a['targets'][:1025,:3].astype(np.float64);normal=a['normals'][:1025,:3].astype(np.float64)
        a['out_0'][:1025]=np.sum(normal*(pnt-target),axis=1)
        jac=np.c_[np.cross(pnt,normal),normal].astype(np.float32)
        a['out_1'][:4100]=jac[:,:4].reshape(-1);a['out_1'][4128:6178]=jac[:,4:].reshape(-1)
        expected=tmp_path/'expected';expected.write_bytes(b''.join(a[k].tobytes() for k in ('points','targets','normals','out_0','out_1','points_indices')))
        events.unlink();env['SINTERGPU_FAKE_POSITIVE']='1';env['SINTERGPU_FAKE_4070']='1';env['SINTERGPU_FAKE_EXPECTED_OUTPUT']=str(expected)
        good=base.copy();result=run(good)
        assert result.returncode==0, result.stderr
        assert output.read_bytes()==expected.read_bytes()
        trace=events.read_text().splitlines()
        assert trace.count('allocate_memory')==7 and trace.count('free_memory')==7
        assert trace.count('create_buffer')==7 and trace.count('destroy_buffer')==7
        assert trace.count('barrier')==2 and trace.count('dispatch_fake_cpu_oracle_copy')==1
        assert 'update_descriptors_unique' in trace and trace[-2:]==['destroy_device','destroy_instance']
        output.unlink();events.unlink();env['SINTERGPU_FAKE_FAIL_SUBMIT']='1'
        failed=run(good);assert failed.returncode!=0 and not output.exists()
        trace=events.read_text().splitlines()
        assert 'queue_submit' in trace and 'queue_wait' not in trace
        assert trace.count('free_memory')==7 and trace[-2:]==['destroy_device','destroy_instance']
        events.unlink();env.pop('SINTERGPU_FAKE_FAIL_SUBMIT');env['SINTERGPU_FAKE_FAIL_TEARDOWN']='1'
        failed=run(good);assert failed.returncode!=0 and not output.exists()
        trace=events.read_text().splitlines()
        assert 'queue_wait' in trace and trace.count('free_memory')==7 and trace[-2:]==['destroy_device','destroy_instance']
    finally:os.close(fd)

def test_cpu_test_build_cannot_link_only_real_vulkan(tmp_path):
    exe=tmp_path/'bad-test-build'
    result=subprocess.run(['c++','-std=c++17','-DSINTERGPU_CPU_TEST_ONLY',str(HERE/'point_plane.cpp'),'-o',str(exe),'-lvulkan','-lcrypto'],capture_output=True,text=True)
    assert result.returncode!=0 and 'sintergpu_test_only_fake_vulkan_marker' in result.stderr
