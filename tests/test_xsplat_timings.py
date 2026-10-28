"""Catch bias-correction, matrix/layout, fixture-coverage and reset mistakes."""
import importlib
import importlib.util
import copy
import numpy as np
import pytest


def recipe():
    assert importlib.util.find_spec('benchmarks.xsplat_timings'), 'xsplat timing recipe missing'
    return importlib.import_module('benchmarks.xsplat_timings')


def test_adam_reference_updates_all_state_and_clamps_step_zero():
    m=recipe()
    for step in (0,1):
        actual=m.reference.adam(np.array([1.,2.],np.float32),np.array([1.,0.],np.float32),np.zeros(2,np.float32),np.zeros(2,np.float32),step=step,learningRate=.1,beta1=.75,beta2=.75,epsilon=0.01)
        for a,e in zip(actual,([1-.1/1.01,2],[.25,0],[.25,0])):np.testing.assert_allclose(a,e,rtol=2e-7,atol=2e-7)
    actual=m.reference.adam(np.array([1.],np.float32),np.array([3.],np.float32),np.array([1.],np.float32),np.array([1.],np.float32),step=2,learningRate=.1,beta1=.5,beta2=.5,epsilon=0.)
    np.testing.assert_allclose(actual[0],[1-.1*(8/3)/np.sqrt(20/3)],rtol=2e-7)
    np.testing.assert_array_equal(actual[1],[2.]);np.testing.assert_array_equal(actual[2],[5.])


def test_projection_reference_uses_normalized_quaternion_and_rejected_depth():
    m=recipe();camera=dict(worldToCamera=np.eye(4),focal=[2,2],principal=[320,240],imageSize=[640,480])
    g=dict(mean=np.array([[0,0,2],[1,0,-1]],np.float32),logScale=np.zeros((2,3),np.float32),rotation=np.array([[0,0,0,2],[0,0,0,1]],np.float32))
    out=m.reference.project(**g,**camera)
    np.testing.assert_allclose(out[0,:6],[320,240,1/1.3,0,1/1.3,2],rtol=2e-7)
    assert out[0,6]==1
    np.testing.assert_array_equal(out[1],[0,0,0,0,0,-1,0])
    # Anisotropic covariance rotated by90deg swaps x/y scales.
    g=dict(mean=np.array([[0,0,2]],np.float32),logScale=np.log([[1,2,1]]),rotation=np.array([[0,0,1,1]],np.float32))
    np.testing.assert_allclose(m.reference.project(**g,**camera)[0,2:5],[1/4.3,0,1/1.3],rtol=2e-7,atol=1e-12)


def test_measured_fixtures_cover_all_entries_state_and_projection_branches():
    m=recipe();cases=list(m.cases())
    assert {c['operation'] for c in cases}=={'xsplat_vk.adam.main','xsplat_vk.project.projectMain','xsplat_vk.backward_probe.backwardProbeMain'}
    assert {c['params']['step'] for c in cases if c['kind']=='adam'}>={0,1,37}
    assert {c['count'] for c in cases if c['kind']=='adam'}>={31,32,1025,65539}
    for c in cases:
        expected=m.reference_call(c);m.check_parity(c,expected,expected)
        assert all(c['arrays'][k].nbytes>c['count']*c['strides'][k] for k in c['outputs'])
        if c['kind']=='project':
            visible=expected['projected'].view(m.PROJECTED_DTYPE)['visible'][:c['count']]
            assert np.any(visible==0) and np.any(visible==1)
            depths=expected['projected'].view(m.PROJECTED_DTYPE)['depth'][:c['count']]
            assert np.any(depths<0) and np.any(depths>0)
        for k in c['outputs']:
            bad={key:a.copy() for key,a in expected.items()};bad[k][-1]^=1
            with pytest.raises(AssertionError):m.check_parity(c,bad,expected)


def test_probe_gradient_is_analytic_and_all_output_tails_are_exact():
    m=recipe();c=next(c for c in m.cases() if c['kind']=='probe')
    out=m.reference_call(c)['output'].view(np.float32)
    np.testing.assert_allclose(out[:5],[.15624,-.04464,.1116,.324,.2695],rtol=2e-7,atol=2e-7)
    np.testing.assert_array_equal(out.view(np.uint8)[32:],c['arrays']['output'][32:])


def test_workload_hash_binds_camera_layout_uniforms_and_inout_seeds():
    m=recipe()
    for kind in ('adam','project','probe'):
        c=next(c for c in m.cases() if c['kind']==kind);identity=m.workload_identity(c)['sha256']
        mutations=[lambda c:c['thread_count'].__setitem__(0,c['thread_count'][0]+1),lambda c:c['arrays'][c['outputs'][0]].__setitem__(-1,0)]
        if kind=='adam':mutations.append(lambda c:c['params'].__setitem__('step',999))
        if kind=='project':mutations.append(lambda c:c['arrays']['camera'].__setitem__(0,43))
        for mutate in mutations:
            changed=copy.deepcopy(c);mutate(changed);assert m.workload_identity(changed)['sha256']!=identity


def test_cpu_collection_records_real_samples_and_scoped_reference_identity(monkeypatch):
    m=recipe();c=next(c for c in m.cases() if c['kind']=='probe')
    monkeypatch.setattr(m,'cases',lambda:iter([c]));report=m.run(iterations=2,warmups=1)
    row,=report['measurements'];assert len(row['metrics']['cpu_reference_wall_ms'])==2
    assert row['workload']['sha256']==m.workload_identity(c)['sha256']
    for key in ('runner_sha256','reference_sha256','runner_helper_sha256','shader_tree_sha256','reference_tree_sha256'):assert len(row[key])==64
    for kwargs in ({'iterations':0},{'warmups':0},{'device_index':True}):
        with pytest.raises(ValueError):m.run(**kwargs)


def test_projection_oracle_matches_frozen_cpp_on_their_shared_valid_domain(tmp_path):
    import shutil,subprocess
    from pathlib import Path
    compiler=shutil.which('g++')
    if compiler is None:pytest.skip('C++ compiler unavailable')
    m=recipe();root=Path(__file__).resolve().parents[1];donor=root/'references/donors/xsplat-vk'
    binary=tmp_path/'reference'
    subprocess.run([compiler,'-std=c++17','-O2','-I'+str(donor/'include'),str(root/'tests/native/xsplat_reference_probe.cpp'),str(donor/'src/reference.cpp'),'-o',str(binary)],check=True,capture_output=True,text=True)
    c=next(c for c in m.cases() if c['kind']=='project');n=c['count']
    g=c['arrays']['gaussians'].view(m.GAUSSIAN_DTYPE)[:n];cam=c['arrays']['camera'].view(m.CAMERA_DTYPE)[0]
    text=' '.join(map(str,cam['worldToCamera'].reshape(-1)))+'\n'+' '.join(map(str,[*cam['focal'],*cam['principal'],*cam['imageSize'],n]))+'\n'
    text+='\n'.join(' '.join(map(str,[*v['mean'],*v['logScale'],*v['rotation']])) for v in g)
    result=subprocess.run([str(binary)],input=text,capture_output=True,text=True,check=True)
    actual=np.loadtxt(result.stdout.splitlines());expected=m.reference_call(c)['projected'].view(m.PROJECTED_DTYPE)[:n]
    valid=expected['depth']>1e-4
    np.testing.assert_array_equal(actual[valid,6],expected['visible'][valid])
    for field,columns in [('meanPx',slice(0,2)),('conic',slice(2,5)),('depth',5)]:
        np.testing.assert_allclose(actual[valid,columns],expected[field][valid],rtol=2e-6,atol=2e-6)
    # Keep the incompatible rejected-depth behavior explicit, not silently unified.
    assert np.any(~valid)
    np.testing.assert_array_equal(actual[~valid],0)
    assert np.any(expected['depth'][~valid]!=0)
