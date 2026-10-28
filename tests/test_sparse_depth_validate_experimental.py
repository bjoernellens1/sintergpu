"""Frozen-donor CPU model and compile/reflection tests, no GPU execution."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import shutil
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'src/sintergpu/kernels/experimental/sparse_depth_validate.slang'
DONOR=ROOT/'references/donors/vkl/src/vkl/kernels/reference/sparse_geometry.py'


def test_donor_identity():
    assert hashlib.sha256(DONOR.read_bytes()).hexdigest()=='c56224f9313c21c0fa4315ae09f14a1e43e61417a3cd4819a5ac096395a56fc9'


def scalar_model(depth,pixels,pixel_valid,threshold,minimum):
    b,h,w=depth.shape; points=pixels.shape[1]
    out={name:np.zeros((b,points),dtype=dtype) for name,dtype in [('depth',np.float32),('valid',bool),('local_dispersion',np.float32),('edge_risk',np.float32),('neighbor_count',np.uint32),('confidence',np.float32)]}
    for batch in range(b):
        for i,(x,y) in enumerate(pixels[batch]):
            if x<0 or y<0 or x>=w or y>=h: continue
            vals=sorted(depth[batch,yy,xx] for yy in range(max(0,y-1),min(h,y+2)) for xx in range(max(0,x-1),min(w,x+2)) if np.isfinite(depth[batch,yy,xx]) and depth[batch,yy,xx]>0)
            count=len(vals);out['neighbor_count'][batch,i]=count
            if not count: continue
            median=vals[(count-1)//2]; deviations=sorted(np.float32(abs(np.float32(v-median))) for v in vals)
            denominator=max(median,np.finfo(np.float32).eps)
            with np.errstate(over='ignore', invalid='ignore'):
                dispersion=np.float32(deviations[(count-1)//2]/denominator); edge=np.float32(np.float32(vals[-1]-vals[0])/denominator)
            center=depth[batch,y,x]
            ok=bool(pixel_valid[batch,i] and np.isfinite(center) and center>0 and count>=minimum and np.isfinite(dispersion) and np.isfinite(edge) and dispersion<=threshold and edge<=threshold)
            out['local_dispersion'][batch,i]=np.nan_to_num(dispersion);out['edge_risk'][batch,i]=np.nan_to_num(edge)
            if ok:
                out['depth'][batch,i]=center;out['valid'][batch,i]=True
                out['confidence'][batch,i]=np.float32(np.float32(count/9)*np.float32(1-np.clip(max(dispersion,edge),0,1)))
    return out


@pytest.mark.parametrize('scenario',['random','empty','flat','extreme','edges'])
def test_model_matches_donor(scenario):
    torch=pytest.importorskip('torch')
    rng=np.random.default_rng(886);d=rng.uniform(.1,4,(2,4,5)).astype(np.float32)
    p=np.array([[[0,0],[1,1],[4,3],[-1,1],[5,2],[2,2]],[[2,2],[0,3],[4,0],[2,-1],[1,4],[3,2]]],np.int64)
    enabled=np.array([[1,1,1,1,1,0],[1,1,0,1,1,1]],bool)
    if scenario=='empty': d[:]=0;d[:,1,1]=np.nan;d[:,2,2]=np.inf
    if scenario=='flat': d[:]=2
    if scenario=='extreme': d[:]=np.float32(1e-30);d[:,1,1]=np.finfo(np.float32).max
    if scenario=='edges': d[:]=2;d[:,1,1]=2.2;d[:,1,2]=np.nan;enabled[:,0]=False
    spec=importlib.util.spec_from_file_location('frozen_sparse_geometry',DONOR);donor=importlib.util.module_from_spec(spec);spec.loader.exec_module(donor)
    threshold=np.float32(.1);ref=donor.sparse_depth_validate(torch.from_numpy(d[:,None]),torch.from_numpy(p),pixels_valid=torch.from_numpy(enabled),max_relative_dispersion=float(threshold),min_neighbors=3)
    actual=scalar_model(d,p,enabled,threshold,3)
    for name in actual:
        np.testing.assert_allclose(actual[name],ref[name].numpy(),atol=2e-6,rtol=2e-6,err_msg=name)


def test_lower_median_diagnostics_and_disabled_center():
    torch=pytest.importorskip('torch')
    spec=importlib.util.spec_from_file_location('frozen_sparse_geometry_edges',DONOR);donor=importlib.util.module_from_spec(spec);spec.loader.exec_module(donor)
    d=np.array([[[1.,2.],[3.,4.]]],np.float32);p=np.array([[[0,0],[1,1],[-1,0]]],np.int64);enabled=np.array([[True,False,True]])
    result=donor.sparse_depth_validate(torch.from_numpy(d[:,None]),torch.from_numpy(p),pixels_valid=torch.from_numpy(enabled),max_relative_dispersion=2,min_neighbors=1)
    actual=scalar_model(d,p,enabled,np.float32(2),1)
    np.testing.assert_array_equal(actual['neighbor_count'],[[4,4,0]])
    np.testing.assert_allclose(actual['local_dispersion'],[[.5,.5,0]])
    np.testing.assert_allclose(actual['edge_risk'],[[1.5,1.5,0]])
    np.testing.assert_array_equal(actual['valid'],[[True,False,False]])
    for name in actual: np.testing.assert_allclose(actual[name],result[name].numpy(),atol=1e-6)


@pytest.mark.parametrize('version',['2026.12','2026.19'])
def test_compile_and_reflect(version,tmp_path):
    key='SINTERGPU_SLANGC_'+version.replace('.','_');compiler=Path(os.environ.get(key,ROOT/'.tools'/('slang-'+version)/'bin/slangc'))
    if not compiler.is_file():
        if key in os.environ: pytest.fail(f'{key} compiler missing: {compiler}')
        pytest.skip(f'Slang {version} unavailable')
    p=subprocess.run([str(compiler),'-version'],capture_output=True,text=True,check=True);assert (p.stdout+p.stderr).strip()==version
    refl=tmp_path/'reflection.json';p=subprocess.run([str(compiler),str(SOURCE),'-entry','sparseDepthValidate','-target','spirv','-profile','glsl_460','-o',str(tmp_path/'kernel.spv'),'-reflection-json',str(refl)],capture_output=True,text=True)
    assert p.returncode==0,p.stdout+p.stderr
    j=json.loads(refl.read_text());ep=j['entryPoints'][0];assert ep['threadGroupSize']==[64,1,1]
    expected=['params','depth','pixels','pixelsValid','depthOut','validOut','dispersionOut','edgeRiskOut','neighborCountOut','confidenceOut']
    assert {b['name'] for b in ep['bindings'] if b['binding'].get('used')==1}==set(expected)
    ps={p['name']:p for p in j['parameters']}; assert {n:ps[n]['binding']['index'] for n in ps}==dict(zip(expected,range(10)))
    fields=ps['params']['type']['elementType']['fields']
    assert [(f['name'],f['type']['scalarType'],f['binding']['offset']) for f in fields]==[(n,t,i*4) for i,(n,t) in enumerate([('batches','uint32'),('points','uint32'),('height','uint32'),('width','uint32'),('minNeighbors','uint32'),('maxRelativeDispersion','float32')])]
    for name in ps.keys()-{'params'}:
        t=ps[name]['type'];assert t['baseShape']=='structuredBuffer';assert (t.get('access')=='readWrite')==name.endswith('Out')
        assert ps[name]['binding'].get('space',0)==0
        result=t['resultType']
        if name=='pixels':
            assert result['kind']=='vector' and result['elementCount']==2 and result['elementType']['scalarType']=='int32'
        else:
            assert result['scalarType']==('uint32' if name in {'pixelsValid','validOut','neighborCountOut'} else 'float32')
    assert ps['params']['binding'].get('space',0)==0
    dis=os.environ.get('SINTERGPU_SPIRV_DIS') if 'SINTERGPU_SPIRV_DIS' in os.environ else shutil.which('spirv-dis')
    if not dis and 'SINTERGPU_SPIRV_DIS' not in os.environ: pytest.skip('SPIR-V disassembler unavailable')
    if not dis or not Path(dis).is_file() or not os.access(dis,os.X_OK): pytest.fail(f'SINTERGPU_SPIRV_DIS is not an executable: {dis}')
    assembly=subprocess.run([dis,str(tmp_path/'kernel.spv')],capture_output=True,text=True,check=True).stdout
    assert 'OpDecorate %_runtimearr_v2int ArrayStride 8' in assembly
    assert 'OpDecorate %_runtimearr_float ArrayStride 4' in assembly
    assert 'OpDecorate %_runtimearr_uint ArrayStride 4' in assembly


def test_launch_bounds():
    from sintergpu.sparse_depth_validate_contract import checked_sparse_depth_validate_launch as plan
    p=plan(2,5,4,5,3,.1,100);assert (p.depth_elements,p.point_elements,p.groups)==(40,10,1);assert p.max_relative_dispersion==float(np.float32(.1))
    for args in [(1,0,1,1,1,.1,10),(1,1,1,1,10,.1,10),(1,1,1,1,1,float('nan'),10),(1,1,1,1,1,1e100,10),(1,65,1,1,1,.1,1),(1,1,1,1<<31,1,.1,100),(1,1<<30,1,1,1,.1,100)]:
        with pytest.raises(ValueError): plan(*args)


def test_threshold_endpoint_is_inclusive():
    torch=pytest.importorskip('torch')
    spec=importlib.util.spec_from_file_location('frozen_sparse_geometry_threshold',DONOR);donor=importlib.util.module_from_spec(spec);spec.loader.exec_module(donor)
    d=np.array([[[1.,1.125]]],np.float32);p=np.array([[[0,0]]],np.int64);enabled=np.ones((1,1),bool)
    for threshold,expected in [(np.float32(.125),True),(np.nextafter(np.float32(.125),np.float32(0)),False)]:
        ref=donor.sparse_depth_validate(torch.from_numpy(d[:,None]),torch.from_numpy(p),pixels_valid=torch.from_numpy(enabled),max_relative_dispersion=float(threshold),min_neighbors=2)
        model=scalar_model(d,p,enabled,threshold,2)
        assert model['valid'].item() is expected
        assert ref['valid'].item() is expected
        np.testing.assert_array_equal(model['edge_risk'],ref['edge_risk'].numpy())


def test_reference_cli_reports_pending(capsys):
    from sintergpu.cli import main
    operation='vkl.sparse_depth_validate.sparseDepthValidate'
    assert main(['test',operation,'--backend','reference'])==2
    result=json.loads(capsys.readouterr().out)
    assert result['status']=='pending'
    assert result['unavailable_operations']==[operation]
    assert result['cases']==[]
