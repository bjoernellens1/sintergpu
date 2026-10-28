"""Independent CPU donor model and ABI compilation, not shader execution."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'src/sintergpu/kernels/experimental/dense_depth_regularize.slang'
DONOR=ROOT/'references/donors/vkl/src/vkl/kernels/reference/dense_depth_regularize.py'


def test_donor_identity():
    assert hashlib.sha256(DONOR.read_bytes()).hexdigest()=='586e2b2e8dfd2a0991f836fb640a7c976fefca14c39dcb6abbf64d44f52d3351'


def scalar_model(depth, valid, confidence, rgb, rgb_limit, depth_limit, minimum):
    b,h,w=depth.shape
    out=np.zeros_like(depth); output_valid=np.zeros_like(valid); output_conf=np.zeros_like(depth)
    direct=np.zeros_like(valid); support=np.zeros((b,h,w),np.uint32)
    for batch in range(b):
        for y in range(h):
            for x in range(w):
                def gray(yy,xx):
                    if rgb.shape[1]==1: return rgb[batch,0,yy,xx]
                    return np.float32(np.float32(np.float32(rgb[batch,0,yy,xx]*np.float32(.299))+np.float32(rgb[batch,1,yy,xx]*np.float32(.587)))+np.float32(rgb[batch,2,yy,xx]*np.float32(.114)))
                center=gray(y,x); values=[]; conf=[]
                for yy in range(max(0,y-1),min(h,y+2)):
                    for xx in range(max(0,x-1),min(w,x+2)):
                        z=depth[batch,yy,xx]
                        if valid[batch,yy,xx] and np.isfinite(z) and z>0 and abs(np.float32(gray(yy,xx)-center))<=rgb_limit:
                            values.append(z);conf.append(confidence[batch,yy,xx])
                count=len(values); support[batch,y,x]=count; values.sort()
                median=values[(count-1)//2] if count else depth[batch,y,x]
                z=depth[batch,y,x]
                direct[batch,y,x]=bool(valid[batch,y,x] and np.isfinite(z) and z>0 and count>=minimum and abs(np.float32(z-median))<=np.float32(depth_limit*max(z,np.finfo(np.float32).eps)))
                ok=count>=minimum and np.isfinite(median) and median>0
                if ok:
                    out[batch,y,x]=median;output_valid[batch,y,x]=True
                    output_conf[batch,y,x]=np.float32(np.float32(np.sum(conf,dtype=np.float32)/np.float32(count))*np.float32(np.float32(count)/np.float32(9)))
    return {'depth':out,'valid':output_valid,'confidence':output_conf,'direct_valid':direct,'support':support}


@pytest.mark.parametrize('channels',[1,3])
@pytest.mark.parametrize('scenario',['random','invalid','outlier','boundaries'])
def test_model_matches_frozen_donor(channels,scenario):
    torch=pytest.importorskip('torch')
    rng=np.random.default_rng(622)
    d=rng.uniform(.1,4,(2,4,5)).astype(np.float32); v=rng.random((2,4,5))>.2
    conf=rng.uniform(0,1,d.shape).astype(np.float32); rgb=rng.uniform(0,1,(2,channels,4,5)).astype(np.float32)
    rgb_limit=np.float32(.15); depth_limit=np.float32(.15); minimum=4
    if scenario=='invalid':
        d[:,0,:4]=[0,-1,np.nan,np.inf]; v[:]=False; rgb[:,0,1,1]=np.nan
    if scenario=='outlier':
        d[:]=1;d[:,1,2]=10;v[:]=True;rgb[:]=.5;conf[:,2,2]=0
    if scenario=='boundaries':
        d[:]=1;v[:]=True;rgb[:]=0;rgb[:,:,0,0]=.15;minimum=1
    spec=importlib.util.spec_from_file_location('frozen_regularize',DONOR); module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    expected=module.regularize_dense_depth(torch.from_numpy(d[:,None]),torch.from_numpy(v[:,None]),torch.from_numpy(conf[:,None]),torch.from_numpy(rgb),max_rgb_difference=float(rgb_limit),max_relative_deviation=float(depth_limit),min_support=minimum)
    actual=scalar_model(d,v,conf,rgb,rgb_limit,depth_limit,minimum)
    for name,a in actual.items():
        np.testing.assert_allclose(a,expected[name].numpy()[:,0],atol=2e-7,rtol=2e-6,err_msg=name)


@pytest.mark.parametrize('version',['2026.12','2026.19'])
def test_compile_and_reflect(version,tmp_path):
    key='SINTERGPU_SLANGC_'+version.replace('.','_'); compiler=Path(os.environ.get(key,ROOT/'.tools'/('slang-'+version)/'bin/slangc'))
    if not compiler.is_file():
        if key in os.environ: pytest.fail(f'{key} compiler missing: {compiler}')
        pytest.skip(f'Slang {version} unavailable')
    p=subprocess.run([str(compiler),'-version'],capture_output=True,text=True,check=True); assert (p.stdout+p.stderr).strip()==version
    spv=tmp_path/'test.spv'; refl=tmp_path/'reflection.json'
    p=subprocess.run([str(compiler),str(SOURCE),'-entry','regularizeDenseDepth','-target','spirv','-profile','glsl_460','-o',str(spv),'-reflection-json',str(refl)],capture_output=True,text=True)
    assert p.returncode==0,p.stdout+p.stderr
    j=json.loads(refl.read_text()); ep=j['entryPoints'][0];assert ep['threadGroupSize']==[64,1,1]
    assert {b['name'] for b in ep['bindings'] if b['binding'].get('used')==1}=={'params','depth','valid','confidence','rgb','depthOut','validOut','confidenceOut','directValidOut','supportOut'}
    ps={p['name']:p for p in j['parameters']}; fields=ps['params']['type']['elementType']['fields']
    assert [(f['name'],f['type']['scalarType'],f['binding']['offset']) for f in fields]==[(n,t,i*4) for i,(n,t) in enumerate([('batches','uint32'),('height','uint32'),('width','uint32'),('channels','uint32'),('minSupport','uint32'),('maxRgbDifference','float32'),('maxRelativeDeviation','float32')])]
    assert {name:ps[name]['binding']['index'] for name in ps}==dict(zip(['params','depth','valid','confidence','rgb','depthOut','validOut','confidenceOut','directValidOut','supportOut'],range(10)))
    for name in ps.keys()-{'params'}:
        t=ps[name]['type']; assert t['baseShape']=='structuredBuffer'; assert t['resultType']['scalarType']==('uint32' if name in {'valid','validOut','directValidOut','supportOut'} else 'float32'); assert (t.get('access')=='readWrite')==name.endswith('Out')


def test_launch_contract_bounds():
    from sintergpu.dense_depth_regularize_contract import checked_dense_depth_regularize_launch as plan
    p=plan(2,4,5,3,4,.15,.15,100); assert (p.pixels,p.rgb_elements,p.groups)==(40,120,1);assert p.max_rgb_difference==float(np.float32(.15))
    for args in [(1,1,1,2,1,.1,.1,10),(1,1,1,1,10,.1,.1,10),(1,1,1,1,1,float('nan'),.1,10),(1,1,1,1,1,1e100,.1,10),(1,1,65,1,1,.1,.1,1),(1,1,1000000000,3,1,.1,.1,100)]:
        with pytest.raises(ValueError): plan(*args)


def test_donor_and_model_hole_fill_and_even_lower_median():
    torch=pytest.importorskip('torch')
    spec=importlib.util.spec_from_file_location('frozen_regularize_edges',DONOR)
    donor=importlib.util.module_from_spec(spec);spec.loader.exec_module(donor)
    for valid,expected_depth,expected_support,expected_direct in [
        ([[[False,True]]],[[[4.,4.]]],[[[1,1]]],[[[False,True]]]),
        ([[[True,True]]],[[[1.,1.]]],[[[2,2]]],[[[True,False]]]),
    ]:
        d=np.array([[[1.,4.]]],np.float32); v=np.array(valid,bool)
        c=np.ones_like(d);rgb=np.zeros((1,1,1,2),np.float32)
        ref=donor.regularize_dense_depth(torch.from_numpy(d[:,None]),torch.from_numpy(v[:,None]),torch.from_numpy(c[:,None]),torch.from_numpy(rgb),min_support=1)
        model=scalar_model(d,v,c,rgb,np.float32(.15),np.float32(.15),1)
        for name,expected in [('depth',expected_depth),('support',expected_support),('direct_valid',expected_direct)]:
            np.testing.assert_array_equal(ref[name].numpy()[:,0],expected)
            np.testing.assert_array_equal(model[name],expected)


def test_donor_and_model_single_pixel_and_nonfinite_luminance():
    torch=pytest.importorskip('torch')
    spec=importlib.util.spec_from_file_location('frozen_regularize_single',DONOR)
    donor=importlib.util.module_from_spec(spec);spec.loader.exec_module(donor)
    d=np.array([[[2.]]],np.float32);v=np.ones((1,1,1),bool);c=np.array([[[.9]]],np.float32)
    for gray,expected_support in [(0.,1),(float('nan'),0),(float('inf'),0)]:
        rgb=np.array([[[[gray]]]],np.float32)
        with np.errstate(invalid='ignore'):
            model=scalar_model(d,v,c,rgb,np.float32(.15),np.float32(.15),1)
        ref=donor.regularize_dense_depth(torch.from_numpy(d[:,None]),torch.from_numpy(v[:,None]),torch.from_numpy(c[:,None]),torch.from_numpy(rgb),min_support=1)
        for name in ('depth','valid','confidence','direct_valid','support'):
            np.testing.assert_allclose(model[name],ref[name].numpy()[:,0],atol=2e-7,rtol=2e-6)
        assert model['support'].item()==expected_support
        assert model['valid'].item()==bool(expected_support)
