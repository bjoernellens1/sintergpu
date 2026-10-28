"""CPU reduction-model comparisons and Slang ABI checks; never executes GPU."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
KERNELS = ROOT / 'src/sintergpu/kernels'
SOURCE = KERNELS / 'experimental/depth_evidence.slang'
DONOR = ROOT / 'references/donors/vkl/src/vkl/kernels/reference/depth_evidence.py'


def model(depth, valid, confidence, gh, gw, near, far):
    # Independent scalar reduction model; bin boundaries follow ceil(grid*h/gh).
    h,w = depth.shape
    cells = np.zeros((gh,gw,8), dtype=np.float64)
    global_totals = np.zeros(8, dtype=np.float64)
    for y in range(h):
        for x in range(w):
            z=float(depth[y,x]); usable=bool(valid[y,x] and np.isfinite(z)); d=z if usable else 0.0
            c=float(confidence[y,x]) if usable else 0.0
            diffs=[]
            for yy,xx in ((y,x+1),(y+1,x)):
                if usable and yy<h and xx<w and valid[yy,xx] and np.isfinite(depth[yy,xx]):
                    diffs.append(float(depth[yy,xx])-d)
            edge=any(abs(delta)>max(abs(d),1e-6)*.25 for delta in diffs)
            f=np.array([usable,usable and d<near,usable and d>far,d,d*d,edge,c,usable and not edge and c>0])
            cells[y*gh//h,x*gw//w]+=f
            global_totals += [usable,usable and d<near,usable and d>far,edge,c,usable and not edge and c>0,len(diffs),sum(v*v for v in diffs)]
    s,n,f,e,c,stable,pairs,sq=global_totals
    g=np.array([s/(h*w),1-s/(h*w),n/max(s,1),f/max(s,1),sq/(4*max(pairs,1)),e/max(s,1),c/max(s,1),stable/max(s,1)])
    for gy in range(gh):
        for gx in range(gw):
            y0=(gy*h+gh-1)//gh; y1=((gy+1)*h+gh-1)//gh
            x0=(gx*w+gw-1)//gw; x1=((gx+1)*w+gw-1)//gw
            s=cells[gy,gx,0]; cells[gy,gx]/=max(s,1); cells[gy,gx,0]=s/((y1-y0)*(x1-x0))
    return g,cells


@pytest.mark.parametrize('kind', ['random', 'invalid', 'thresholds'])
@pytest.mark.parametrize('use_confidence', [False, True])
def test_reduction_model_against_frozen_donor(kind, use_confidence):
    torch=pytest.importorskip('torch')
    rng=np.random.default_rng(12534)
    depth=rng.uniform(-.5,10,(2,3,5)).astype(np.float32)
    valid=rng.random((2,3,5))>.25
    confidence=rng.uniform(0,1,(2,3,5)).astype(np.float32) if use_confidence else np.ones((2,3,5),np.float32)
    if kind=='invalid':
        valid[:]=False; depth[0,0,:2]=[np.nan,np.inf]
    elif kind=='thresholds':
        depth[:]=1; depth[:,0,:]=[.1,8.,1.,1.25,1.250001]
        valid[:]=True; confidence[:,1,1]=0 if use_confidence else 1
    spec=importlib.util.spec_from_file_location('depth_evidence_donor',DONOR)
    donor=importlib.util.module_from_spec(spec); spec.loader.exec_module(donor)
    tensor=lambda a: torch.from_numpy(a)[:,None,:,:]
    result=donor.depth_evidence(tensor(depth),tensor(valid),tensor(confidence) if use_confidence else None,grid_size=(2,3),near=.1,far=8.)
    names=['valid_fraction','hole_fraction','near_fraction','far_fraction','local_variance','edge_fraction','confidence_mean','stable_fraction']
    for b in range(2):
        global_values, grid_values=model(depth[b],valid[b],confidence[b],2,3,.1,8.)
        np.testing.assert_allclose([result[n][b].item() for n in names],global_values,atol=2e-6,rtol=2e-6)
        np.testing.assert_allclose(result['grid_evidence'][b].numpy(),grid_values,atol=2e-6,rtol=2e-6)


@pytest.mark.parametrize('version', ['2026.12','2026.19'])
@pytest.mark.parametrize('entry', ['depthEvidenceGlobal','depthEvidenceGrid'])
def test_dispatch_compiles_and_reflects(version,entry,tmp_path):
    key='SINTERGPU_SLANGC_'+version.replace('.','_')
    compiler=Path(os.environ.get(key,ROOT/'.tools'/('slang-'+version)/'bin/slangc'))
    if not compiler.is_file():
        if key in os.environ: pytest.fail(f'{key} compiler missing: {compiler}')
        pytest.skip(f'Slang {version} unavailable')
    identity=subprocess.run([str(compiler),'-version'],capture_output=True,text=True,check=True)
    assert (identity.stdout+identity.stderr).strip()==version
    reflection=tmp_path/'reflect.json'
    proc=subprocess.run([str(compiler),str(SOURCE),'-I',str(KERNELS),'-entry',entry,'-target','spirv','-profile','glsl_460','-o',str(tmp_path/'kernel.spv'),'-reflection-json',str(reflection)],capture_output=True,text=True)
    assert proc.returncode==0,proc.stdout+proc.stderr
    data=json.loads(reflection.read_text())
    ep=data['entryPoints'][0]
    assert ep['threadGroupSize']==[64,1,1]
    used={b['name'] for b in ep['bindings'] if b['binding'].get('used')==1}
    assert used=={'params','depth','valid','confidence', 'globalOut' if entry.endswith('Global') else 'gridOut'}
    parameters={p['name']: p for p in data['parameters']}
    fields=parameters['params']['type']['elementType']['fields']
    assert [(f['name'], f['type']['scalarType'], f['binding']['offset']) for f in fields] == [(name, typ, i*4) for i,(name,typ) in enumerate([('batches','uint32'),('height','uint32'),('width','uint32'),('gridHeight','uint32'),('gridWidth','uint32'),('useConfidence','uint32'),('nearBound','float32'),('farBound','float32')])]
    assert {n:parameters[n]['binding']['index'] for n in parameters} == {'params':0,'depth':1,'valid':2,'confidence':3,'globalOut':4,'gridOut':5}
    for name in ('depth','valid','confidence','globalOut','gridOut'):
        typ=parameters[name]['type']
        assert typ['baseShape']=='structuredBuffer'
        assert typ['resultType']['scalarType']==('uint32' if name=='valid' else 'float32')
        assert (typ.get('access')=='readWrite') == name.endswith('Out')


def test_checked_launch_limits_and_clamped_grid():
    from sintergpu.depth_evidence_contract import checked_depth_evidence_launch as launch
    p=launch(2,3,5,99,99,.1,8,1024)
    assert (p.grid_height,p.grid_width,p.input_elements,p.grid_elements,p.global_groups,p.grid_groups)==(3,5,30,240,1,1)
    assert p.near==float(np.float32(.1)) and p.far==8.0
    with pytest.raises(ValueError,match='pair counters'):
        launch(1,1,8388609,1,1,.1,8,1024)
    with pytest.raises(ValueError,match='byte range'):
        launch(1024,1024,1024,1,1,.1,8,1024)
    with pytest.raises(ValueError,match='device'):
        launch(65,1,1,1,1,.1,8,1)
    for near,far in [(float('nan'),8),(.1,float('inf')),(1,1),(-1,1)]:
        with pytest.raises(ValueError,match='bounds'):
            launch(1,1,1,1,1,near,far,1024)
    with pytest.raises(ValueError,match='float32'):
        launch(1,1,1,1,1,1,1+1e-10,1024)
    with pytest.raises(ValueError,match='float32'):
        launch(1,1,1,1,1,0,1e100,1024)
    with pytest.raises(ValueError,match='positive integers'):
        launch(True,1,1,1,1,.1,8,1024)


@pytest.mark.parametrize('entry', ['depthEvidenceGlobal', 'depthEvidenceGrid'])
def test_reference_cli_stays_pending_for_unqualified_stages(entry, capsys):
    from sintergpu.cli import main
    operation = 'vkl.depth_evidence.' + entry
    assert main(['test', operation, '--backend', 'reference']) == 2
    result = json.loads(capsys.readouterr().out)
    assert result['status'] == 'pending'
    assert result['unavailable_operations'] == [operation]
    assert result['cases'] == []
