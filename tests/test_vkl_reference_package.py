"""Executable packaged Torch references, distinct from experimental Slang ports."""
import ast
import importlib
import os
import re
from pathlib import Path
import subprocess
import sys
import types

import pytest

ROOT = Path(__file__).resolve().parents[1]
DONOR = ROOT/'references/donors/vkl/src/vkl/kernels/reference'
PACKAGE = Path(os.environ.get('SINTERGPU_REFERENCE_PACKAGE_ROOT', str(ROOT/'src/sintergpu/references/vkl')))
MODULES = sorted(p.stem for p in DONOR.glob('*.py') if p.name != '__init__.py')


def test_base_import_does_not_load_torch():
    subprocess.run([sys.executable, '-c',
        "import sys; import sintergpu; assert 'torch' not in sys.modules"], check=True)


def test_optional_dependency_and_numpy_profile_without_torch():
    script = '''
import importlib.abc
import sys
class NoTorch(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'torch' or fullname.startswith('torch.'):
            raise ModuleNotFoundError('deliberately unavailable Torch', name='torch')
sys.meta_path.insert(0, NoTorch())
from sintergpu.references.vkl.numpy import mask_u32
import sintergpu.references.vkl as vkl
assert 'depth_convert' in vkl.__all__
try:
    vkl.depth_convert
except ModuleNotFoundError as error:
    assert error.name == 'torch' and 'sintergpu[torch]' in str(error)
else:
    raise AssertionError('Torch dependency must be explicit')
'''
    subprocess.run([sys.executable, '-c', script], check=True)


@pytest.mark.parametrize('name', MODULES)
def test_all_47_executable_module_bodies_unchanged(name):
    # License headers can change; no implementation, operation order, validation,
    # device placement, detach, dtype conversion or gradient semantics can drift.
    assert len(MODULES) == 47
    donor = ast.dump(ast.parse((DONOR/(name+'.py')).read_text()), include_attributes=False)
    actual = ast.dump(ast.parse((PACKAGE/(name+'.py')).read_text()), include_attributes=False)
    assert actual == donor


def raw_module(name):
    package = '_sintergpu_reference_package_oracle'
    if package not in sys.modules:
        mod = types.ModuleType(package)
        mod.__path__ = [str(DONOR)]
        sys.modules[package] = mod
    return importlib.import_module(package+'.'+name)


@pytest.fixture
def torch():
    return pytest.importorskip('torch', reason='optional Torch reference dependency not installed')


def exact(torch, actual, expected, device=None):
    if isinstance(expected, torch.Tensor):
        assert actual.shape == expected.shape and actual.dtype == expected.dtype
        assert actual.device == expected.device
        if device is not None:
            assert actual.device == device
        torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            exact(torch, actual[key], expected[key], device)
    elif isinstance(expected, (list, tuple)):
        assert type(actual) is type(expected) and len(actual) == len(expected)
        for a,e in zip(actual,expected):
            exact(torch,a,e,device)
    else:
        assert actual == expected


def test_all_modules_import_and_original_public_exports(torch):
    import sintergpu.references.vkl as vkl
    original = ast.parse((DONOR/'__init__.py').read_text())
    expected = next(ast.literal_eval(node.value) for node in original.body
                    if isinstance(node, ast.Assign) and any(isinstance(t,ast.Name) and t.id=='__all__' for t in node.targets))
    assert vkl.__all__ == expected
    for name in expected:
        assert callable(getattr(vkl,name)), name
    for name in MODULES:
        module = importlib.import_module('sintergpu.references.vkl.'+name)
        assert str(PACKAGE) in module.__file__


@pytest.mark.parametrize('capacity', [0,2,7])
def test_stable_compaction_capacity_and_autograd(torch, capacity):
    from sintergpu.references.vkl import select_compact
    source = raw_module('compact').select_compact
    values = torch.arange(30,dtype=torch.float64).reshape(2,5,3).requires_grad_()
    reference_values = values.detach().clone().requires_grad_()
    mask = torch.tensor([[True,False,True,True,False],[False,False,False,True,True]])
    codes = torch.tensor([[5,4,3,2,1],[0,1,2,3,4]],dtype=torch.int64)
    actual = select_compact(values,mask,capacity=capacity,action_codes=codes)
    expected = source(reference_values,mask,capacity=capacity,action_codes=codes)
    exact(torch,actual,expected,values.device)
    if capacity:
        a = torch.autograd.grad(actual['values'].square().sum(),values)[0]
        e = torch.autograd.grad(expected['values'].square().sum(),reference_values)[0]
        exact(torch,a,e)
        assert not a[~mask].any()


@pytest.mark.parametrize('representation', ['DEPTH','INVERSE_DEPTH','DISPARITY'])
def test_depth_conversion_status_invalid_and_autograd(torch, representation):
    from sintergpu.references.vkl import depth_convert
    source = raw_module('depth_convert').depth_convert
    values = torch.tensor([.5,2.,4.,8.],dtype=torch.float64,requires_grad=True)
    ref = values.detach().clone().requires_grad_()
    kwargs = dict(representation=representation,scale=2.,focal_length=8.,baseline=.1,
                  valid=torch.tensor([True,False,True,True]),confidence=torch.tensor([.5,.2,.8,1.],dtype=torch.float64))
    actual = depth_convert(values=values,**kwargs)
    expected = source(values=ref,**kwargs)
    exact(torch,actual,expected,values.device)
    exact(torch,torch.autograd.grad(actual['depth'].sum(),values)[0],torch.autograd.grad(expected['depth'].sum(),ref)[0])
    assert actual['metric_status'] == 'METRIC_ANCHORED'
    invalid = torch.tensor([0.,-1.,float('nan'),float('inf')])
    exact(torch,depth_convert(values=invalid,**{k:v for k,v in kwargs.items() if k not in ('valid','confidence')}),source(values=invalid,**{k:v for k,v in kwargs.items() if k not in ('valid','confidence')}))


def test_photometric_full_forward_and_parameter_gradients(torch):
    from sintergpu.references.vkl import photometric_isp
    source = raw_module('photometric_isp').photometric_isp
    generator = torch.Generator().manual_seed(2026)
    rgb = (torch.rand((2,3,3,5),generator=generator,dtype=torch.float64)*.4+.1).requires_grad_()
    kwargs = dict(exposure_ev=torch.tensor([.1,-.2],dtype=torch.float64).reshape(2,1,1,1).requires_grad_(),
                  rgb_gain=torch.full((2,3,1,1),.8,dtype=torch.float64,requires_grad=True),
                  color_matrix=torch.eye(3,dtype=torch.float64).repeat(2,1,1).requires_grad_(),
                  vignette=torch.full((2,3,3),.01,dtype=torch.float64,requires_grad=True),
                  response_gamma=torch.full((2,1,1,1),.7,dtype=torch.float64,requires_grad=True))
    inputs = [rgb,*kwargs.values()]
    refs = [x.detach().clone().requires_grad_() for x in inputs]
    actual = photometric_isp(rgb,**kwargs)
    expected = source(refs[0],**dict(zip(kwargs,refs[1:])))
    exact(torch,actual,expected,rgb.device)
    grad_actual = torch.autograd.grad(actual['rgb_observed'].square().sum(),inputs)
    grad_expected = torch.autograd.grad(expected['rgb_observed'].square().sum(),refs)
    for a,e in zip(grad_actual,grad_expected):
        exact(torch,a,e)
        assert torch.isfinite(a).all() and a.abs().sum() > 0


def test_geometry_pose_normals_and_depth_gradients(torch):
    from sintergpu.references.vkl import depth_geometry
    source = raw_module('geometry').depth_geometry
    depth = torch.full((2,1,3,5),2.,dtype=torch.float64,requires_grad=True)
    reference = depth.detach().clone().requires_grad_()
    k = torch.tensor([[[3.,0.,2.],[0.,4.,1.],[0.,0.,1.]]],dtype=torch.float64).repeat(2,1,1)
    pose = torch.eye(4,dtype=torch.float64).repeat(2,1,1);pose[1,:3,3] = torch.tensor([1.,2.,3.])
    actual = depth_geometry(depth,k,pose)
    expected = source(reference,k,pose)
    exact(torch,actual,expected,depth.device)
    assert not actual['normals'][...,-1,:].any() and not actual['normals'][...,:,-1].any()
    assert (actual['normals'][:,:,0,0] == torch.tensor([0.,0.,-1.])).all()
    loss = lambda out: out['xyz_world'].square().sum()+out['normals'].square().sum()
    exact(torch,torch.autograd.grad(loss(actual),depth)[0],torch.autograd.grad(loss(expected),reference)[0])


def test_temporal_validity_alignment_and_autograd(torch):
    from sintergpu.references.vkl import temporal_restore_rgbd
    source = raw_module('temporal').temporal_restore_rgbd
    generator = torch.Generator().manual_seed(12)
    target = torch.rand((2,3,3,5),generator=generator,dtype=torch.float64).requires_grad_()
    context = torch.rand((2,3,3,3,5),generator=generator,dtype=torch.float64).requires_grad_()
    target_ref = target.detach().clone().requires_grad_()
    context_ref = context.detach().clone().requires_grad_()
    valid = torch.ones((2,3,1,3,5),dtype=torch.bool);valid[:,1] = False
    confidence = torch.rand((2,3,1,3,5),generator=generator,dtype=torch.float64)
    confidence[0,0,0,0,0] = float('nan')
    kwargs = dict(context_valid=valid,context_confidence=confidence,alignment='rgbd_reprojected',lookahead=2)
    actual = temporal_restore_rgbd(target,context,**kwargs)
    expected = source(target_ref,context_ref,**kwargs)
    exact(torch,actual,expected,target.device)
    for a,e in zip(torch.autograd.grad(actual['rgb'].sum(),(target,context)),torch.autograd.grad(expected['rgb'].sum(),(target_ref,context_ref))):
        exact(torch,a,e)
        assert torch.isfinite(a).all()


def test_temporal_depth_support_and_trust_gates(torch):
    from sintergpu.references.vkl import temporal_depth_fuse
    source = raw_module('temporal_depth_fuse').temporal_depth_fuse
    target = torch.tensor([[[[1.,2.,0.,3.]]]])
    neighbor = target.clone()+.01
    valid = target>0
    confidence = torch.ones_like(target)*.8
    trusted = torch.tensor([[[[True,False,False,True]]]])
    args = (target,[neighbor],valid,[valid],confidence,[confidence])
    kwargs = dict(trusted=[trusted],target_trusted=trusted,consistency_threshold=.05,min_support=2,min_temporal_support=1)
    exact(torch,temporal_depth_fuse(*args,**kwargs),source(*args,**kwargs),target.device)


def test_validation_preserves_original_errors(torch):
    from sintergpu.references.vkl import select_compact, depth_convert
    pairs = [(select_compact,raw_module('compact').select_compact,dict(values=torch.ones(1,3),mask=torch.ones(1,3),capacity=2)),
             (depth_convert,raw_module('depth_convert').depth_convert,dict(values=torch.ones(3),representation='inverse_depth',scale=0))]
    for actual,expected,kwargs in pairs:
        with pytest.raises(ValueError) as e:
            expected(**kwargs)
        with pytest.raises(type(e.value),match=re.escape(str(e.value))):
            actual(**kwargs)
