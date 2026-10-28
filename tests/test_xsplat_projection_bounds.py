"""Per-input arithmetic bounds must reject wrong outputs and uncertain masks."""
import importlib,importlib.util
from decimal import Decimal,localcontext
import itertools
import numpy as np
import pytest


def model():
    assert importlib.util.find_spec('benchmarks.xsplat_projection_bounds'), 'input-derived pixel error model missing'
    return importlib.import_module('benchmarks.xsplat_projection_bounds')


def test_bounds_contain_high_precision_perspective_under_cancellation():
    m=model()
    view=np.array([[.9921977,0,.12467473,.125],[0,1,0,-.25],[-.12467473,0,.9921977,.5],[0,0,0,1]],np.float32)
    points=np.array([[30,-.9469113,3.2761207],[.25,.5,4]],np.float32)
    b=m.projected_pixel_bounds(points,view,[400,425],[320,240],[640,480])
    with localcontext() as context:
        context.prec=60
        for i,point in enumerate(points):
            c=[sum(Decimal(float(view[r,k]))*Decimal(float(point[k])) for k in range(3))+Decimal(float(view[r,3])) for r in range(3)]
            for axis in range(2):
                exact=Decimal([400,425][axis])*c[axis]/c[2]+Decimal([320,240][axis])
                assert Decimal(float(b['lower'][i,axis]))<=exact<=Decimal(float(b['upper'][i,axis]))
    assert b['classification'].tolist()==[0,1]
    inside=(b['lower']+b['upper'])/2;m.validate_pixels(inside,b['classification'],b)
    bad=inside.copy();bad[0,0]=np.nextafter(b['upper'][0,0],np.inf)
    with pytest.raises(AssertionError):m.validate_pixels(bad,b['classification'],b)


def test_uncertain_rejection_or_viewport_classification_never_accepts_either_flag():
    m=model();identity=np.eye(4,dtype=np.float32)
    for points in (np.array([[1,1,np.float32(1e-4)]],np.float32),np.array([[0,1,1]],np.float32)):
        b=m.projected_pixel_bounds(points,identity,[1,1],[0,0],[10,10])
        assert b['classification'][0]==-1
        for flag in (0,1):
            with pytest.raises(AssertionError):m.validate_pixels(np.zeros((1,2)),[flag],b)
    b=m.projected_pixel_bounds(np.array([[1,1,-1]],np.float32),identity,[1,1],[0,0],[10,10])
    assert b['classification'][0]==0
    m.validate_pixels([[0,0]],[0],b)


def test_enclosure_contains_independent_float32_operation_orders_for_all_fixture_sizes():
    from benchmarks import xsplat_timings as timing
    m=model()
    for case in timing.cases():
        if case['kind']!='project':continue
        g=case['arrays']['gaussians'].view(timing.GAUSSIAN_DTYPE)[:case['count']]
        cam=case['arrays']['camera'].view(timing.CAMERA_DTYPE)[0]
        b=m.projected_pixel_bounds(g['mean'],*(cam[k] for k in ('worldToCamera','focal','principal','imageSize')))
        assert np.all(b['classification']>=0)
        homogeneous=np.c_[g['mean'],np.ones(case['count'],np.float32)]
        terms=homogeneous[:,None,:]*cam['worldToCamera'][None,:3,:]
        for order in itertools.permutations(range(4)):
            center=np.zeros((case['count'],3),np.float32)
            for index in order:center=(center+terms[:,:,index]).astype(np.float32)
            valid=center[:,2]>np.float32(1e-4)
            with np.errstate(divide='ignore',invalid='ignore'):
                reciprocals=(np.float32(1)/center[:,2]).astype(np.float32)
                variants=[((cam['focal']*center[:,:2])*reciprocals[:,None]+cam['principal']).astype(np.float32),((center[:,:2]/center[:,2,None])*cam['focal']+cam['principal']).astype(np.float32)]
            for pixels in variants:
                pixels[~valid]=0
                flags=valid&np.all((pixels>=0)&(pixels<cam['imageSize']),axis=1)
                m.validate_pixels(pixels,flags,b)


def test_projection_parity_accepts_allowed_cancellation_but_rejects_outside_enclosure():
    from benchmarks import xsplat_timings as timing
    m=model();c=next(c for c in timing.cases() if c['kind']=='project' and c['count']==1025)
    expected=timing.reference_call(c);actual={k:v.copy() for k,v in expected.items()}
    # Frozen donor C++ output for the prepared point664, not a tuned GPU value.
    actual['projected'].view(timing.PROJECTED_DTYPE)['meanPx'][664]=[1175019.0,-49064.1836]
    timing.check_parity(c,actual,expected)
    g=c['arrays']['gaussians'].view(timing.GAUSSIAN_DTYPE)[:c['count']];cam=c['arrays']['camera'].view(timing.CAMERA_DTYPE)[0]
    b=m.projected_pixel_bounds(g['mean'],*(cam[k] for k in ('worldToCamera','focal','principal','imageSize')))
    actual['projected'].view(timing.PROJECTED_DTYPE)['meanPx'][664,0]=b['upper'][664,0]+100
    with pytest.raises(AssertionError):timing.check_parity(c,actual,expected)
