"""Setup-time capacity/overflow guards, without GPU runtime dependencies."""
import importlib
import pytest


def api():
    from importlib.util import find_spec
    assert find_spec('sintergpu.vkl') is not None,'VKL setup-time dispatch/capacity planning missing'
    return importlib.import_module('sintergpu.vkl')


def test_geometry_layout_binds_optional_world_only_when_requested():
    plan=api().geometry_layout(2,17,19,max_workgroups_x=65535,world=True,confidence=True)
    assert plan['thread_count']==(646,1,1)
    assert plan['elements']['xyz_cameraOut']==1938 and plan['elements']['pose']==32
    assert plan['params']['useWorld']==1
    plain=api().geometry_layout(2,17,19,max_workgroups_x=65535)
    assert plain['elements']['xyz_worldOut']==1 and plain['elements']['pose']==1
    transformed=api().geometry_layout(2,17,19,max_workgroups_x=65535,operation='world_transform')
    assert transformed['elements']=={'xyz':1938,'pose':32,'xyz_worldOut':1938}


@pytest.mark.parametrize('kwargs',[dict(batches=1,height=65536,width=65536,max_workgroups_x=2**32),dict(batches=1,height=1,width=2**31,max_workgroups_x=2**32),dict(batches=1,height=1,width=257,max_workgroups_x=1),dict(batches=True,height=1,width=1,max_workgroups_x=1)])
def test_geometry_rejects_overflow_or_device_dispatch_limit(kwargs):
    with pytest.raises(ValueError):api().geometry_layout(**kwargs)


def test_fusion_layout_has_complete_capacity_and_provenance_outputs():
    plan=api().fusion_layout(2,1,269,4,temporal=True,max_workgroups_x=65535)
    assert plan['elements']['depths']==2152
    for name in ('support_countOut','temporal_support_countOut','conflict_owner_sourceOut','dominant_contributorOut','source_disagreementOut'):
        assert plan['elements'][name]==538
    api().validate_capacities(plan,plan['elements'])
    with pytest.raises(ValueError,match='confidenceOut'):
        api().validate_capacities(plan,{**plan['elements'],'confidenceOut':537})


@pytest.mark.parametrize('kwargs',[dict(sources=2**31),dict(sources=2**30),dict(sources=1,temporal=True),dict(sources=2,threshold=float('nan')),dict(sources=2,temporal=True,min_support=0),dict(sources=2,temporal=True,min_temporal_support=-1)])
def test_fusion_rejects_unsafe_products_or_invalid_contract(kwargs):
    with pytest.raises(ValueError):api().fusion_layout(2,1,269,max_workgroups_x=65535,**kwargs)


def test_four_registered_geometry_depth_profiles_are_discoverable():
    from sintergpu.catalog import get_operation
    for op in ['vkl.geometry.depthGeometry','vkl.geometry.worldTransform','vkl.depth_fusion.fuseDepth','vkl.depth_fusion.temporalDepthFuse']:
        record=get_operation(op)
        assert record['contract']['profile']=='vkl-geometry-depth-forward'

    assert 'mean-consistency' in get_operation('vkl.depth_fusion.fuseDepth')['contract']['numerical_behavior']
    assert 'asymmetric' in get_operation('vkl.depth_fusion.temporalDepthFuse')['contract']['numerical_behavior']
