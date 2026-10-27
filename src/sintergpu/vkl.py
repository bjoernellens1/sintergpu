"""Setup-time layouts for portable VKL ports; no GPU imports or buffer work.

Counts describe scalar32 elements, not bytes. Callers allocate and bind resources,
validate tensor contents/camera contracts and establish storage visibility. Query
max_workgroups_x from the existing device; this module never chooses a device.
"""
import math

UINT_MAX=0xffffffff
INT_MAX=0x7fffffff


def _integer(name,value,minimum=0,maximum=UINT_MAX):
    if type(value) is not int or not minimum<=value<=maximum:
        raise ValueError(f'{name} must be an integer in [{minimum},{maximum}]')
    return value


def _dispatch(batches,height,width,max_workgroups_x):
    for name,value,minimum in [('batches',batches,0),('height',height,1),('width',width,1),('max_workgroups_x',max_workgroups_x,1)]:
        _integer(name,value,minimum)
    count=batches*height*width
    if count>UINT_MAX:raise ValueError('pixel count exceeds uint32 offsets')
    if (count+255)//256>max_workgroups_x:raise ValueError('pixel count exceeds device X dispatch limit')
    return count


def geometry_layout(batches,height,width,*,max_workgroups_x,operation='depth_geometry',world=False,confidence=False):
    count=_dispatch(batches,height,width,max_workgroups_x)
    if 3*count>UINT_MAX or 16*batches>UINT_MAX:raise ValueError('geometry buffer offsets overflow uint32')
    if type(world) is not bool or type(confidence) is not bool:raise ValueError('world/confidence must be booleans')
    params=dict(batches=batches,height=height,width=width,useWorld=int(world),useConfidence=int(confidence))
    if operation=='world_transform':
        entry='worldTransform';elements=dict(xyz=3*count,pose=16*batches,xyz_worldOut=3*count)
    elif operation=='depth_geometry':
        entry='depthGeometry'
        elements=dict(depth=count,intrinsics=9*batches,pose=16*batches if world else 1,
                      confidence=count if confidence else 1,validOut=count,rays_cameraOut=3*count,
                      xyz_cameraOut=3*count,normalsOut=3*count,geometry_confidenceOut=count,
                      xyz_worldOut=3*count if world else 1)
    else:raise ValueError('unknown geometry operation')
    return dict(entrypoint=entry,thread_count=(count,1,1),params=params,elements=elements)


def fusion_layout(batches,height,width,sources,*,max_workgroups_x,temporal=False,relative=False,threshold=.05,min_support=2,min_temporal_support=1):
    count=_dispatch(batches,height,width,max_workgroups_x)
    _integer('sources',sources,2 if temporal else 1,INT_MAX)
    if count*sources>UINT_MAX:raise ValueError('source fields overflow uint32 offsets')
    if type(temporal) is not bool or type(relative) is not bool:raise ValueError('temporal/relative must be booleans')
    if isinstance(threshold,bool) or not math.isfinite(threshold) or threshold<=0:raise ValueError('threshold must be positive finite')
    _integer('min_support',min_support,1,INT_MAX);_integer('min_temporal_support',min_temporal_support,0,INT_MAX)
    params=dict(count=count,sources=sources,relative=int(relative),minSupport=min_support,minTemporalSupport=min_temporal_support,threshold=threshold)
    elements=dict(depths=sources*count,valids=sources*count,confidences=sources*count,
                  depthOut=count,validOut=count,confidenceOut=count,winning_sourceOut=count)
    if temporal:
        elements.update({k:sources*count for k in ('visibility','consistency','trusted')})
        elements.update({k:count for k in ('support_countOut','temporal_support_countOut','depth_varianceOut',
                        'conflict_owner_sourceOut','dominant_contributorOut','temporal_weight_fractionOut','source_disagreementOut')})
    return dict(entrypoint='temporalDepthFuse' if temporal else 'fuseDepth',thread_count=(count,1,1),params=params,elements=elements)


def validate_capacities(layout,capacities):
    """Validate caller-supplied scalar32 capacities without reading GPU contents.

    Types, device descriptor range limits, resource offset alignment and overlapping storage hazards must still be
    checked by the caller's binding adapter; capacities alone cannot prove them.
    """
    for name,count in layout['elements'].items():
        capacity=capacities.get(name)
        if type(capacity) is not int or capacity<count:raise ValueError(f'insufficient scalar32 capacity for {name}: need {count}')


def temporal_best_layout(batches,neighbors,*,max_workgroups_x,quality=False,confidence=False):
    count=_dispatch(batches,1,1,max_workgroups_x)
    _integer('neighbors',neighbors,1,INT_MAX)
    if count*neighbors>UINT_MAX:raise ValueError('neighbor fields overflow uint32 offsets')
    if type(quality) is not bool or type(confidence) is not bool:raise ValueError('quality/confidence must be booleans')
    return dict(entrypoint='temporalBest',thread_count=(count,1,1),
                params=dict(batches=count,neighbors=neighbors,useQuality=int(quality),useConfidence=int(confidence)),
                elements=dict(equivalence=count*neighbors,target_quality=count,neighbor_quality=1 if quality else count*neighbors,
                              pairwise_quality=count*neighbors if quality else 1,
                              pairwise_confidence=count*neighbors if confidence else 1,
                              best_indexOut=count,best_qualityOut=count,replaceabilityOut=count,replacement_confidenceOut=count))
