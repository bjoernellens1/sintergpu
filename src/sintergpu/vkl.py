"""Setup-time layouts for portable VKL ports; no GPU imports or buffer work.

Counts describe scalar32 elements, not bytes. Callers allocate and bind resources,
validate tensor contents/camera contracts and establish storage visibility. Query
max_workgroups_x from the existing device; this module never chooses a device.
"""
import math
import numbers

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


def _resampling_params(batches,height,width,output_height=None,output_width=None):
    return dict(batches=batches,height=height,width=width,
                outputHeight=height if output_height is None else output_height,
                outputWidth=width if output_width is None else output_width,
                channels=1,firstStep=1,useDirect=0,useConfidence=1,minSupport=2,
                scaleX=1.0,scaleY=1.0,maxRGBDifference=.15,maxRelativeSpread=.1)


def depth_lift_layout(batches,height,width,output_height,output_width,*,max_workgroups_x):
    inputs=_dispatch(batches,height,width,max_workgroups_x=UINT_MAX)
    count=_dispatch(batches,output_height,output_width,max_workgroups_x)
    if max(height,width,output_height,output_width)>2**24:
        raise ValueError('float32 nearest coordinates require dimensions<=2**24')
    params=_resampling_params(batches,height,width,output_height,output_width)
    params.update(scaleX=width/output_width,scaleY=height/output_height)
    return dict(entrypoint='depthUpsample',thread_count=(count,1,1),params=params,
                elements=dict(depth=inputs,valid=inputs,confidence=inputs,
                              **{k+'Out':count for k in ('depth','valid','confidence','direct_valid','propagated_valid','provenance')}))


def depth_densify_plan(batches,height,width,*,max_workgroups_x,channels=3,steps=1,direct=False,
                       max_rgb_difference=.15,max_relative_spread=.1,min_support=2):
    count=_dispatch(batches,height,width,max_workgroups_x)
    if channels not in (1,3) or type(channels) is not int:raise ValueError('RGB channels must be1or3')
    if count*channels>UINT_MAX or max(height,width)>INT_MAX-1:raise ValueError('RGB offsets or neighbor coordinates overflow')
    _integer('steps',steps,1,INT_MAX);_integer('min_support',min_support,1,INT_MAX)
    if type(direct) is not bool:raise ValueError('direct must beboolean')
    for name,value in [('max_rgb_difference',max_rgb_difference),('max_relative_spread',max_relative_spread)]:
        if isinstance(value,bool) or not isinstance(value,numbers.Real) or value<0:raise ValueError(name+' must be nonnegative float32')
    plans=[]
    for n in range(steps):
        params=_resampling_params(batches,height,width)
        params.update(channels=channels,firstStep=int(n==0),useDirect=int(direct or n>0),minSupport=min_support,
                      maxRGBDifference=float(max_rgb_difference),maxRelativeSpread=float(max_relative_spread))
        plans.append(dict(entrypoint='densifyDepth',thread_count=(count,1,1),params=params,
                          elements=dict(depth=count,valid=count,confidence=count,rgb=channels*count,
                                        direct_valid=count if direct or n>0 else 1,
                                        **{k+'Out':count for k in ('depth','valid','confidence','direct_valid','propagated_valid','provenance','support')})))
    return plans


def depth_pyramid_plan(batches,height,width,*,max_workgroups_x,scale=None,confidence=True):
    count=_dispatch(batches,height,width,max_workgroups_x)
    if max(height,width)>INT_MAX-1:raise ValueError('pyramid ceil dimensions overflow')
    if type(confidence) is not bool:raise ValueError('confidence must beboolean')
    if scale is not None and (type(scale) is not int or scale not in (2,4,8)):raise ValueError('scale must be2,4or8')
    params=_resampling_params(batches,height,width);params['useConfidence']=int(confidence)
    plans=[dict(level=1,shape=(batches,1,height,width),entrypoint='initializeDepth',thread_count=(count,1,1),params=params,
                elements=dict(depth=count,valid=count,confidence=count if confidence else 1,
                              depthOut=count,validOut=count,confidenceOut=count))]
    level=1
    while level<(8 if scale is None else scale) and (height,width)!=(1,1):
        oh,ow=(height+1)//2,(width+1)//2;output=_dispatch(batches,oh,ow,max_workgroups_x);level*=2
        plans.append(dict(level=level,shape=(batches,1,oh,ow),entrypoint='depthPyramidLevel',thread_count=(output,1,1),
                          params=dict(batches=batches,height=height,width=width,epsilon=1.1920928955078125e-7),
                          elements=dict(depth=count,valid=count,confidence=count,depthOut=output,validOut=output,confidenceOut=output)))
        height,width,count=oh,ow,output
    if scale is not None and plans[-1]['level']!=scale:raise ValueError('scale exceeds available depth pyramid levels')
    return plans


def depth_downsample_plan(batches,height,width,*,scale,max_workgroups_x):
    stages=depth_pyramid_plan(batches,height,width,scale=scale,max_workgroups_x=max_workgroups_x)
    shape=stages[-1]['shape'];count=batches*shape[-2]*shape[-1];level_index=len(stages)-1
    labels=dict(entrypoint='labelDirectDepth',thread_count=(count,1,1),params=_resampling_params(batches,*shape[-2:]),
                shape=shape,level=scale,elements=dict(valid=count,direct_validOut=count,propagated_validOut=count,provenanceOut=count))
    stages.append(labels)
    return dict(stages=stages,shape=shape,outputs={**{name:dict(stage=level_index,binding=name+'Out') for name in ('depth','valid','confidence')},
                **{name:dict(stage=level_index+1,binding=name+'Out') for name in ('direct_valid','propagated_valid','provenance')}})
