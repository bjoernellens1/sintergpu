# SPDX-License-Identifier: Apache-2.0
"""Setup-only typed feature layouts and bounded streaming matching recipes."""
import math
import struct

UINT_MAX=0xffffffff


def _integer(name,x,minimum=1):
    if type(x) is not int or not minimum<=x<=UINT_MAX:raise ValueError('invalid '+name)
    return x


def _buffers(elements,dtypes,limit):
    _integer('max_storage_buffer_range',limit)
    for k,v in elements.items():
        if v>UINT_MAX or v*(8 if dtypes[k]=='int64' else 4)>limit:raise ValueError('buffer extent exceeds device/index capacity: '+k)


def _stage(entry,count,params,bindings,limit,*,group=False,module='binary_match'):
    threads=count*256 if group else count
    if threads>UINT_MAX or (threads+255)//256>limit:raise ValueError('device X dispatch limit exceeded')
    return dict(operation=f'vkl.{module}.{entry}',entrypoint=entry,thread_count=(threads,1,1),params=dict(params),bindings=tuple(bindings))


def brief_layout(batches,channels,height,width,points,bits=256,*,max_workgroups_x,max_storage_buffer_range,default_pairs=True,use_pixels_valid=False,use_rgb_valid=False):
    for n,v,lo in [('batches',batches,1),('channels',channels,1),('height',height,1),('width',width,1),('points',points,0),('bits',bits,1),('max_workgroups_x',max_workgroups_x,1)]:_integer(n,v,lo)
    if channels not in (1,3) or any(type(v) is not bool for v in (default_pairs,use_pixels_valid,use_rgb_valid)):raise ValueError('invalid channel/flags')
    if default_pairs and bits>961*960:raise ValueError('default BRIEF pattern exhausted')
    n=batches*points;pixels=batches*height*width
    elements=dict(rgb=pixels*channels,pixels=n*2,pixelsValid=n if use_pixels_valid else 1,rgbValid=pixels if use_rgb_valid else 1,pairs=1 if default_pairs else bits*4,descriptorOut=n*bits,briefValidOut=n)
    dtypes={k:'int64' if k in ('pixels','pairs') else 'float32' if k=='rgb' else 'uint32' for k in elements}
    _buffers(elements,dtypes,max_storage_buffer_range)
    params=dict(batches=batches,channels=channels,height=height,width=width,points=points,bits=bits,defaultPairs=int(default_pairs),usePixelsValid=int(use_pixels_valid),useRgbValid=int(use_rgb_valid))
    stages=[_stage('briefDescribe',n,params,elements,max_workgroups_x,group=True,module='brief')] if n else []
    return dict(elements=elements,dtypes=dtypes,stages=stages,operations=('vkl.brief.briefDescribe',))


class _MatchStages:
    def __init__(self,p,limit):self.p=p;self.limit=limit
    def __len__(self):return 3+((self.p['candidates']+self.p['chunkCount']-1)//self.p['chunkCount'])*(2+self.p['mutual'])
    def __iter__(self):
        p=self.p;b=p['batches'];m=p['landmarks'];n=p['candidates'];limit=self.limit
        yield _stage('packBinary',b*(m+n)*p['words'],p,('landmarkDescriptor','candidateDescriptor','landmarkPacked','candidatePacked'),limit)
        yield _stage('initializeMatch',b*m,p,('bestDistance','bestIndex'),limit)
        for start in range(0,n,p['chunkCount']):
            q=p|dict(chunkStart=start,chunkCount=min(p['chunkCount'],n-start))
            yield _stage('matchPairs',b*m*q['chunkCount'],q,('landmarkPacked','candidatePacked','projected','candidateXY','landmarkValid','candidateValid','pairDistance'),limit)
            yield _stage('rowMinimum',b*m,q,('pairDistance','bestDistance','bestIndex'),limit,group=True)
            if p['mutual']:yield _stage('reverseMinimum',b*q['chunkCount'],q,('pairDistance','reverseIndex'),limit,group=True)
        yield _stage('finalizeMatch',b*m,p,('bestDistance','bestIndex','reverseIndex','landmarkValid','projected','matchedIndexOut','distanceOut','matchValidOut'),limit)


def binary_match_plan(batches,landmarks,candidates,bits,*,radius_px,max_hamming_distance,max_workgroups_x,max_storage_buffer_range,chunk_size=None,mutual=True,use_landmark_valid=False,use_candidate_valid=False):
    for n,v in [('batches',batches),('landmarks',landmarks),('candidates',candidates),('bits',bits),('max_workgroups_x',max_workgroups_x)]:_integer(n,v)
    if bits==UINT_MAX or type(max_hamming_distance) is not int or not 0<=max_hamming_distance<=bits:raise ValueError('invalid Hamming threshold')
    if isinstance(radius_px,bool) or not isinstance(radius_px,(int,float)) or not math.isfinite(radius_px) or radius_px<=0:raise ValueError('positive finite radius required')
    if any(type(v) is not bool for v in (mutual,use_landmark_valid,use_candidate_valid)):raise ValueError('invalid mask/mutual flags')
    if chunk_size is None:chunk_size=candidates
    else:_integer('chunk_size',chunk_size)
    chunk=min(chunk_size,candidates);b=batches;m=landmarks;n=candidates;words=(bits+31)//32
    elements=dict(landmarkDescriptor=b*m*bits,candidateDescriptor=b*n*bits,projected=2*b*m,candidateXY=2*b*n,landmarkValid=b*m if use_landmark_valid else 1,candidateValid=b*n if use_candidate_valid else 1,landmarkPacked=b*m*words,candidatePacked=b*n*words,pairDistance=b*m*chunk,bestDistance=b*m,bestIndex=b*m,reverseIndex=b*n if mutual else 1,matchedIndexOut=b*m,distanceOut=b*m,matchValidOut=b*m)
    dtypes={k:'int64' if k in ('candidateXY','matchedIndexOut','distanceOut') else 'float32' if k=='projected' else 'uint32' for k in elements}
    _buffers(elements,dtypes,max_storage_buffer_range)
    value=radius_px*radius_px
    try:radius_squared=struct.unpack('<f',struct.pack('<f',value))[0]
    except OverflowError:radius_squared=math.inf
    p=dict(batches=b,landmarks=m,candidates=n,bits=bits,words=words,chunkStart=0,chunkCount=chunk,maxDistance=max_hamming_distance,mutual=int(mutual),useLandmarkValid=int(use_landmark_valid),useCandidateValid=int(use_candidate_valid),radiusSquared=radius_squared)
    # Check worst extents without materializing an enormous host command list.
    for count,group in ((b*(m+n)*words,False),(b*m,True),(b*m*chunk,False))+(((b*chunk,True),) if mutual else ()):_stage('check',count,p,(),max_workgroups_x,group=group)
    operations=tuple('vkl.binary_match.'+e for e in ('packBinary','initializeMatch','matchPairs','rowMinimum','reverseMinimum','finalizeMatch') if mutual or e!='reverseMinimum')
    return dict(elements=elements,dtypes=dtypes,stages=_MatchStages(p,max_workgroups_x),operations=operations)


def record_feature_plan(kernels,buffers,plan,*,command_encoder):
    if command_encoder is None:raise ValueError('caller-owned command encoder required')
    if not len(plan['stages']):return
    if any(op not in kernels for op in plan['operations']):raise ValueError('prepare every kernel before recording')
    if any(k not in buffers for k,v in plan['elements'].items() if v):raise ValueError('missing caller-owned buffer')
    for index,stage in enumerate(plan['stages']):
        bindings={k:buffers[k] for k in stage['bindings']};bindings['Params']=stage['params']
        kernels[stage['operation']].dispatch(thread_count=stage['thread_count'],vars=bindings,command_encoder=command_encoder)
        if index+1<len(plan['stages']):command_encoder.global_barrier()
