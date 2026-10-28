# SPDX-License-Identifier: Apache-2.0
"""ABI-preserving export adapters for the shared inlined point-plane functions."""
from pathlib import Path
import importlib.util

CANDIDATE=Path(__file__).resolve().parents[1]
ROOT=CANDIDATE.parents[1]
spec=importlib.util.spec_from_file_location('point_base_support',CANDIDATE/'benchmarks/family_support.py')
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
check_point_oracle=base.check_point_oracle
point_oracle=base.point_oracle


def point_case(device,count,profile,*,shared_resources=True,candidate_local_size=1024):
    if candidate_local_size not in (256,1024):raise ValueError('unsupported point-map launch size')
    original=base.point_case(None,count,profile)
    helper=(CANDIDATE/'kernels/optimization/point_plane_inlined.slang').read_text()
    if profile=='aosoa':
        frozen=original['sources']['baseline']
        header=frozen[:frozen.index('float caspar_copysign')]
        # Same load/store/arithmetic phases as the generated dispatch. Keeping
        # the residual in a local before its store preserves instruction order.
        candidate=header+helper+'''
[shader("compute")][numthreads(1024,1,1)]
void main(uint3 dispatchThreadID:SV_DispatchThreadID) {
 uint i=dispatchThreadID.x;if(i>=params.problem_size)return;
 uint points_idx=points_indices[i];
 float3 n=float3(normals[i*4],normals[i*4+1],normals[i*4+2]);
 float3 p=float3(points[points_idx*4],points[points_idx*4+1],points[points_idx*4+2]);
 float3 q=float3(targets[i*4],targets[i*4+1],targets[i*4+2]);
 float r=sinterPointPlaneResidual<true>(p,q,n);
 out_0[i]=r;
 float3 rotation=sinterPointPlaneRotation(p,n);
 out_1[i*4]=rotation.x;out_1[i*4+1]=rotation.y;out_1[i*4+2]=rotation.z;
 out_1[i*4+3]=n.x;
 out_1[4*params.out_1_num_alloc+i*2]=n.y;out_1[4*params.out_1_num_alloc+i*2+1]=n.z;
}
'''
    elif profile=='scalar':
        frozen=original['sources']['baseline']
        adapter=frozen[frozen.index('[[vk::binding'):]
        candidate=helper+adapter.replace('caspar_point_plane(', 'sinterPointPlaneScalars<false>(')
    else:raise ValueError('unknown profile')
    candidate=candidate.replace('numthreads(1024,1,1)',f'numthreads({candidate_local_size},1,1)')
    candidate=candidate.replace('numthreads(1024, 1, 1)',f'numthreads({candidate_local_size}, 1, 1)')
    if device is None:
        original['sources']['candidate']=candidate
        return original
    case=base.make_case(device,dict(baseline=original['sources']['baseline'],candidate=candidate),
        original['arrays'],original['outputs'],original['params'],original['threads'])
    case.update(count=count,allocation=original['allocation'],profile=profile,family='point_plane')
    if shared_resources:
        case['buffers']['candidate']=case['buffers']['baseline']
        case['bindings']['candidate']=case['bindings']['baseline']
        def read(variant):
            # Independent poisoning is mandatory when resources are shared:
            # otherwise an unwritten candidate output could inherit baseline data.
            case['upload'](variant)
            case['dispatch'](variant);device.wait();return case['download'](variant)
        case['read']=read
    case['shared_resources']=shared_resources
    return case
