# SPDX-License-Identifier: Apache-2.0
"""Point-plane qualification plumbing; not a library runtime.

Ported from candidate2def63b family_support/inlined_point_support, Apache-2.0.

The baseline sources are extracted unchanged from hash-frozen packaged exports.
All device creation/allocation/submission/readback is explicit at this boundary.
"""
from pathlib import Path
import hashlib

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
KERNELS = ROOT / 'src/sintergpu/kernels'


def function(source, name):
    start = source.index('void ' + name + '(')
    brace = source.index('{', start)
    depth = 1
    end = brace + 1
    while depth:
        depth += (source[end] == '{') - (source[end] == '}')
        end += 1
    return source[start:end]


def compile_kernel(device, source, entry='main'):
    module = device.load_module_from_source('family_' + hashlib.sha256(source.encode()).hexdigest()[:16], source)
    program = device.link_program([module], [module.entry_point(entry)])
    return device.create_compute_kernel(program)


def buffer(device, array):
    import slangpy as spy
    array = np.ascontiguousarray(array)
    return device.create_buffer(element_count=array.size, struct_size=4,
        usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access, data=array.reshape(-1))


def make_case(device, sources, arrays, output_names, params, threads):
    if device is None:
        return dict(sources=sources,arrays=arrays,outputs=output_names,params=params,threads=threads)
    kernels = {name: compile_kernel(device, source) for name, source in sources.items()}
    buffers = {variant: {name: buffer(device, array) for name, array in arrays.items()} for variant in kernels}
    bindings = {variant: {**values, **params} for variant, values in buffers.items()}
    def dispatch(variant, **kwargs):
        kernels[variant].dispatch(thread_count=threads, vars=bindings[variant], **kwargs)
    def upload(variant):
        for name, array in arrays.items():
            buffers[variant][name].copy_from_numpy(np.ascontiguousarray(array).reshape(-1))
    def download(variant):
        return {name: buffers[variant][name].to_numpy().view(arrays[name].dtype).reshape(arrays[name].shape).copy()
            for name in output_names}
    def read(variant):
        upload(variant)
        dispatch(variant)
        device.wait()
        return download(variant)
    return dict(device=device, kernels=kernels, buffers=buffers, bindings=bindings,
        sources=sources, arrays=arrays, outputs=output_names, dispatch=dispatch,
        upload=upload, download=download, read=read, threads=threads,
        uploaded_bytes=sum(a.nbytes for a in arrays.values()),
        readback_bytes=sum(arrays[name].nbytes for name in output_names))


def point_case(device, count, profile, *, local_size=1024):
    if local_size not in (256,1024):raise ValueError("unsupported local_size")
    if profile not in ("aosoa","scalar"):raise ValueError("unsupported profile")
    packed=False
    rng = np.random.default_rng(2101 + count)
    allocation = count if packed else count + 7
    point_count = max(1, count // 2)
    points = rng.normal(size=(point_count, 4)).astype(np.float32)
    targets = rng.normal(size=(allocation, 4)).astype(np.float32)
    normals = rng.normal(size=(allocation, 4)).astype(np.float32)
    indices = rng.integers(0, point_count, count, dtype=np.uint32)
    # Deliberate duplicate gathers, signed zero and exact zero residual.
    indices[:min(4, count)] = 0
    points[0] = [0, -0., 0, -77]
    targets[0, :3] = 0
    normals[0, :3] = [0, 0, 1]
    frozen = (KERNELS / 'caspar_slang/composed/point_plane_residual.slang').read_text()
    header = frozen[:frozen.index('float caspar_copysign')]
    scalar = function((KERNELS / 'caspar_slang/composed/reprojection.slang').read_text(), 'caspar_point_plane')
    def adapter(call):
        return header + '''
[shader("compute")][numthreads(1024,1,1)]
void main(uint3 tid:SV_DispatchThreadID) {
 uint i=tid.x;if(i>=params.problem_size)return;uint p=points_indices[i];
 float3 point=float3(points[p*4],points[p*4+1],points[p*4+2]);
 float3 target=float3(targets[i*4],targets[i*4+1],targets[i*4+2]);
 float3 normal=float3(normals[i*4],normals[i*4+1],normals[i*4+2]);
 float r,j0,j1,j2,j3,j4,j5;
''' + call + '''
 out_0[i]=r;out_1[i*4]=j0;out_1[i*4+1]=j1;out_1[i*4+2]=j2;out_1[i*4+3]=j3;
 out_1[4*params.out_1_num_alloc+i*2]=j4;out_1[4*params.out_1_num_alloc+i*2+1]=j5;
}
'''
    baseline = frozen.replace('void computeMain(', 'void main(') if profile == 'aosoa' else scalar + adapter('''
 caspar_point_plane(point.x,point.y,point.z,target.x,target.y,target.z,normal.x,normal.y,normal.z,r,j0,j1,j2,j3,j4,j5);
''')
    if profile=='aosoa':
        shared=header+'''
[shader("compute")][numthreads(1024,1,1)]
void main(uint3 tid:SV_DispatchThreadID) {
 uint i=tid.x;if(i>=params.problem_size)return;
 uint p=points_indices[i];
 float3 n=float3(normals[i*4],normals[i*4+1],normals[i*4+2]);
 float3 point=float3(points[p*4],points[p*4+1],points[p*4+2]);
 float3 q=float3(targets[i*4],targets[i*4+1],targets[i*4+2]);
 float r=sinterPointPlaneResidual<true>(point,q,n);
 out_0[i]=r;
 float3 rotation=sinterPointPlaneRotation(point,n);
 out_1[i*4]=rotation.x;out_1[i*4+1]=rotation.y;out_1[i*4+2]=rotation.z;out_1[i*4+3]=n.x;
 out_1[4*params.out_1_num_alloc+i*2]=n.y;out_1[4*params.out_1_num_alloc+i*2+1]=n.z;
}
'''
    else:
        shared=adapter('''
 sinterPointPlaneScalars<false>(point.x,point.y,point.z,target.x,target.y,target.z,normal.x,normal.y,normal.z,r,j0,j1,j2,j3,j4,j5);
''')
    candidate='import optimization.point_plane;\n'+shared.replace('numthreads(1024,1,1)',f'numthreads({local_size},1,1)')
    arrays = dict(points=points, targets=targets, normals=normals, points_indices=indices,
        out_0=np.full(allocation+7, -901, np.float32), out_1=np.full(6 * allocation+7, -902, np.float32))
    params = dict(params=dict(problem_size=count, points_num_alloc=point_count,
        targets_num_alloc=allocation, normals_num_alloc=allocation,
        out_0_num_alloc=allocation, out_1_num_alloc=allocation))
    case = make_case(device, dict(baseline=baseline, candidate=candidate), arrays,
        ('out_0', 'out_1'), params, [count, 1, 1])
    case.update(count=count, allocation=allocation, profile=profile, family='point_plane')
    return case


def point_oracle(case):
    arrays = case['arrays']; count = case['count']
    p = arrays['points'][arrays['points_indices'], :3].astype(np.float64)
    q = arrays['targets'][:count, :3].astype(np.float64)
    n = arrays['normals'][:count, :3].astype(np.float64)
    return np.sum(n * (p - q), axis=1), np.c_[np.cross(p, n), n]


def check_point_oracle(case, actual):
    r, jacobian = point_oracle(case)
    count, allocation = case['count'], case['allocation']
    live = np.c_[actual['out_1'][:count*4].reshape(count, 4),
        actual['out_1'][4*allocation:4*allocation+count*2].reshape(count, 2)]
    np.testing.assert_allclose(actual['out_0'][:count], r, atol=2e-6, rtol=3e-6)
    np.testing.assert_allclose(live, jacobian, atol=2e-6, rtol=3e-6)
    np.testing.assert_array_equal(actual['out_0'][count:], -901)
    np.testing.assert_array_equal(actual['out_1'][count*4:allocation*4], -902)
    np.testing.assert_array_equal(actual['out_1'][4*allocation+count*2:], -902)
