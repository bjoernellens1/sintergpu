# SPDX-License-Identifier: Apache-2.0
"""Shared *test/benchmark* plumbing; not a library runtime or numerical oracle.

The baseline sources are extracted unchanged from hash-frozen packaged exports.
All device creation/allocation/submission/readback is explicit at this boundary.
"""
from pathlib import Path
import hashlib

import numpy as np

CANDIDATE = Path(__file__).resolve().parents[1]
ROOT = CANDIDATE.parents[1]
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
        dispatch(variant)
        device.wait()
        return download(variant)
    return dict(device=device, kernels=kernels, buffers=buffers, bindings=bindings,
        sources=sources, arrays=arrays, outputs=output_names, dispatch=dispatch,
        upload=upload, download=download, read=read, threads=threads,
        uploaded_bytes=sum(a.nbytes for a in arrays.values()),
        readback_bytes=sum(arrays[name].nbytes for name in output_names))


def point_case(device, count, profile, *, packed=False):
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
    helper = (CANDIDATE / 'kernels/optimization/point_plane.slang').read_text()
    candidate = helper + adapter('''
 SinterPointPlaneFactor f=sinterPointPlane<''' + str(profile == 'aosoa').lower() + '''>(point,target,normal);
 r=f.residual;j0=f.rotation.x;j1=f.rotation.y;j2=f.rotation.z;
 j3=f.translation.x;j4=f.translation.y;j5=f.translation.z;
''')
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


def solve_case(device, dimension, kind='spd', *, schur=None):
    n = dimension
    schur = n != 6 if schur is None else schur
    rng = np.random.default_rng(100 + n)
    a = rng.normal(size=(n, n))
    h = (a.T @ a + np.eye(n) * 2).astype(np.float32)
    b = rng.normal(size=n).astype(np.float32)
    damping = .01
    observations = 100
    if kind == 'singular': h[:] = 0; damping = 0
    elif kind == 'indefinite': h[0, 0] = -1
    elif kind == 'nonfinite': h[0, 0] = np.nan
    elif kind == 'rhs-nonfinite': b[0] = np.inf
    elif kind == 'few-observations': observations = 0
    elif kind == 'damping': h[:] = 0; damping = .5
    elif kind == 'upper-poison': h[np.triu_indices(n,1)]=np.nan
    terms = np.r_[h.reshape(-1), b, np.float32(12), np.float32(observations)].astype(np.float32)
    source = (KERNELS / ('slang_rgbd_slam/dense_ba.slang' if schur else 'caspar_slang/gn.slang')).read_text()
    name = 'solveSchur' if schur else 'solveSystem'
    baseline_function = function(source, name).replace('void ' + name + '(', 'void main(')
    capacity = 18 if schur else 6
    header = '''
StructuredBuffer<float> inputTerms;
RWStructuredBuffer<float> delta;
RWStructuredBuffer<uint> status;
'''
    if schur:
        header += 'struct Config {uint terms;uint activeFrames;uint dofs;float damping;};ConstantBuffer<Config> cfg;\n'
        params = dict(cfg=dict(terms=len(terms), activeFrames=n//6+1, dofs=n, damping=damping))
        arguments = 'cfg.dofs,cfg.damping,inputTerms[cfg.terms-1]>=float(6*(cfg.activeFrames-1))'
        size = 'cfg.dofs'
    else:
        header += 'struct SolveParams {float damping;};ConstantBuffer<SolveParams> solveParams;\n'
        params = dict(solveParams=dict(damping=damping))
        arguments = '6,solveParams.damping,true'
        size = '6'
    baseline = header + '[shader("compute")][numthreads(1,1,1)]\n' + baseline_function
    candidate = (CANDIDATE / 'kernels/optimization/small_cholesky.slang').read_text() + header + f'''
[shader("compute")][numthreads(1,1,1)]
void main() {{
 float x[{capacity}];bool valid=sinterSmallCholeskyCore<{capacity}>(inputTerms,0,{arguments},x);
 status[0]=valid?0u:1u;for(uint i=0;i<{size};++i)delta[i]=valid?x[i]:0.0;
}}
'''
    arrays = dict(inputTerms=terms, delta=np.full(n+7, -903, np.float32), status=np.full(3, 999, np.uint32))
    case = make_case(device, dict(baseline=baseline, candidate=candidate), arrays, ('delta','status'), params, [1,1,1])
    case.update(dimension=n, schur=schur, kind=kind, h=h, b=b, damping=damping,
        enough_observations=observations>=n or not schur, family='small_cholesky')
    return case


def solve_oracle(case):
    lower = np.tril(case['h'].astype(np.float64))
    h = lower + lower.T - np.diag(np.diag(lower)) + np.eye(case['dimension']) * case['damping']
    b = case['b'].astype(np.float64)
    if not case['enough_observations'] or not np.isfinite(h).all() or not np.isfinite(b).all(): return None
    try:
        np.linalg.cholesky(h)
        return np.linalg.solve(h, -b)
    except np.linalg.LinAlgError:
        return None


def check_solve_oracle(case, actual):
    n = case['dimension']; expected = solve_oracle(case)
    assert actual['status'][0] == (1 if expected is None else 0)
    if expected is None: np.testing.assert_array_equal(actual['delta'][:n], 0)
    else: np.testing.assert_allclose(actual['delta'][:n], expected, atol=2e-6, rtol=4e-6)
    np.testing.assert_array_equal(actual['delta'][n:], -903)
    np.testing.assert_array_equal(actual['status'][1:], 999)


def fusion_case(device, count, tree):
    point = point_case(None, count, 'aosoa', packed=True)
    arrays = dict(point['arrays'])
    partial_counts = [(count+63)//64]
    while partial_counts[-1]>1: partial_counts.append((partial_counts[-1]+63)//64)
    for level, groups in enumerate(partial_counts):
        arrays['partial'+str(level)] = np.full(groups*43, -904, np.float32)
    arrays.update(delta=np.full(13,-903,np.float32),status=np.full(3,999,np.uint32))
    gn = (KERNELS/'caspar_slang/gn.slang').read_text()
    fused = (CANDIDATE/'kernels/optimization/point_plane_gn.slang').read_text()
    for name in ('point_plane','normal_terms'):
        fused = fused.replace('#include "'+name+'.slang"', (CANDIDATE/'kernels/optimization'/f'{name}.slang').read_text())
    entry = 'pointPlaneGNWorkgroup' if tree=='workgroup' else 'pointPlaneGNSubgroup'
    residual_entry = 'reduceResidual' if tree=='workgroup' else 'reduceResidualSubgroup'
    k = dict(point=compile_kernel(device,point['sources']['baseline']), fused=compile_kernel(device,fused,entry),
        build=compile_kernel(device,gn,residual_entry), partial=compile_kernel(device,gn,'reducePartials'),
        solve=compile_kernel(device,gn,'solveSystem'))
    buffers = {variant: {name:buffer(device,array) for name,array in arrays.items()} for variant in ('baseline','candidate')}
    def dispatch(variant, **kwargs):
        b = buffers[variant]; encoder = device.create_command_encoder()
        pool = kwargs.get('query_pool')
        if pool is not None: encoder.write_timestamp(pool,kwargs['query_index_before'])
        if variant=='baseline':
            binds = {name:b[name] for name in point['arrays']}
            binds.update(point['params'])
            k['point'].dispatch(thread_count=[count,1,1],vars=binds,command_encoder=encoder)
            k['build'].dispatch(thread_count=[count,1,1],vars=dict(residual=b['out_0'],jacobian=b['out_1'],
                outputTerms=b['partial0'],params=dict(count=count)),command_encoder=encoder)
        else:
            binds = {name:b[name] for name in ('points','targets','normals','points_indices','out_0','out_1')}
            binds.update(outputTerms=b['partial0'],cfg=dict(count=count))
            k['fused'].dispatch(thread_count=[count,1,1],vars=binds,command_encoder=encoder)
        if pool is not None and kwargs.get('region')=='producer': encoder.write_timestamp(pool,kwargs['query_index_after'])
        for level, groups in enumerate(partial_counts[:-1]):
            k['partial'].dispatch(thread_count=[groups,1,1],vars=dict(inputTerms=b['partial'+str(level)],
                outputTerms=b['partial'+str(level+1)],params=dict(count=groups)),command_encoder=encoder)
        k['solve'].dispatch(thread_count=[1,1,1],vars=dict(inputTerms=b['partial'+str(len(partial_counts)-1)],
            delta=b['delta'],status=b['status'],solveParams=dict(damping=.01)),command_encoder=encoder)
        if pool is not None and kwargs.get('region')!='producer': encoder.write_timestamp(pool,kwargs['query_index_after'])
        device.submit_command_buffer(encoder.finish())
    outputs = ('out_0','out_1',*('partial'+str(i) for i in range(len(partial_counts))),'delta','status')
    def upload(variant):
        for name,array in arrays.items(): buffers[variant][name].copy_from_numpy(array.reshape(-1))
    def download(variant):
        return {name:buffers[variant][name].to_numpy().view(arrays[name].dtype).reshape(arrays[name].shape).copy() for name in outputs}
    def read(variant):
        dispatch(variant);device.wait();return download(variant)
    case = dict(point)
    case.update(device=device,kernels=k,buffers=buffers,dispatch=dispatch,upload=upload,download=download,read=read,arrays=arrays,
        outputs=outputs,sources=dict(baseline=point['sources']['baseline']+'\n'+gn,candidate=fused+'\n'+gn),
        family='point_plane_gn',profile=tree,partial_counts=partial_counts,
        dispatch_counts=dict(baseline=2+len(partial_counts),candidate=1+len(partial_counts)),
        uploaded_bytes=sum(a.nbytes for a in arrays.values()),readback_bytes=sum(arrays[name].nbytes for name in outputs))
    return case


def fusion_oracle(case):
    r,J = point_oracle(case)
    H,b = J.T@J,J.T@r
    return dict(residual=r,jacobian=J,hessian=H,gradient=b,cost=r@r,
        delta=np.linalg.solve(H+np.eye(6)*.01,-b))


def check_fusion_oracle(case, actual):
    check_point_oracle(case, actual)
    expected=fusion_oracle(case)
    final=actual['partial'+str(len(case['partial_counts'])-1)]
    target=np.r_[expected['hessian'].reshape(-1),expected['gradient'],expected['cost']]
    np.testing.assert_allclose(final,target,atol=2e-5,rtol=2e-5)
    np.testing.assert_allclose(actual['delta'][:6],expected['delta'],atol=2e-6,rtol=8e-5)
    np.testing.assert_array_equal(actual['delta'][6:],-903)
    np.testing.assert_array_equal(actual['status'],[0,999,999])
