import numpy as np
import pytest
from sintergpu.references import camera_registration_numeric as c


def camera(batch=1):
    return np.broadcast_to(np.eye(3, dtype=np.float32), (batch, 3, 3)).copy()


def test_identity_grid_and_sampling():
    k = camera(8)
    image = np.arange(8 * 3 * 2 * 3, dtype=np.float32).reshape(8, 3, 2, 3)
    grid = c.camera_rectify_grid(k, k, (2, 3), (2, 3), debug=True)
    assert set(grid) == {'source_uv', 'valid', 'target_ray', 'displacement'}
    np.testing.assert_array_equal(grid['source_uv'][0], [[[0, 0], [1, 0], [2, 0]], [[0, 1], [1, 1], [2, 1]]])
    np.testing.assert_array_equal(grid['displacement'], 0)
    out = c.camera_rectify(image, k, k, (2, 3), debug=True)
    np.testing.assert_array_equal(out['image'], image)
    np.testing.assert_array_equal(out['image'], c.camera_rectify_from_grid(image, grid['source_uv'], grid['valid'])['image'])


def test_cached_nonfinite_and_nearest_ties():
    image = np.array([[[[2, 4, 8, 16]]]], dtype=np.float32)
    uv = np.array([[[[np.nan, np.inf], [-.5, 0], [.5, 0], [1.5, 0], [3.5, 0]]]], dtype=np.float32)
    valid = np.array([[[[True, True, True, True, False]]]])
    out = c.camera_rectify_from_grid(image, uv, valid, sampling='nearest')
    np.testing.assert_array_equal(out['image'], [[[[2, 2, 4, 8, 0]]]])


def test_registration_collisions_and_invalid_diagnostics():
    k = camera()
    rays = np.zeros((1, 1, 4, 2), dtype=np.float32)
    depth = np.array([[[[2, 1, 1, np.nan]]]], dtype=np.float32)
    transform = np.eye(4, dtype=np.float32)[None]
    out = c.depth_register(depth, k, k, transform, (1, 1), depth_rays=rays, debug=True)
    assert set(out) == {'depth', 'valid', 'winner_source_index', 'collision_count', 'source_uv', 'projected_z', 'source_valid'}
    assert out['depth'].item() == 1 and out['winner_source_index'].item() == 1
    assert out['collision_count'].item() == 3
    np.testing.assert_array_equal(out['source_valid'], [[True, True, True, False]])
    assert out['winner_source_index'].dtype == np.int64


def test_brown_inverse_five_steps_and_forward():
    k = camera()
    coeff = np.array([[.1, -.01, .02, -.03, .005]], dtype=np.float32)
    rays = c.depth_camera_rays(k, (1, 2), depth_distortion_model='brown_conrady', depth_distortion_coefficients=coeff)
    x, y = np.float32(1), np.float32(0)
    for _ in range(5):
        r = x*x+y*y
        radial = 1 + coeff[0,0]*r + coeff[0,1]*r*r + coeff[0,4]*r*r*r
        px = x*radial + 2*coeff[0,2]*x*y + coeff[0,3]*(r+2*x*x)
        py = y*radial + coeff[0,2]*(r+2*y*y) + 2*coeff[0,3]*x*y
        x, y = np.float32(x + 1 - px), np.float32(y - py)
    np.testing.assert_allclose(rays[0,0,1], [x,y], rtol=1e-5, atol=1e-6)


def _donor():
    import importlib.util
    from pathlib import Path
    pytest.importorskip('torch')
    path = Path(__file__).resolve().parents[1] / 'references/donors/vkl/src/vkl/kernels/reference/camera_registration.py'
    spec = importlib.util.spec_from_file_location('frozen_vkl_camera_registration', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _compare(actual, expected):
    if not isinstance(actual, dict):
        actual, expected = {'result': actual}, {'result': expected}
    assert actual.keys() == expected.keys()
    for key, value in actual.items():
        ref = expected[key].detach().cpu().numpy()
        assert value.dtype == ref.dtype and value.shape == ref.shape
        if value.dtype.kind == 'f':
            np.testing.assert_allclose(value, ref, rtol=8e-6, atol=8e-7, equal_nan=True, err_msg=key)
        else:
            np.testing.assert_array_equal(value, ref, err_msg=key)


@pytest.mark.parametrize('shape,channels', [((1,1),1), ((3,5),3), ((4,7),4)])
@pytest.mark.parametrize('sampling', ['nearest','bilinear'])
@pytest.mark.parametrize('brown', [False, True])
def test_frozen_donor_differential(shape, channels, sampling, brown):
    torch = pytest.importorskip('torch')
    donor = _donor()
    rng = np.random.default_rng(19)
    b = 2
    k = camera(b); k[:,0,0] = [1.3,2.2]; k[:,1,1] = [1.7,2.8]; k[:,0,2] = [.2,.7]; k[:,1,2] = [.3,.8]
    target = camera(b); target[:,0,0] = [1.8,2.6]; target[:,1,1] = [1.2,2.1]
    coeff = np.tile(np.array([.02,-.003,.004,-.005,.0001], dtype=np.float32), (b,1))
    options = {'source_distortion_model':'brown_conrady', 'source_distortion_coefficients':coeff} if brown else {}
    to_torch = lambda d: {key: torch.from_numpy(value) if isinstance(value,np.ndarray) else value for key,value in d.items()}
    image = rng.random((b,channels,*shape), dtype=np.float32)
    grid = c.camera_rectify_grid(k,target,shape,(3,4),debug=True,**options)
    tg = donor.camera_rectify_grid(torch.from_numpy(k),torch.from_numpy(target),shape,(3,4),debug=True,**to_torch(options))
    _compare(grid,tg)
    _compare(c.camera_rectify(image,k,target,(3,4),sampling=sampling,debug=True,**options), donor.camera_rectify(torch.from_numpy(image),torch.from_numpy(k),torch.from_numpy(target),(3,4),sampling=sampling,debug=True,**to_torch(options)))
    ray_opts = {'depth_distortion_model':'brown_conrady','depth_distortion_coefficients':coeff} if brown else {}
    rays = c.depth_camera_rays(k,shape,**ray_opts)
    _compare(rays, donor.depth_camera_rays(torch.from_numpy(k),shape,**to_torch(ray_opts)))
    depth = rng.random((b,1,*shape),dtype=np.float32) + np.float32(.1)
    depth.flat[0] = np.nan
    transform = np.tile(np.eye(4,dtype=np.float32), (b,1,1)); transform[:,0,3] = .05
    regopts = {'depth_distortion_model':'brown_conrady','depth_distortion_coefficients':coeff,'rgb_distortion_model':'brown_conrady','rgb_distortion_coefficients':coeff} if brown else {}
    _compare(c.depth_register(depth,k,target,transform,(3,4),debug=True,**regopts), donor.depth_register(torch.from_numpy(depth),torch.from_numpy(k),torch.from_numpy(target),torch.from_numpy(transform),(3,4),debug=True,**to_torch(regopts)))


def test_depth_interval_epsilon_and_raw_invalid_projection():
    k = camera()
    eps = np.finfo(np.float32).eps
    depth = np.array([[[[1, 2, 3, np.nan]]]], dtype=np.float32)
    rays = np.zeros((1,1,4,2), dtype=np.float32)
    transform = np.eye(4,dtype=np.float32)[None]; transform[0,2,3] = eps
    out = c.depth_register(depth,k,k,transform,(1,1),depth_rays=rays,min_depth=1,max_depth=2,debug=True)
    np.testing.assert_array_equal(out['source_valid'], [[True,True,False,False]])
    np.testing.assert_array_equal(out['projected_z'][0,2:], [eps,eps])
    assert out['collision_count'].item() == 2 and out['winner_source_index'].item() == 0


def test_border_tolerance_and_source_diagnostics():
    k = camera(); target = camera()
    tol = np.float32(16*np.finfo(np.float32).eps*4)
    for offset, expected in [(-tol, True), (-tol*2, False), (np.float32(3)+tol, True), (np.float32(3)+tol*2, False)]:
        k[0,0,2] = offset
        out = c.camera_rectify_grid(k,target,(4,4),(1,1),debug=True)
        assert bool(out['valid'].item()) == expected
        assert out['source_uv'][0,0,0,0] == offset


@pytest.mark.parametrize('sampling', ['nearest', 'bilinear'])
def test_frozen_cached_grid_nonfinite_and_borders(sampling):
    torch = pytest.importorskip('torch')
    donor = _donor()
    image = np.array([[[[1,2,3,4],[5,6,7,8]]]], dtype=np.float32)
    uv = np.array([[[[np.nan,np.inf],[-.5,0],[.5,0],[1.5,0],[3.5,1],[2.25,.5],[-np.inf,np.nan]]]], dtype=np.float32)
    valid = np.array([[[[True,True,True,True,True,True,False]]]])
    _compare(c.camera_rectify_from_grid(image,uv,valid,sampling=sampling),donor.camera_rectify_from_grid(torch.from_numpy(image),torch.from_numpy(uv),torch.from_numpy(valid),sampling=sampling))


def test_exact_and_one_ulp_winner_with_cached_rays():
    torch = pytest.importorskip('torch')
    donor = _donor()
    k = camera(); transform = np.eye(4,dtype=np.float32)[None]
    depth = np.array([[[[1, np.nextafter(np.float32(1), np.float32(0)), 1, np.nan]]]], dtype=np.float32)
    rays = np.zeros((1,1,4,2),dtype=np.float32)
    out = c.depth_register(depth,k,k,transform,(1,1),depth_rays=rays,debug=True)
    assert out['winner_source_index'].item() == 1 and out['collision_count'].item() == 3
    _compare(out, donor.depth_register(torch.from_numpy(depth),torch.from_numpy(k),torch.from_numpy(k),torch.from_numpy(transform),(1,1),depth_rays=torch.from_numpy(rays),debug=True))


@pytest.mark.parametrize('zshift', [-np.finfo(np.float32).eps, 0, np.finfo(np.float32).eps])
def test_projected_z_epsilon_threshold(zshift):
    torch = pytest.importorskip('torch')
    donor = _donor()
    k=camera(); transform=np.eye(4,dtype=np.float32)[None]
    transform[0,2,3] = np.float32(-1 + zshift)
    depth=np.array([[[[1, np.nan]]]],dtype=np.float32)
    args=(depth,k,k,transform,(1,2))
    actual=c.depth_register(*args,debug=True)
    expected=donor.depth_register(*(torch.from_numpy(x) if isinstance(x,np.ndarray) else x for x in args),debug=True)
    _compare(actual,expected)


def test_bilinear_cancellation_matches_eager_donor():
    torch = pytest.importorskip('torch')
    image=np.array([[[[-7561.8125,-6079.17919921875],[-1044.6041259765625,54042.46875]]]],dtype=np.float32)
    uv=np.array([[[[0.3012532889842987,0.3139173984527588]]]],dtype=np.float32)
    valid=np.ones((1,1,1,1),dtype=bool)
    actual=c.camera_rectify_from_grid(image,uv,valid)['image']
    expected=_donor().camera_rectify_from_grid(torch.from_numpy(image),torch.from_numpy(uv),torch.from_numpy(valid))['image'].numpy()
    assert expected.item() == np.float32(2.9704300686717033e-05)
    np.testing.assert_allclose(actual, expected, rtol=8e-6, atol=8e-7)


def test_bilinear_seeded_cancellation_against_eager_donor():
    torch = pytest.importorskip('torch')
    rng=np.random.default_rng(6743); n=512
    uv=rng.uniform(.1,.9,(n,1,1,2)).astype(np.float32)
    image=rng.uniform(-10000,10000,(n,1,2,2)).astype(np.float32)
    u=uv[:,0,0,0].astype(float); v=uv[:,0,0,1].astype(float)
    image[:,0,1,1]=-((image[:,0,0,0]*(1-u)*(1-v)+image[:,0,0,1]*u*(1-v)+image[:,0,1,0]*(1-u)*v)/(u*v)).astype(np.float32)
    valid=np.ones((n,1,1,1),dtype=bool)
    _compare(c.camera_rectify_from_grid(image,uv,valid),_donor().camera_rectify_from_grid(torch.from_numpy(image),torch.from_numpy(uv),torch.from_numpy(valid)))


def test_bilinear_fused_rounding_literal_without_torch():
    image=np.array([[[[-7561.8125,-6079.17919921875],[-1044.6041259765625,54042.46875]]]],dtype=np.float32)
    uv=np.array([[[[0.3012532889842987,0.3139173984527588]]]],dtype=np.float32)
    valid=np.ones((1,1,1,1),dtype=bool)
    assert c.camera_rectify_from_grid(image,uv,valid)['image'].item() == np.float32(2.9704300686717033e-05)


def test_portable_fma_matches_libm_random_and_special_values():
    native = c._load_fmaf()
    if native is None:
        pytest.skip('native fmaf unavailable for differential check')
    rng=np.random.default_rng(5481)
    raw=rng.integers(0,2**32,size=(30000,3),dtype=np.uint32)
    special=np.array([0,0x80000000,1,0x80000001,0x007fffff,0x00800000,0x3f800000,0xbf800000,0x7f7fffff,0x7f800000,0xff800000,0x7fc00001],dtype=np.uint32)
    triples=np.vstack((raw,np.array([(a,b,d) for a in special for b in special for d in special],dtype=np.uint32)))
    for triple in triples:
        a,b,d=triple.view(np.float32)
        expected=np.float32(native(a,b,d)); actual=c._fma_fallback(a,b,d)
        if np.isnan(expected):
            assert np.isnan(actual)
        else:
            assert actual.view(np.uint32) == expected.view(np.uint32), (triple,actual,expected)


def test_missing_libm_and_missing_symbol_fall_back(monkeypatch):
    def unavailable(*args):
        raise OSError('library unavailable')
    monkeypatch.setattr(c.ctypes,'CDLL',unavailable)
    assert c._load_fmaf() is None
    monkeypatch.setattr(c.ctypes,'CDLL',lambda *args: object())
    assert c._load_fmaf() is None
    monkeypatch.setattr(c,'_fmaf',c._fma_fallback)
    test_bilinear_fused_rounding_literal_without_torch()


def test_seeded_cancellation_portable_fallback(monkeypatch):
    monkeypatch.setattr(c, '_fmaf', c._fma_fallback)
    test_bilinear_seeded_cancellation_against_eager_donor()
