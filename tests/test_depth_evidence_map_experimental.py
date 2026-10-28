"""CPU donor semantics and compiler checks; no device execution claim."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / 'src/sintergpu/kernels/experimental/depth_evidence_map.slang'
DONOR = ROOT / 'references/donors/vkl/src/vkl/kernels/reference/depth_evidence.py'


def test_frozen_donor_identity():
    assert hashlib.sha256(DONOR.read_bytes()).hexdigest() == '5976e426e33d1e1430cc5c233bd351fe7e226f2289c1cd43966a34392957fe18'


def _map(depth, valid, confidence, y, x, near, far):
    h, w = depth.shape
    z = depth[y, x]
    usable = bool(valid[y, x] and np.isfinite(z))
    d = z if usable else np.float32(0)
    diffs = []
    for yy, xx in ((y, x+1), (y+1, x)):
        if yy < h and xx < w and usable and valid[yy, xx] and np.isfinite(depth[yy, xx]):
            diffs.append(np.float32(depth[yy, xx] - d))
    edge = any(abs(v) > max(abs(d), np.float32(1e-6)) * np.float32(.25) for v in diffs)
    conf = confidence[y, x] if usable else np.float32(0)
    return np.array([usable, usable and d < near, usable and d > far, d, d*d,
                     edge, conf, usable and not edge and conf > 0,
                     len(diffs), sum(float(v*v) for v in diffs)], dtype=np.float32)


@pytest.mark.parametrize('shape,grid', [((1, 1), (8, 8)), ((3, 5), (2, 3)), ((4, 4), (4, 4))])
@pytest.mark.parametrize('use_confidence', [False, True])
def test_pixel_fields_aggregate_to_frozen_donor(shape, grid, use_confidence):
    torch = pytest.importorskip('torch')
    rng = np.random.default_rng(92011)
    depth = rng.uniform(-.5, 10, shape).astype(np.float32)
    valid = rng.random(shape) > .25
    confidence = rng.uniform(0, 1, shape).astype(np.float32) if use_confidence else np.ones(shape, np.float32)
    if depth.size > 2:
        depth.flat[0] = np.nan
        depth.flat[1] = np.inf
        if use_confidence:
            confidence.flat[2] = 0
    spec = importlib.util.spec_from_file_location('frozen_depth_evidence', DONOR)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    t = lambda a: torch.from_numpy(a)[None, None]
    donor = module.depth_evidence(t(depth), t(valid), t(confidence) if use_confidence else None, grid_size=grid, near=.1, far=8.)
    fields = np.array([[_map(depth, valid, confidence, y, x, .1, 8.) for x in range(shape[1])] for y in range(shape[0])])
    totals = fields.sum(axis=(0, 1), dtype=np.float32)
    n = max(totals[0], 1)
    expected = {'valid_fraction': totals[0]/depth.size, 'hole_fraction': 1-totals[0]/depth.size,
                'near_fraction': totals[1]/n, 'far_fraction': totals[2]/n,
                'edge_fraction': totals[5]/n, 'confidence_mean': totals[6]/n,
                'stable_fraction': totals[7]/n, 'local_variance': totals[9]/(4*max(totals[8], 1))}
    for name, value in expected.items():
        np.testing.assert_allclose(donor[name].numpy(), [value], rtol=2e-6, atol=1e-6)
    gh, gw = min(grid[0], shape[0]), min(grid[1], shape[1])
    result = np.zeros((gh, gw, 8), np.float32)
    for y in range(shape[0]):
        for x in range(shape[1]):
            result[y*gh//shape[0], x*gw//shape[1]] += fields[y, x, :8]
    for row in range(gh):
        for col in range(gw):
            area = sum(1 for y in range(shape[0]) for x in range(shape[1]) if y*gh//shape[0] == row and x*gw//shape[1] == col)
            support = result[row, col, 0]
            result[row, col] /= max(support, 1)
            result[row, col, 0] = support/area
    np.testing.assert_allclose(donor['grid_evidence'].numpy()[0], result, rtol=2e-6, atol=1e-6)


@pytest.mark.parametrize('version', ['2026.12', '2026.19'])
def test_helper_compiles(version, tmp_path):
    key = 'SINTERGPU_SLANGC_' + version.replace('.', '_')
    compiler = Path(os.environ.get(key, ROOT / '.tools' / ('slang-' + version) / 'bin/slangc'))
    if not compiler.is_file():
        if key in os.environ:
            pytest.fail(f'{key} compiler missing: {compiler}')
        pytest.skip(f'Slang {version} unavailable')
    identity = subprocess.run([str(compiler), '-version'], capture_output=True, text=True, check=True)
    assert (identity.stdout + identity.stderr).strip() == version
    wrapper = tmp_path / 'test.slang'
    wrapper.write_text('''import experimental.depth_evidence_map;\nStructuredBuffer<float> d; RWStructuredBuffer<float> o;\n[shader("compute")][numthreads(1,1,1)] void main(uint3 id : SV_DispatchThreadID) { DepthEvidenceMap r = depthEvidenceMap(d[0], true, d[1], d[2], true, d[3], true, 0.1, 8.0); o[0]=r.support+r.nearCount+r.farCount+r.depth+r.depthSquared+r.edge+r.confidence+r.stable+r.pairCount+r.pairSquaredDifference; }\n''')
    proc = subprocess.run([str(compiler), str(wrapper), '-I', str(ROOT / 'src/sintergpu/kernels'), '-entry', 'main', '-target', 'spirv', '-profile', 'glsl_460', '-o', str(tmp_path / 'a.spv'), '-reflection-json', str(tmp_path / 'reflection.json')], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert json.loads((tmp_path / 'reflection.json').read_text())['entryPoints'][0]['threadGroupSize'] == [1, 1, 1]
