"""Compile original and additive candidate entry points; GPU parity is separate."""
import json
import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / 'src/sintergpu/kernels/common'
VKL = ROOT / 'src/sintergpu/kernels/vkl/depth_pyramid.slang'
SLAM = ROOT / 'src/sintergpu/kernels/slang_rgbd_slam/tracker.slang'
PRIMITIVES = ROOT / 'src/sintergpu/kernels/generated/primitives.generated.slang'
CANDIDATES = ROOT / 'benchmarks/shaders'


@pytest.mark.parametrize('version', ['2026.12', '2026.19'])
@pytest.mark.parametrize('profile,entry,original,candidate', [
    ('vkl', 'depthPyramidLevel', VKL, CANDIDATES / 'depth_pyramid_vkl_candidate.slang'),
    ('slam', 'downsample', SLAM, CANDIDATES / 'depth_pyramid_slam_candidate.slang'),
])
def test_original_candidate_abi(version, profile, entry, original, candidate, tmp_path):
    assert candidate.is_file(), f'missing {profile} candidate shader'
    key = 'SINTERGPU_SLANGC_' + version.replace('.', '_')
    compiler = Path(os.environ.get(key, ROOT / '.tools' / ('slang-' + version) / 'bin/slangc'))
    if not compiler.is_file():
        if key in os.environ:
            pytest.fail(f'{key} compiler unavailable: {compiler}')
        pytest.skip(f'Slang {version} unavailable')
    ident = subprocess.run([str(compiler), '-version'], capture_output=True, text=True, check=True)
    assert (ident.stdout + ident.stderr).strip() == version
    results = []
    for label, source in [('original', original), ('candidate', candidate)]:
        text = source.read_text()
        if profile == 'slam':
            text = PRIMITIVES.read_text() + '\n' + text
        composed = tmp_path / (label + '.slang')
        composed.write_text(text)
        reflect = tmp_path / (label + '.json')
        proc = subprocess.run([str(compiler), str(composed), '-I', str(CORE), '-entry', entry,
            '-target', 'spirv', '-profile', 'glsl_460', '-o', str(tmp_path / (label+'.spv')),
            '-reflection-json', str(reflect)], capture_output=True, text=True)
        assert proc.returncode == 0, proc.stdout + proc.stderr
        data = json.loads(reflect.read_text())
        results.append((data['parameters'], data['entryPoints'][0]))
    assert results[0] == results[1]
    assert not any('policy' in p['name'].lower() for p in results[1][0])


def test_candidates_call_only_their_static_policy():
    vkl = (CANDIDATES / 'depth_pyramid_vkl_candidate.slang').read_text()
    slam = (CANDIDATES / 'depth_pyramid_slam_candidate.slang').read_text()
    assert 'depthPyramidVklBlock(d,v,c,params.epsilon)' in vkl
    assert 'depthPyramidSlamBlock(z,mapParams.depthJump)' in slam
    assert 'depthPyramidSlamBlock' not in vkl
    assert 'depthPyramidVklBlock' not in slam
    # All remaining tracker entry points remain byte-identical to the donor.
    marker = '[shader("compute")]\n[numthreads(16,16,1)]\nvoid makeVertices'
    assert slam[slam.index(marker):] == SLAM.read_text()[SLAM.read_text().index(marker):]
