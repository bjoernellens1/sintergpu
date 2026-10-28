# SPDX-License-Identifier: Apache-2.0
"""Candidate test setup. No device creation, submission, or synchronization."""
import hashlib
from pathlib import Path
from benchmarks.depth_pyramid_gpu_fixture import Fixture, ROOT, preflight

CANDIDATES = {
    'src/sintergpu/kernels/common/depth_pyramid_core.slang': 'd4fc879ba28812fc2f5c1e9bf2f6f1d9c382c15ec9ac3013ae2d024d77a4745a',
    'benchmarks/shaders/depth_pyramid_vkl_candidate.slang': '42f6b34a012093c17b953bbcfd5d2eb7104b3bdaa48b1beaf29a54516eb15bb3',
    'benchmarks/shaders/depth_pyramid_slam_candidate.slang': 'ff517b77a9539f917487c9c164604dfa7107bb62565e3fc34d4bcc5f82a695b9',
}


def candidate_preflight(root=ROOT):
    root=Path(root)
    identities=preflight(root)
    for name, expected in CANDIDATES.items():
        path=root/name
        if path.is_symlink() or not path.is_file():
            raise ValueError('missing or symlinked candidate source: '+name)
        digest=hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != expected:
            raise ValueError('candidate source digest mismatch: '+name)
        identities[name]=digest
    return identities


def candidate_source(profile, root=ROOT):
    if profile not in ('vkl','slam'):
        raise ValueError('unknown candidate profile')
    candidate_preflight(root)
    root=Path(root)
    core=(root/'src/sintergpu/kernels/common/depth_pyramid_core.slang').read_text()
    path=root/('benchmarks/shaders/depth_pyramid_'+profile+'_candidate.slang')
    source=path.read_text()
    include='#include "depth_pyramid_core.slang"'
    if source.count(include) != 1:
        raise ValueError('candidate must import exactly one core')
    source=source.replace(include,core)
    if profile == 'slam':
        source=(root/'src/sintergpu/kernels/generated/primitives.generated.slang').read_text()+'\n'+source
    return source


def candidate_fixture(device, spec):
    """Use identical original resource/dispatch setup, then link candidate code."""
    source=candidate_source(spec['profile'])
    fixture=Fixture(device,spec)
    name='depth_pyramid_candidate_'+spec['profile']+'_'+hashlib.sha256(source.encode()).hexdigest()[:16]
    module=device.load_module_from_source(name,source)
    fixture.kernel=device.create_compute_kernel(device.link_program([module],[module.entry_point(spec['entry'])]))
    return fixture
