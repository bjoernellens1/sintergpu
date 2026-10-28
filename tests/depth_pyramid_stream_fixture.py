# SPDX-License-Identifier: Apache-2.0
"""Test-only streaming performance probe setup; no device or submission ownership."""
import hashlib
from pathlib import Path
from benchmarks.depth_pyramid_gpu_fixture import ROOT, Fixture, preflight

STREAM='benchmarks/shaders/depth_pyramid_vkl_stream_candidate.slang'
STREAM_SHA='078393352ffb207755da14b4673c6e51d34ed7603ebc95c8598e251ba0e76db4'


def stream_preflight(root=ROOT):
    root=Path(root)
    identities=preflight(root)
    path=root/STREAM
    if path.is_symlink() or not path.is_file(): raise ValueError('missing or symlinked stream candidate')
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    if digest!=STREAM_SHA: raise ValueError('stream candidate digest mismatch')
    identities[STREAM]=digest
    return identities


def stream_source(root=ROOT):
    stream_preflight(root)
    return (Path(root)/STREAM).read_text()


def stream_fixture(device,spec):
    if spec['profile']!='vkl': raise ValueError('stream candidate only supports VKL')
    source=stream_source()
    f=Fixture(device,spec)
    module=device.load_module_from_source('depth_pyramid_stream_'+hashlib.sha256(source.encode()).hexdigest()[:16],source)
    f.kernel=device.create_compute_kernel(device.link_program([module],[module.entry_point(spec['entry'])]))
    return f
