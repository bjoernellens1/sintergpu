# SPDX-License-Identifier: Apache-2.0
"""Incumbent-only depth-pyramid differential fixture; no implicit GPU execution.

Construct with a caller-owned device after preflight. Recording only writes a
barrier and dispatch into a borrowed encoder. Upload, submission and readback
are explicit test boundaries, and are never part of the recorded hot path.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    'src/sintergpu/kernels/generated/primitives.generated.slang': '0cba624d734902ce6a788b5c37b5fe332a4e2d1b81a1fb2d149a1a101edb8f44',
    'references/donors/slang-rgbd-slam/src/slang_rgbd_slam/tracker.py': '00601a3d4c53c169b4780be2ca848e08474dd6ca523b51fb6e85936f14846ea8',
    'src/sintergpu/kernels/vkl/depth_pyramid.slang': '55fea213f1f1614ed3307a6c7a2a3e62765e20db86fbe7da9790c1379f4d4aa2',
    'src/sintergpu/kernels/slang_rgbd_slam/tracker.slang': 'd6575a6d6f73df2dbc98e23c3b935ce0eb9e697112c64dd85f096fad649b690a',
    'references/donors/vkl/src/vkl/kernels/reference/depth_pyramid.py': 'a1e7fdd46785607a9b52f98819469e5ebe7192558df2c574a583c553b33b6ed1',
    'references/donors/slang-rgbd-slam/src/slang_rgbd_slam/shaders/tracker.slang': 'd6575a6d6f73df2dbc98e23c3b935ce0eb9e697112c64dd85f096fad649b690a',
}
PADDING = 19
MAX_BYTES = 32 << 20
UINT_MAX = (1 << 32) - 1


def preflight(root=ROOT):
    identities = {}
    for name, expected in SOURCES.items():
        path = Path(root) / name
        if path.is_symlink() or not path.is_file():
            raise ValueError('missing or symlinked incumbent source: ' + name)
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError('incumbent source digest mismatch: ' + name)
        identities[name] = actual
    return identities


def composed_slam_source(root=ROOT):
    """Frozen generated primitives are byte-identical to donor primitives_source()."""
    preflight(root)
    root = Path(root)
    return ((root / 'src/sintergpu/kernels/generated/primitives.generated.slang').read_text() + '\n'
            + (root / 'src/sintergpu/kernels/slang_rgbd_slam/tracker.slang').read_text())


def _input(value, dtype, ndim, name):
    a = np.asarray(value)
    if a.dtype != dtype or a.ndim != ndim or any(n < 1 for n in a.shape):
        raise ValueError(name + ' requires nonempty typed input')
    if a.nbytes + 4 * PADDING > MAX_BYTES:
        raise ValueError('fixture buffer budget exceeded')
    return np.ascontiguousarray(a)


def specification(profile, depth, *, valid=None, confidence=None, threshold=0., max_groups=(65535, 65535)):
    """Validate CPU inputs and produce original entry ABI, live counts and launch."""
    if len(max_groups) != 2 or any(type(n) is not int or n < 1 for n in max_groups):
        raise ValueError('positive device dispatch limits required')
    if profile == 'vkl':
        d = _input(depth, np.float32, 3, 'depth')
        v = _input(valid, np.uint32, 3, 'valid')
        c = _input(confidence, np.float32, 3, 'confidence')
        if d.shape != v.shape or d.shape != c.shape:
            raise ValueError('input shape mismatch')
        b, h, w = d.shape
        oh, ow = (h + 1)//2, (w + 1)//2
        count = b * oh * ow
        if b*h*w > UINT_MAX or count > UINT_MAX or w == UINT_MAX or h == UINT_MAX:
            raise ValueError('uint indexing overflow')
        groups = ((count + 255)//256, 1)
        params = dict(batches=b, height=h, width=w, epsilon=float(np.float32(threshold)))
        inputs = dict(depth=d, valid=v, confidence=c)
        outputs = {'depthOut': np.float32, 'validOut': np.uint32, 'confidenceOut': np.float32}
        path, entry, parameter = 'src/sintergpu/kernels/vkl/depth_pyramid.slang', 'depthPyramidLevel', 'params'
        threads = (count, 1, 1)
        shape = (b, oh, ow)
    elif profile == 'slam':
        d = _input(depth, np.float32, 2, 'depth')
        if valid is not None or confidence is not None:
            raise ValueError('SLAM entry has no mask or confidence binding')
        h, w = d.shape
        if h % 2 or w % 2 or h*w > UINT_MAX:
            raise ValueError('SLAM requires even, uint-addressable input dimensions')
        oh, ow = h//2, w//2
        count = oh * ow
        groups = ((ow + 15)//16, (oh + 15)//16)
        params = dict(width=ow, height=oh, fx=1., fy=1., cx=0., cy=0., depthJump=float(np.float32(threshold)))
        inputs, outputs = {'inputDepth': d}, {'outputDepth': np.float32}
        path, entry, parameter = 'src/sintergpu/kernels/slang_rgbd_slam/tracker.slang', 'downsample', 'mapParams'
        threads, shape = (ow, oh, 1), (oh, ow)
    else:
        raise ValueError('unknown profile')
    if groups[0] > max_groups[0] or groups[1] > max_groups[1] or (count + PADDING)*4 > MAX_BYTES:
        raise ValueError('fixture dispatch or output budget exceeded')
    tracker_api_compatible = (profile == 'slam' and h >= 4 and w >= 4 and np.isfinite(d).all() and (d >= 0).all() and np.isfinite(np.float32(threshold)) and np.float32(threshold) > 0)
    return dict(tracker_api_compatible=bool(tracker_api_compatible), profile=profile, inputs=inputs, outputs=outputs, count=count, shape=shape, params=params,
                source=path, entry=entry, parameter=parameter, threads=threads, groups=groups)


class Fixture:
    """Own test buffers only; caller owns device, command encoder and submission."""
    def __init__(self, device, spec):
        import slangpy as spy
        self.device, self.spec = device, spec
        limits = device.info.limits
        gx, gy = int(limits.max_compute_dispatch_thread_groups.x), int(limits.max_compute_dispatch_thread_groups.y)
        if spec['groups'][0] > gx or spec['groups'][1] > gy:
            raise ValueError('dispatch exceeds device limit')
        if int(limits.max_compute_threads_per_group) < 256 or int(limits.max_compute_thread_group_size.x) < (256 if spec['profile'] == 'vkl' else 16) or (spec['profile'] == 'slam' and int(limits.max_compute_thread_group_size.y) < 16):
            raise ValueError('device cannot execute incumbent workgroup')
        usage = spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access
        self.initial = {}
        for name, value in spec['inputs'].items():
            self.initial[name] = np.full(value.size + PADDING, 73, dtype=value.dtype)
            self.initial[name][:value.size] = value.ravel()
        for name, dtype in spec['outputs'].items():
            self.initial[name] = np.full(spec['count'] + PADDING, 73, dtype=dtype)
        self.buffers = {name: device.create_buffer(element_count=a.size, struct_size=4, usage=usage, data=a) for name, a in self.initial.items()}
        if spec['profile'] == 'slam':
            source = composed_slam_source()
            module = device.load_module_from_source('depth_pyramid_incumbent_' + hashlib.sha256(source.encode()).hexdigest()[:16], source)
        else:
            module = device.load_module(str(ROOT / spec['source']))
        self.kernel = device.create_compute_kernel(device.link_program([module], [module.entry_point(spec['entry'])]))
        self.bindings = {**self.buffers, spec['parameter']: spec['params']}

    def upload(self, inputs):
        if set(inputs) != set(self.spec['inputs']):
            raise ValueError('complete input set required')
        for name, original in self.spec['inputs'].items():
            value = np.asarray(inputs[name])
            if value.shape != original.shape or value.dtype != original.dtype:
                raise ValueError('input shape or dtype changed')
        for name, value in inputs.items():
            self.initial[name][:value.size] = value.ravel()
            self.buffers[name].copy_from_numpy(self.initial[name])

    def record(self, encoder):
        if encoder is None:
            raise ValueError('borrowed command encoder required')
        encoder.global_barrier()
        self.kernel.dispatch(thread_count=self.spec['threads'], vars=self.bindings, command_encoder=encoder)

    def read_checked(self):
        result = {}
        for name, before in self.initial.items():
            actual = self.buffers[name].to_numpy().view(before.dtype).reshape(-1)
            if actual.shape != before.shape:
                raise AssertionError('buffer length changed: ' + name)
            live = before.size - PADDING
            np.testing.assert_array_equal(actual[live:].view(np.uint32), before[live:].view(np.uint32), err_msg=name + ' tail')
            if name in self.spec['inputs']:
                np.testing.assert_array_equal(actual.view(np.uint32), before.view(np.uint32), err_msg=name + ' input mutated')
            else:
                result[name] = actual[:live].reshape(self.spec['shape']).copy()
        return result
