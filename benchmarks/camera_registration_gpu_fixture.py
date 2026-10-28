# SPDX-License-Identifier: Apache-2.0
"""Fixture-to-graph mapping and explicit owner-side allocation for GPU tests.

No device is created here. Allocation metadata is captured at allocation time;
readback and submission are explicit test boundaries, never recorder behavior.
"""
import sys

import numpy as np

from sintergpu.camera_registration import (camera_rectify_grid_plan, camera_rectify_from_grid_plan,
    camera_rectify_plan, depth_camera_rays_plan, depth_register_plan)
from benchmarks.camera_registration_support import BufferDescriptor, PreparedCameraGraph

DTYPES = {'float32': np.dtype('float32'), 'uint32': np.dtype('uint32'), 'int64': np.dtype('int64')}


def fixture_graph(payload, *, max_workgroups_x):
    kw = payload['kwargs']; fn = payload['function']; debug = kw.get('debug', False)
    opts = dict(max_workgroups_x=max_workgroups_x)
    inputs = {}
    if fn == 'camera_rectify_grid':
        b = kw['source_intrinsics'].shape[0]; hs, ws = kw['source_hw']; ht, wt = kw['output_hw']
        brown = kw.get('source_distortion_model', 'pinhole') == 'brown_conrady'
        graph = camera_rectify_grid_plan(b, hs, ws, ht, wt, distortion=brown, debug=debug, **opts)
        inputs.update(source_intrinsics=kw['source_intrinsics'], target_intrinsics=kw['target_intrinsics'])
        if brown: inputs['source_distortion'] = kw['source_distortion_coefficients']
    elif fn in ('camera_rectify', 'camera_rectify_from_grid'):
        image = kw['image']; b, c, hs, ws = image.shape
        if fn == 'camera_rectify':
            ht, wt = kw['output_hw']; brown = kw.get('source_distortion_model', 'pinhole') == 'brown_conrady'
            graph = camera_rectify_plan(b,c,hs,ws,ht,wt,distortion=brown,debug=debug,sampling=kw.get('sampling','bilinear'),**opts)
            inputs.update(source_intrinsics=kw['source_intrinsics'],target_intrinsics=kw['target_intrinsics'])
            if brown: inputs['source_distortion'] = kw['source_distortion_coefficients']
        else:
            ht, wt = kw['source_uv'].shape[1:3]
            graph = camera_rectify_from_grid_plan(b,c,hs,ws,ht,wt,sampling=kw.get('sampling','bilinear'),**opts)
            inputs.update(source_uv=kw['source_uv'],valid=kw['valid'].astype(np.uint32))
        inputs['input_image'] = image
    elif fn == 'depth_camera_rays':
        b = kw['depth_intrinsics'].shape[0]; h, w = kw['depth_hw']
        brown = kw.get('depth_distortion_model', 'pinhole') == 'brown_conrady'
        graph = depth_camera_rays_plan(b,h,w,distortion=brown,**opts)
        inputs['depth_intrinsics'] = kw['depth_intrinsics']
        if brown: inputs['depth_distortion'] = kw['depth_distortion_coefficients']
    elif fn == 'depth_register':
        b, _, h, w = kw['depth'].shape; ht, wt = kw['output_hw']
        dbrown = kw.get('depth_distortion_model', 'pinhole') == 'brown_conrady'
        rbrown = kw.get('rgb_distortion_model', 'pinhole') == 'brown_conrady'
        cached = kw.get('depth_rays') is not None
        graph = depth_register_plan(b,h,w,ht,wt,depth_distortion=dbrown,rgb_distortion=rbrown,cached_rays=cached,
                                    min_depth=kw.get('min_depth',1e-6),max_depth=kw.get('max_depth'),debug=debug,**opts)
        inputs.update(input_depth=kw['depth'],depth_intrinsics=kw['depth_intrinsics'],rgb_intrinsics=kw['rgb_intrinsics'],transform_rgb_depth=kw['transform_rgb_depth'])
        if dbrown: inputs['depth_distortion'] = kw['depth_distortion_coefficients']
        if rbrown: inputs['rgb_distortion'] = kw['rgb_distortion_coefficients']
        if cached: inputs['depth_rays'] = kw['depth_rays']
    else:
        raise ValueError('unsupported fixture operation: ' + fn)
    for name, spec in graph['resources'].items():
        if spec['role'] == 'input' and name not in inputs:
            if name != 'dummy_float': raise ValueError('missing fixture input: ' + name)
            inputs[name] = np.zeros(spec['shape'], dtype=DTYPES[spec['dtype']])
    for name, array in inputs.items():
        spec = graph['resources'][name]
        if array.dtype != DTYPES[spec['dtype']] or array.shape != spec['shape']:
            raise ValueError('fixture resource dtype/shape mismatch: ' + name)
    return graph, inputs


class FixtureResources:
    def __init__(self, device, graph, inputs, *, enabled_features, padding=13):
        if sys.byteorder != 'little':
            raise RuntimeError('camera diagnostic int64 readback requires little-endian host')
        import slangpy as spy
        self.graph = graph; self.buffers = {}; self.initial = {}; descriptors = {}
        usage = spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access
        for name, spec in graph['resources'].items():
            dtype = DTYPES[spec['dtype']]
            data = np.full(spec['elements'] + padding, 73, dtype=dtype)
            if name in inputs: data[:spec['elements']] = inputs[name].reshape(-1)
            buffer = device.create_buffer(element_count=data.size, struct_size=spec['element_size'], usage=usage, data=data)
            # SlangPy Buffer.size is the actual byte capacity, not the requested
            # allocation size. Fail before binding if the allocator returned less.
            if type(buffer.size) is not int or buffer.size < data.nbytes:
                raise ValueError('allocated buffer is undersized: ' + name)
            self.buffers[name] = buffer
            if name in inputs: self.initial[name] = data.copy()
            # Each create_buffer owns a distinct whole allocation, created on this
            # device with exactly this allocation byte capacity and usage mask.
            descriptors[name] = BufferDescriptor(buffer, device, buffer, 0, buffer.size, frozenset({'shader_resource','unordered_access'}))
        self.prepared = PreparedCameraGraph(device, graph, self.buffers, descriptors=descriptors, enabled_features=enabled_features)

    def upload(self, inputs):
        if set(inputs) != set(self.initial): raise ValueError('reuse input sets differ')
        for name, value in inputs.items():
            spec = self.graph['resources'][name]
            if value.dtype != DTYPES[spec['dtype']] or value.shape != spec['shape']: raise ValueError('reuse input layout mismatch')
            data = self.initial[name]; data[:spec['elements']] = value.reshape(-1)
            self.buffers[name].copy_from_numpy(data)

    def read(self):
        return {name: buffer.to_numpy().view(DTYPES[self.graph['resources'][name]['dtype']]).reshape(-1) for name, buffer in self.buffers.items()}

    def outputs_and_guards(self, expected):
        states = self.read(); outputs = {}
        for name, spec in self.graph['resources'].items():
            np.testing.assert_array_equal(states[name][spec['elements']:], np.full(states[name].size-spec['elements'],73,dtype=DTYPES[spec['dtype']]),err_msg=name+' tail')
            if name in self.initial:
                np.testing.assert_array_equal(states[name].view(np.uint8),self.initial[name].view(np.uint8),err_msg=name+' input changed')
        for key, expected_array in expected.items():
            resource_key = 'depth_rays' if key == 'result' else key
            name = self.graph['outputs'][resource_key]; spec = self.graph['resources'][name]
            output = states[name][:spec['elements']].reshape(expected_array.shape)
            if expected_array.dtype == np.dtype('bool'):
                assert np.all((output == 0) | (output == 1)), key + ' mask is not canonical 0/1'
                outputs[key] = output.astype(bool)
            else:
                assert output.dtype == expected_array.dtype, key + ' dtype mismatch'
                outputs[key] = output
        return outputs
