# SPDX-License-Identifier: Apache-2.0
"""Setup-only VKL camera resource graphs; all buffers and execution are caller-owned.

Calibration content and cache lifetime are the caller's responsibility. Plans describe
storage and dependencies only; they do not allocate, compile, record or synchronize.
"""
import math
import numbers
import struct

from .vkl import _integer, UINT_MAX


def _flag(name, value):
    if type(value) is not bool:
        raise ValueError(name + ' must be boolean')


def _size(name, value):
    _integer(name, value, 1, UINT_MAX)


def _depth(value, name):
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError(name + ' must be finite positive float32')
    try:
        result = struct.unpack('f', struct.pack('f', float(value)))[0]
    except (OverflowError, struct.error):
        raise ValueError(name + ' must be finite positive float32') from None
    if not math.isfinite(result) or result <= 0:
        raise ValueError(name + ' must be finite positive float32')
    return result


def _sampling(value):
    if value not in ('nearest', 'bilinear') or not isinstance(value, str):
        raise ValueError('sampling must be nearest or bilinear')


def _checked_product(name, *values):
    product = math.prod(values)
    if product > UINT_MAX:
        raise ValueError(name + ' exceeds uint32 indexing')
    return product


class _Graph:
    def __init__(self, limit):
        _size('max_workgroups_x', limit)
        self.limit = limit
        self.resources = {}
        self.stages = []

    def resource(self, name, shape, dtype='float32', role='scratch'):
        size = 8 if dtype == 'int64' else 4
        count = _checked_product(name, *shape)
        # Explicit byte capacities use the Vulkan uint64 size domain; the caller
        # must additionally validate its actual device buffer/allocation limits.
        byte_size = count * size
        if byte_size > 2**64 - 1:
            raise ValueError(name + ' byte size exceeds uint64')
        descriptor = dict(shape=tuple(shape), elements=count, dtype=dtype, element_size=size, bytes=byte_size, role=role)
        if name in self.resources and self.resources[name] != descriptor:
            raise ValueError('incompatible resource reuse: ' + name)
        self.resources[name] = descriptor
        return name

    def stage(self, entry, count, params, bindings, barriers=()):
        padded = ((count + 255) // 256) * 256
        if padded > UINT_MAX or padded // 256 > self.limit:
            raise ValueError(entry + ' exceeds padded uint32 or device X dispatch limit')
        parameter_binding = {'rectifyGrid': 'gridParams', 'rectifyFromGrid': 'sampleParams',
                             'depthCameraRays': 'raysParams'}.get(entry, 'registrationParams')
        used = {'registrationMin': {'projectedZOut', 'sourceValidOut', 'destinationIndexOut', 'minDepthBits'},
                'registrationWinnerCount': {'projectedZOut', 'sourceValidOut', 'destinationIndexOut', 'minDepthBits', 'winnerIndex', 'collisionCount'},
                'finalizeRegistration': {'minDepthBits', 'depthOut', 'validOut'}}
        if entry in used:
            bindings = {key: value for key, value in bindings.items() if key in used[entry]}
        self.stages.append(dict(operation='vkl.camera_registration.' + entry, entrypoint=entry,
                                thread_count=(count, 1, 1), params=params.copy(), params_binding=parameter_binding,
                                bindings=bindings.copy(), barriers=list(dict.fromkeys(barriers))))

    def finish(self, outputs, profile, features=()):
        for name in outputs.values():
            if self.resources[name]['role'] != 'input':
                self.resources[name]['role'] = 'output'
        return dict(resources=self.resources, stages=self.stages, outputs=outputs,
                    profile=profile, features=list(features))


def _dimensions(**values):
    for name, value in values.items():
        _size(name, value)
        if name.endswith(('height', 'width')) and value > 2**24:
            raise ValueError(name + ' exceeds exactly representable float32 pixel coordinates')


def _profile(kind, dimensions, **options):
    if kind == 'rectify_grid':
        cache_kind, shape, distortion = 'rectify_grid', dimensions, options['distortion']
    elif kind == 'rectify':
        cache_kind, shape, distortion = 'rectify_grid', (dimensions[0], *dimensions[2:]), options['distortion']
    elif kind == 'depth_camera_rays':
        cache_kind, shape, distortion = 'depth_camera_rays', dimensions, options['distortion']
    elif kind == 'depth_register':
        cache_kind, shape, distortion = 'depth_camera_rays', dimensions[:3], options['depth_distortion']
    else:
        # A caller-provided grid has no known calibration model at this boundary.
        cache_kind, shape, distortion = 'external_grid', (dimensions[0], *dimensions[2:]), None
    return dict(operation=kind, numerical='vkl-eager-float32', **options,
                cache=dict(kind=cache_kind, dimensions=tuple(shape), distortion=distortion,
                           dtype='float32', caller_calibration_identity_required=True,
                           caller_shader_identity_required=True))


def cache_identity(plan, calibration_identity, shader_identity):
    """Return a cache key using immutable caller-managed calibration/source identities.

    Caller must change calibration identity for any calibration input or model change.
    No contents are fetched from a device. Comparing this key rejects structural,
    sampling/profile and calibration changes before a cached buffer is reused.
    """
    if not isinstance(calibration_identity, str) or not calibration_identity:
        raise ValueError('calibration identity must be a nonempty immutable string')
    if not isinstance(shader_identity, str) or not shader_identity:
        raise ValueError('shader identity must be a nonempty immutable string')
    p = plan['profile']
    cache = p['cache']
    return (cache['kind'], p['numerical'], cache['dimensions'], cache['dtype'],
            cache['distortion'], calibration_identity, shader_identity)


def validate_cache(plan, identity, calibration_identity, shader_identity):
    if identity != cache_identity(plan, calibration_identity, shader_identity):
        raise ValueError('cached calibration identity or structure mismatch')


def _grid(g, b, hs, ws, ht, wt, distortion, debug):
    params = dict(batches=b, sourceHeight=hs, sourceWidth=ws, targetHeight=ht,
                  targetWidth=wt, distortion=int(distortion), debug=int(debug))
    source_k = g.resource('source_intrinsics', (b, 3, 3), role='input')
    target_k = g.resource('target_intrinsics', (b, 3, 3), role='input')
    coeff = g.resource('source_distortion', (b, 5), role='input') if distortion else g.resource('dummy_float', (1,), role='input')
    uv = g.resource('source_uv', (b, ht, wt, 2))
    valid = g.resource('valid', (b, 1, ht, wt), 'uint32')
    ray = g.resource('target_ray', (b, ht, wt, 3)) if debug else g.resource('dummy_output_float', (1,))
    displacement = g.resource('displacement', (b, 1, ht, wt)) if debug else 'dummy_output_float'
    bindings = dict(sourceIntrinsics=source_k, targetIntrinsics=target_k, sourceDistortion=coeff,
                    sourceUVOut=uv, validOut=valid, targetRayOut=ray, displacementOut=displacement)
    g.stage('rectifyGrid', b * ht * wt, params, bindings, (uv, valid, ray, displacement))
    outputs = dict(source_uv=uv, valid=valid)
    if debug:
        outputs.update(target_ray=ray, displacement=displacement)
    return outputs


def _sample(g, b, c, hs, ws, ht, wt, sampling, grid=None):
    image = g.resource('input_image', (b, c, hs, ws), role='input')
    if grid is None:
        uv = g.resource('source_uv', (b, ht, wt, 2), role='input')
        valid = g.resource('valid', (b, 1, ht, wt), 'uint32', role='input')
    else:
        uv, valid = grid['source_uv'], grid['valid']
    output = g.resource('image', (b, c, ht, wt))
    params = dict(batches=b, channels=c, sourceHeight=hs, sourceWidth=ws,
                  targetHeight=ht, targetWidth=wt, sampling=0 if sampling == 'nearest' else 1)
    g.stage('rectifyFromGrid', b * c * ht * wt, params,
            dict(image=image, sourceUV=uv, valid=valid, imageOut=output), (uv, valid, output))
    return dict(image=output, valid=valid)


def camera_rectify_grid_plan(batches, source_height, source_width, target_height, target_width, *, max_workgroups_x, distortion=False, debug=False):
    _dimensions(batches=batches, source_height=source_height, source_width=source_width, target_height=target_height, target_width=target_width)
    _flag('distortion', distortion); _flag('debug', debug)
    g = _Graph(max_workgroups_x)
    out = _grid(g, batches, source_height, source_width, target_height, target_width, distortion, debug)
    return g.finish(out, _profile('rectify_grid', (batches, source_height, source_width, target_height, target_width), distortion=distortion, debug=debug))


def camera_rectify_from_grid_plan(batches, channels, source_height, source_width, target_height, target_width, *, max_workgroups_x, sampling='bilinear'):
    _dimensions(batches=batches, channels=channels, source_height=source_height, source_width=source_width, target_height=target_height, target_width=target_width)
    _sampling(sampling)
    g = _Graph(max_workgroups_x)
    out = _sample(g, batches, channels, source_height, source_width, target_height, target_width, sampling)
    return g.finish(out, _profile('rectify_from_grid', (batches, channels, source_height, source_width, target_height, target_width), sampling=sampling))


def camera_rectify_plan(batches, channels, source_height, source_width, target_height, target_width, *, max_workgroups_x, distortion=False, debug=False, sampling='bilinear'):
    _dimensions(batches=batches, channels=channels, source_height=source_height, source_width=source_width, target_height=target_height, target_width=target_width)
    _flag('distortion', distortion); _flag('debug', debug); _sampling(sampling)
    g = _Graph(max_workgroups_x)
    grid = _grid(g, batches, source_height, source_width, target_height, target_width, distortion, debug)
    out = _sample(g, batches, channels, source_height, source_width, target_height, target_width, sampling, grid)
    if debug:
        out.update({k: grid[k] for k in ('source_uv', 'target_ray', 'displacement')})
    return g.finish(out, _profile('rectify', (batches, channels, source_height, source_width, target_height, target_width), distortion=distortion, debug=debug, sampling=sampling))


def _rays(g, b, h, w, distortion):
    intrinsics = g.resource('depth_intrinsics', (b, 3, 3), role='input')
    coeff = g.resource('depth_distortion', (b, 5), role='input') if distortion else g.resource('dummy_float', (1,), role='input')
    rays = g.resource('depth_rays', (b, h, w, 2))
    g.stage('depthCameraRays', b * h * w, dict(batches=b, depthHeight=h, depthWidth=w, depthDistortion=int(distortion)),
            dict(depthIntrinsics=intrinsics, depthDistortion=coeff, depthRaysOut=rays), (rays,))
    return rays


def depth_camera_rays_plan(batches, depth_height, depth_width, *, max_workgroups_x, distortion=False):
    _dimensions(batches=batches, depth_height=depth_height, depth_width=depth_width)
    _flag('distortion', distortion)
    g = _Graph(max_workgroups_x)
    rays = _rays(g, batches, depth_height, depth_width, distortion)
    return g.finish(dict(depth_rays=rays), _profile('depth_camera_rays', (batches, depth_height, depth_width), distortion=distortion))


def depth_register_plan(batches, depth_height, depth_width, target_height, target_width, *, max_workgroups_x, depth_distortion=False, rgb_distortion=False, cached_rays=False, min_depth=1e-6, max_depth=None, debug=False):
    _dimensions(batches=batches, depth_height=depth_height, depth_width=depth_width, target_height=target_height, target_width=target_width)
    for name, value in [('depth_distortion', depth_distortion), ('rgb_distortion', rgb_distortion), ('cached_rays', cached_rays), ('debug', debug)]:
        _flag(name, value)
    minimum = _depth(min_depth, 'min_depth')
    maximum = None if max_depth is None else _depth(max_depth, 'max_depth')
    if maximum is not None and maximum <= minimum:
        raise ValueError('depth interval must be strictly ordered in float32')
    points = _checked_product('source pixels', depth_height, depth_width)
    if points == UINT_MAX:
        raise ValueError('source pixel count reserves uint32 winner sentinel')
    g = _Graph(max_workgroups_x)
    b, h, w, th, tw = batches, depth_height, depth_width, target_height, target_width
    depth = g.resource('input_depth', (b, 1, h, w), role='input')
    # Intrinsics remain part of the caller contract even when rays are cached.
    if cached_rays:
        g.resource('depth_intrinsics', (b, 3, 3), role='input')
        if depth_distortion:
            g.resource('depth_distortion', (b, 5), role='input')
        rays = g.resource('depth_rays', (b, h, w, 2), role='input')
    else:
        rays = _rays(g, b, h, w, depth_distortion)
    rgb_k = g.resource('rgb_intrinsics', (b, 3, 3), role='input')
    transform = g.resource('transform_rgb_depth', (b, 4, 4), role='input')
    coeff = g.resource('rgb_distortion', (b, 5), role='input') if rgb_distortion else g.resource('dummy_float', (1,), role='input')
    uv = g.resource('source_uv', (b, points, 2))
    z = g.resource('projected_z', (b, points))
    valid_src = g.resource('source_valid', (b, points), 'uint32')
    destination = g.resource('destination_index', (b, points), 'uint32')
    minimum_bits = g.resource('min_depth_bits', (b, th, tw), 'uint32')
    winner = g.resource('winner_index', (b, th, tw), 'uint32')
    collisions = g.resource('collision_count_scratch', (b, th, tw), 'uint32')
    depth_out = g.resource('depth', (b, 1, th, tw))
    valid_out = g.resource('valid', (b, 1, th, tw), 'uint32')
    params = dict(batches=b, depthHeight=h, depthWidth=w, targetHeight=th, targetWidth=tw,
                  points=points, rgbDistortion=int(rgb_distortion), minDepth=minimum,
                  hasMaxDepth=int(maximum is not None), maxDepth=maximum if maximum is not None else 0.0, debug=int(debug))
    scratch = dict(minDepthBits=minimum_bits, winnerIndex=winner, collisionCount=collisions)
    projection = dict(sourceUVOut=uv, projectedZOut=z, sourceValidOut=valid_src, destinationIndexOut=destination)
    g.stage('clearRegistration', b * th * tw, params, scratch, (minimum_bits, winner, collisions))
    g.stage('projectRegistration', b * points, params,
            dict(depth=depth, depthRays=rays, rgbIntrinsics=rgb_k, transformRgbDepth=transform, rgbDistortion=coeff, **projection),
            (rays, uv, z, valid_src, destination))
    g.stage('registrationMin', b * points, params, dict(**projection, **scratch), (uv, z, valid_src, destination, minimum_bits))
    g.stage('registrationWinnerCount', b * points, params, dict(**projection, **scratch), (z, valid_src, destination, minimum_bits, winner, collisions))
    out = dict(depth=depth_out, valid=valid_out)
    bindings = dict(**scratch, depthOut=depth_out, validOut=valid_out)
    if debug:
        winner_out = g.resource('winner_source_index', (b, 1, th, tw), 'int64')
        count_out = g.resource('collision_count', (b, 1, th, tw), 'int64')
        bindings.update(winnerSourceIndexOut=winner_out, collisionCountOut=count_out)
        out.update(winner_source_index=winner_out, collision_count=count_out, source_uv=uv, projected_z=z, source_valid=valid_src)
    g.stage('finalizeRegistrationDebug' if debug else 'finalizeRegistration', b * th * tw, params, bindings, (minimum_bits, winner, collisions, depth_out, valid_out) + ((winner_out, count_out) if debug else ()))
    profile = _profile('depth_register', (b, h, w, th, tw), depth_distortion=depth_distortion,
                       rgb_distortion=rgb_distortion, cached_rays=cached_rays, min_depth=minimum,
                       max_depth=maximum, debug=debug)
    return g.finish(out, profile, ('shaderInt64',) if debug else ())
