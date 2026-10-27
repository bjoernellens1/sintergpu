"""Deterministic static world texture on independently ray-rendered room surfaces.

Truth poses are supplied only to render/evaluate the fixture. No tracker uses
these poses as estimates. The volumetric procedural texture is sampled at world
intersection points, so a surface retains its appearance across viewpoints.
"""

import numpy as np

from .dataset import Frame
from .synthetic import render_depth
from .tracker import Camera


def _lattice_noise(integer, seed):
    xyz = np.asarray(integer, np.int64)
    with np.errstate(over="ignore"):
        value = (
            (xyz[..., 0].astype(np.uint32) * np.uint32(0x8DA6B343))
            ^ (xyz[..., 1].astype(np.uint32) * np.uint32(0xD8163841))
            ^ (xyz[..., 2].astype(np.uint32) * np.uint32(0xCB1AB31F))
            ^ np.uint32(seed)
        )
        value ^= value >> np.uint32(16)
        value *= np.uint32(0x7FEB352D)
        value ^= value >> np.uint32(15)
        value *= np.uint32(0x846CA68B)
        value ^= value >> np.uint32(16)
    return value.astype(np.float64) / (2**32 - 1) * 2 - 1


def _noise(points, scale, seed):
    lattice = np.asarray(points, np.float64) / scale
    base = np.floor(lattice).astype(np.int64)
    fraction = lattice - base
    output = np.zeros(lattice.shape[:-1])
    for z in (0, 1):
        for y in (0, 1):
            for x in (0, 1):
                corner = np.array([x, y, z])
                weight = np.prod(np.where(corner > 0, fraction, 1 - fraction), axis=-1)
                output += weight * _lattice_noise(base + corner, seed)
    return output


def world_texture(world):
    """RGB bytes from position only: multiscale checkers and seeded lattice noise."""
    world = np.asarray(world, np.float64)
    if world.shape[-1:] != (3,) or not np.isfinite(world).all():
        raise ValueError("texture points must be finite xyz")
    checker = (np.floor(world / 0.14).astype(np.int64).sum(axis=-1) % 2) * 2 - 1
    coarse = _noise(world, 0.1, 173)
    fine = _noise(world, 0.04, 293)
    luminance = 128 + 62 * checker + 55 * coarse + 24 * fine
    tint = _noise(world, 0.35, 719) * 14
    return np.clip(
        np.stack((luminance + tint, luminance, luminance - tint), axis=-1), 0, 255
    ).astype(np.uint8)


def render_textured_rgb(camera, depth, world_from_camera):
    yy, xx = np.indices(depth.shape, dtype=np.float64)
    points = np.stack(
        ((xx - camera.cx) * depth / camera.fx, (yy - camera.cy) * depth / camera.fy, depth), axis=-1
    )
    pose = np.asarray(world_from_camera, np.float64)
    world = points @ pose[:3, :3].T + pose[:3, 3]
    rgb = world_texture(world)
    rgb[depth <= 0] = 0
    return rgb


def textured_sequence(frames=40, width=160, height=120, closed_loop=True):
    """Smooth small RGB-D camera motion; closed path returns exactly to identity."""
    if not isinstance(frames, int) or isinstance(frames, bool) or frames < 1:
        raise ValueError("frames must be a positive integer")
    camera = Camera(width, height, width * 0.9, width * 0.9, (width - 1) / 2, (height - 1) / 2)
    result = []
    for i in range(frames):
        phase = i / max(frames - 1, 1)
        progress = np.sin(np.pi * phase) ** 2 if closed_loop else phase
        if closed_loop and i in (0, frames - 1):
            progress = 0.0
        angle = 0.045 * progress
        pose = np.eye(4)
        pose[:3, :3] = [
            [np.cos(angle), 0, np.sin(angle)],
            [0, 1, 0],
            [-np.sin(angle), 0, np.cos(angle)],
        ]
        pose[:3, 3] = [
            0.08 * progress,
            0.012 * np.sin(2 * np.pi * phase) if progress else 0.0,
            0.035 * progress,
        ]
        depth = render_depth(camera, pose)
        result.append(Frame(i / 30, depth, render_textured_rgb(camera, depth, pose), pose))
    return camera, result
