"""Checked launch geometry for the experimental VKL depth-evidence shaders."""
from dataclasses import dataclass
import math
import struct

_U32_MAX = (1 << 32) - 1
_EXACT_FLOAT_INTEGER = 1 << 24


@dataclass(frozen=True)
class DepthEvidenceLaunch:
    batches: int
    height: int
    width: int
    grid_height: int
    grid_width: int
    near: float
    far: float
    global_groups: int
    grid_groups: int
    input_elements: int
    global_elements: int
    grid_elements: int


def checked_depth_evidence_launch(batches, height, width, grid_height, grid_width,
                                  near, far, max_groups_x):
    """Validate dimensions and return launch/layout sizes. Caller validates buffer content and device limits."""
    dims = (batches, height, width, grid_height, grid_width, max_groups_x)
    if any(type(x) is not int or x <= 0 for x in dims):
        raise ValueError('dimensions and max_groups_x must be positive integers')
    if not (math.isfinite(near) and math.isfinite(far) and 0 <= near < far):
        raise ValueError('bounds must be finite and 0 <= near < far')
    try:
        near32 = struct.unpack('f', struct.pack('f', near))[0]
        far32 = struct.unpack('f', struct.pack('f', far))[0]
    except (OverflowError, struct.error) as exc:
        raise ValueError('bounds must be representable in float32') from exc
    if not (math.isfinite(near32) and math.isfinite(far32) and 0 <= near32 < far32):
        raise ValueError('bounds must remain distinct and finite in float32')
    gh, gw = min(grid_height, height), min(grid_width, width)
    area = height * width
    inputs = batches * area
    bins = batches * gh * gw
    # All float counters must stay exact, including the two directed pairs per pixel.
    if 2 * area > _EXACT_FLOAT_INTEGER:
        raise ValueError('pixel and pair counters exceed exact float integer range')
    # Byte-size bound also prevents u32 element indexing wrap for every binding.
    if any(4 * n > _U32_MAX for n in (inputs, batches * 8, bins * 8)):
        raise ValueError('buffer exceeds checked 32-bit byte range')
    if gh * height > _U32_MAX or gw * width > _U32_MAX:
        raise ValueError('grid boundary product exceeds uint32')
    global_groups = (batches + 63) // 64
    grid_groups = (bins + 63) // 64
    # The last invocation ID, including inactive lanes, must fit uint32.
    if max(global_groups, grid_groups) * 64 - 1 > _U32_MAX:
        raise ValueError('rounded dispatch overflows uint32')
    if max(global_groups, grid_groups) > max_groups_x:
        raise ValueError('dispatch exceeds device maxComputeWorkGroupCount[0]')
    return DepthEvidenceLaunch(batches,height,width,gh,gw,near32,far32,global_groups,grid_groups,
                               inputs,batches * 8,bins * 8)
