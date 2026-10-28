"""Launch bounds for the experimental dense depth regularization stage."""
from dataclasses import dataclass
import math
import struct


@dataclass(frozen=True)
class DenseDepthRegularizeLaunch:
    batches: int
    height: int
    width: int
    channels: int
    min_support: int
    max_rgb_difference: float
    max_relative_deviation: float
    pixels: int
    rgb_elements: int
    groups: int


def checked_dense_depth_regularize_launch(batches, height, width, channels, min_support,
                                          max_rgb_difference, max_relative_deviation, max_groups_x):
    dims=(batches,height,width,channels,min_support,max_groups_x)
    if any(type(x) is not int or x <= 0 for x in dims):
        raise ValueError('dimensions and limits must be positive integers')
    if channels not in (1,3) or min_support > 9:
        raise ValueError('channels must be 1 or 3 and min_support must be in [1,9]')
    if height > 0x7ffffffe or width > 0x7ffffffe:
        raise ValueError('coordinates exceed signed int32 stencil range')
    try:
        thresholds=tuple(struct.unpack('f',struct.pack('f',v))[0] for v in (max_rgb_difference,max_relative_deviation))
    except (OverflowError,struct.error) as exc:
        raise ValueError('thresholds must be representable float32') from exc
    if not all(math.isfinite(v) and v>=0 for v in (max_rgb_difference,max_relative_deviation,*thresholds)):
        raise ValueError('thresholds must be nonnegative finite float32')
    pixels=batches*height*width; rgb_elements=pixels*channels
    if 4*rgb_elements > 0xffffffff:
        raise ValueError('input exceeds checked uint32 byte range')
    groups=(pixels+63)//64
    if groups>max_groups_x or groups*64-1>0xffffffff:
        raise ValueError('dispatch exceeds device or uint32 limits')
    return DenseDepthRegularizeLaunch(batches,height,width,channels,min_support,*thresholds,pixels,rgb_elements,groups)
