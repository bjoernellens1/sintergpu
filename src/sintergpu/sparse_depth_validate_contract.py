"""Checked dimensions for experimental sparse depth validation."""
from dataclasses import dataclass
import math
import struct


@dataclass(frozen=True)
class SparseDepthValidateLaunch:
    batches: int
    points: int
    height: int
    width: int
    min_neighbors: int
    max_relative_dispersion: float
    depth_elements: int
    point_elements: int
    groups: int


def checked_sparse_depth_validate_launch(batches,points,height,width,min_neighbors,max_relative_dispersion,max_groups_x):
    if any(type(x) is not int or x<=0 for x in (batches,points,height,width,min_neighbors,max_groups_x)):
        raise ValueError('dimensions and limits must be positive integers')
    if min_neighbors>9:
        raise ValueError('min_neighbors must be in [1,9]')
    if height>0x7ffffffe or width>0x7ffffffe:
        raise ValueError('coordinates exceed signed int32 stencil range')
    try:
        threshold=struct.unpack('f',struct.pack('f',max_relative_dispersion))[0]
    except (OverflowError,struct.error) as exc:
        raise ValueError('threshold must be representable float32') from exc
    if not all(math.isfinite(v) and v>=0 for v in (max_relative_dispersion,threshold)):
        raise ValueError('threshold must be nonnegative finite float32')
    depths=batches*height*width; n=batches*points
    if 4*depths>0xffffffff or 8*n>0xffffffff:
        raise ValueError('input exceeds checked uint32 byte range')
    groups=(n+63)//64
    if groups>max_groups_x or groups*64-1>0xffffffff:
        raise ValueError('dispatch exceeds device or uint32 limits')
    return SparseDepthValidateLaunch(batches,points,height,width,min_neighbors,threshold,depths,n,groups)
