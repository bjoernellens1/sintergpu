"""Static gates for bounded GPU feature memory, work, and synchronization.

These are performance acceptance contracts rather than numerical GPU tests.
"""

import inspect
from pathlib import Path

from slang_rgbd_slam.features import GPUFeatures, feature_grids
from slang_rgbd_slam.tracker import Camera

SHADER = Path(__file__).parents[1] / "src/slang_rgbd_slam/shaders/features.slang"


def _entry(name):
    source = SHADER.read_text()
    start = source.index("void " + name + "(")
    following = source.find("[shader(", start)
    return source[start:] if following < 0 else source[start:following]


def test_feature_budget_all_levels_never_exceeds_persistent_capacity():
    for count in (1, 4, 31, 1024, 4096):
        for levels in (1, 2, 4, 8):
            grids = feature_grids(Camera(640, 480, 525, 525, 319.5, 239.5), count, levels, 4)
            assert sum(x * y * q for x, y, q in grids) <= count
            assert all(q <= 4 for _, _, q in grids)


def test_matching_caches_target_tiles_with_uniform_barriers():
    source = SHADER.read_text()
    assert "groupshared uint targetTile[MATCH_GROUP_SIZE*8]" in source
    for name in ("matchForward", "matchReverse"):
        kernel = _entry(name)
        assert "begin+=MATCH_GROUP_SIZE" in kernel
        assert kernel.count("GroupMemoryBarrierWithGroupSync()") >= 2
        assert "return;" not in kernel
        assert "countbits(" in kernel
        assert "min(" in kernel and "cfg.maxFeatures" in kernel


def test_selector_parallelizes_cell_pixels_and_bounds_local_candidates():
    kernel = _entry("selectCells")
    assert "k=lane;k<area;k+=64" in kernel
    assert "localScore[4]" in kernel and "localIndex[4]" in kernel
    assert "stride=32;stride>0;stride>>=1" in kernel
    assert "compactBase+lane<cfg.maxFeatures" in kernel


def test_frontend_record_methods_never_readback_or_allocate_frame_buffers():
    for method in (GPUFeatures.record_extract, GPUFeatures.record_match):
        source = inspect.getsource(method)
        for forbidden in ("to_numpy(", "create_buffer(", "self._buffer(", "submit_command_buffer("):
            assert forbidden not in source


def test_fast_and_descriptor_have_bounded_pixel_neighborhoods():
    assert "start<16" in _entry("fastDetect")
    assert "n<9" in _entry("fastDetect")
    descriptor = _entry("describe")
    assert "word<8" in descriptor and "bit<32" in descriptor
    assert "x*x+y*y<=225" in descriptor
    assert "cfg.maxFeatures" in descriptor
