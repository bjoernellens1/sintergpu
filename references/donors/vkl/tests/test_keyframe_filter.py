import pytest
import torch

from vkl.kernels.reference.keyframe_filter import (
    REASON_MAX_GAP,
    REASON_NOVELTY,
    REASON_TRANSLATION,
    keyframe_filter,
)


def test_keyframe_filter_motion_and_gap():
    ref = torch.eye(4).repeat(3, 1, 1)
    pose = ref.clone()
    pose[1, 0, 3] = 0.12
    out = keyframe_filter(
        pose=pose,
        reference_pose=ref,
        frame_score=torch.tensor([0.2, 0.8, 0.2]),
        tile_score=torch.zeros(3, 12, 16),
        frame_gap=torch.tensor([1, 1, 40]),
    )
    assert out["admit_mask"].tolist() == [False, True, True]
    assert int(out["reason_bits"][1]) & REASON_TRANSLATION
    assert int(out["reason_bits"][2]) & REASON_MAX_GAP


def test_keyframe_filter_motion_requires_quality_before_gap_backstop():
    ref = torch.eye(4)[None]
    pose = ref.clone()
    pose[:, 0, 3] = 0.12
    out = keyframe_filter(
        pose=pose,
        reference_pose=ref,
        frame_score=torch.tensor([0.1]),
        tile_score=torch.zeros(1, 4, 4),
        frame_gap=torch.tensor([1]),
    )
    assert out["admission_trigger"].item()
    assert not out["quality_gate"].item()
    assert not out["admit_mask"].item()

    backstop = keyframe_filter(
        pose=pose,
        reference_pose=ref,
        frame_score=torch.tensor([0.1]),
        tile_score=torch.zeros(1, 4, 4),
        frame_gap=torch.tensor([30]),
    )
    assert backstop["admit_mask"].item()
    assert int(backstop["reason_bits"][0]) & REASON_MAX_GAP


def test_keyframe_filter_local_tile_quality_can_gate_motion():
    ref = torch.eye(4)[None]
    pose = ref.clone()
    pose[:, 0, 3] = 0.12
    tiles = torch.zeros(1, 4, 4)
    tiles[:, 0, 0] = 1.0
    out = keyframe_filter(
        pose=pose,
        reference_pose=ref,
        frame_score=torch.tensor([0.1]),
        tile_score=tiles,
        min_tile_score=0.8,
        topk_tile_fraction=1 / 16,
    )
    assert out["quality_gate"].item()
    assert out["admit_mask"].item()


def test_keyframe_filter_depth_novelty_is_not_vetoed_by_poor_rgb():
    out = keyframe_filter(
        frame_score=torch.tensor([0.05]),
        tile_score=torch.zeros(1, 4, 4),
        depth_novelty=torch.tensor([0.8]),
        has_depth=torch.tensor([True]),
        min_novelty=0.2,
    )
    assert not out["quality_gate"].item()
    assert out["non_rgb_trigger"].item()
    assert out["admit_mask"].item()
    assert int(out["reason_bits"][0]) & REASON_NOVELTY


def test_keyframe_filter_stationary_quality_is_not_novelty():
    ref = torch.eye(4).unsqueeze(0)
    tiles = torch.zeros(1, 12, 16)
    tiles[:, :2, :2] = 1.0
    out = keyframe_filter(
        pose=ref,
        reference_pose=ref,
        frame_score=torch.tensor([0.1]),
        tile_score=tiles,
        frame_gap=torch.tensor([1]),
        min_tile_score=0.1,
        topk_tile_fraction=0.02,
    )
    assert not bool(out["admit_mask"][0])


def test_keyframe_filter_admits_rgb_novelty_only_when_rgb_available():
    tiles = torch.zeros(2, 12, 16)
    out = keyframe_filter(
        frame_score=torch.tensor([0.8, 0.8]),
        tile_score=tiles,
        rgb_novelty=torch.tensor([0.5, 0.5]),
        has_rgb=torch.tensor([True, False]),
        min_novelty=0.2,
    )
    assert out["admit_mask"].tolist() == [True, False]
    assert int(out["reason_bits"][0]) & REASON_NOVELTY


def test_keyframe_filter_rejects_nonrigid_pose_by_default():
    bad = torch.eye(4)[None]
    bad[:, 0, 0] = 2
    with pytest.raises(ValueError):
        keyframe_filter(
            pose=bad,
            reference_pose=torch.eye(4)[None],
            frame_score=torch.ones(1),
            tile_score=torch.ones(1, 4, 4),
        )


def test_keyframe_filter_value_validation_is_optional_on_prevalidated_hot_path():
    with pytest.raises(ValueError, match="finite"):
        keyframe_filter(
            frame_score=torch.tensor([float("nan")]),
            tile_score=torch.ones(1, 2, 2),
        )
    with pytest.raises(ValueError, match="non-negative"):
        keyframe_filter(
            frame_score=torch.ones(1),
            tile_score=torch.ones(1, 2, 2),
            frame_gap=torch.tensor([-1]),
        )

    out = keyframe_filter(
        frame_score=torch.tensor([float("nan")]),
        tile_score=torch.ones(1, 2, 2),
        frame_gap=torch.tensor([-1]),
        validate=False,
    )
    assert out["admit_mask"].shape == (1,)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA/HIP")
def test_keyframe_filter_prevalidated_path_compiles_fullgraph_on_gpu():
    pose = torch.eye(4, device="cuda")[None].repeat(2, 1, 1)
    pose[1, 0, 3] = 0.12
    reference = torch.eye(4, device="cuda")[None].repeat(2, 1, 1)
    frame_score = torch.tensor([0.8, 0.2], device="cuda")
    tile_score = torch.ones(2, 4, 4, device="cuda")
    frame_gap = torch.tensor([1, 40], dtype=torch.int32, device="cuda")

    def transaction(frame_score, tile_score, pose, reference, frame_gap):
        return keyframe_filter(
            frame_score=frame_score,
            tile_score=tile_score,
            pose=pose,
            reference_pose=reference,
            frame_gap=frame_gap,
            validate=False,
        )["admit_mask"]

    eager = transaction(frame_score, tile_score, pose, reference, frame_gap)
    compiled = torch.compile(transaction, fullgraph=True)(
        frame_score, tile_score, pose, reference, frame_gap
    )
    assert torch.equal(compiled, eager)


def test_keyframe_filter_rejects_broadcastable_frame_gap():
    with pytest.raises(ValueError):
        keyframe_filter(
            frame_score=torch.ones(2),
            tile_score=torch.zeros(2, 4, 4),
            frame_gap=torch.tensor([[1], [2]]),
        )


def test_keyframe_filter_rejects_invalid_topk_fraction():
    with pytest.raises(ValueError):
        keyframe_filter(
            frame_score=torch.ones(1),
            tile_score=torch.zeros(1, 4, 4),
            topk_tile_fraction=1.1,
        )
