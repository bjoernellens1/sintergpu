import torch

from vkl.kernels import temporal_restore_rgb


def test_temporal_restore_rgb_cleaner_equivalent_neighbor_recovers_detail():
    target = torch.full((1, 3, 5, 5), 0.5)
    sharp = target.clone()
    sharp[:, :, 2, 2] = 1.0
    out = temporal_restore_rgb(target, sharp[:, None], alignment_confidence=torch.ones(1, 1, 1, 5, 5))

    assert out["restore_valid"][0, 0, 2, 2]
    assert out["restored_rgb"][0, 0, 2, 2] > target[0, 0, 2, 2]
    assert out["restore_confidence"][0, 0, 2, 2] > 0


def test_temporal_restore_rgb_rejects_dynamic_or_inconsistent_context():
    target = torch.full((1, 3, 3, 3), 0.5)
    context = torch.full((1, 1, 3, 3, 3), 0.1)
    out = temporal_restore_rgb(target, context, photometric_agreement=torch.zeros(1, 1, 1, 3, 3))

    assert torch.equal(out["restored_rgb"], target)
    assert not out["restore_valid"].any()
    assert not out["restore_confidence"].any()


def test_temporal_restore_rgb_requires_temporal_restoration_support():
    target = torch.full((1, 3, 5, 5), 0.5)
    sharper = target.clone()
    sharper[:, :, 2, 2] = 1.0
    out = temporal_restore_rgb(
        target,
        sharper[:, None],
        restoration_support=torch.zeros(1, 1, 1, 5, 5),
    )

    assert torch.equal(out["restored_rgb"], target)
    assert not out["restore_valid"].any()


def test_temporal_restore_rgb_rejects_weak_temporal_restoration_support():
    target = torch.full((1, 3, 5, 5), 0.5)
    sharper = target.clone()
    sharper[:, :, 2, 2] = 1.0
    out = temporal_restore_rgb(
        target,
        sharper[:, None],
        restoration_support=torch.full((1, 1, 1, 5, 5), 0.24),
    )

    assert torch.equal(out["restored_rgb"], target)
    assert not out["restore_valid"].any()


def test_temporal_restore_rgb_rejects_sparse_high_support_from_weak_context():
    target = torch.full((1, 3, 5, 5), 0.5)
    sharper = target.clone()
    sharper[:, :, 2, 2] = 1.0
    support = torch.zeros(1, 1, 1, 5, 5)
    support[:, :, :, 2, 2] = 1.0
    out = temporal_restore_rgb(target, sharper[:, None], restoration_support=support)

    assert torch.equal(out["restored_rgb"], target)
    assert not out["restore_valid"].any()


def test_temporal_restore_rgb_preserves_unsupported_target_pixels_exactly():
    target = torch.arange(27, dtype=torch.float32).reshape(1, 3, 3, 3) / 27
    context = torch.zeros(1, 1, 3, 3, 3)
    invalid = torch.zeros(1, 1, 1, 3, 3, dtype=torch.bool)
    out = temporal_restore_rgb(target, context, aligned_valid=invalid)

    assert torch.equal(out["restored_rgb"], target)


def test_temporal_restore_rgb_keeps_target_low_frequency_colour_dominant():
    target = torch.zeros(1, 3, 7, 7)
    target[:, 0] = 0.8
    context = torch.zeros(1, 1, 3, 7, 7)
    context[:, :, 1] = 1.0
    context[:, :, :, 3, 3] = 1.0
    out = temporal_restore_rgb(target, context)

    assert out["restored_rgb"][0, 0].mean() > 0.7
    assert out["restored_rgb"][0, 1].mean() < 0.2
