import pytest
import torch

from vkl.kernels.reference import project_depth_evidence
from vkl.kernels.triton import project_depth_evidence_triton


def _inputs(device: torch.device, batch: int):
    xyz = torch.tensor(
        [[[0.0, 0.0, 2.0], [0.0, 0.0, 1.0], [20.0, 0.0, 2.0], [float("nan"), 0.0, 2.0]]],
        device=device,
    ).expand(batch, -1, -1).clone()
    visible = torch.tensor([[True, True, True, True]], device=device).expand(batch, -1).clone()
    provisional = torch.tensor([[True, False, True, True]], device=device).expand(batch, -1).clone()
    candidate = torch.tensor([[False, True, False, False]], device=device).expand(batch, -1).clone()
    w2c = torch.eye(4, device=device)[None].expand(batch, -1, -1).clone()
    intrinsics = torch.tensor(
        [[[4.0, 0.0, 2.0], [0.0, 4.0, 2.0], [0.0, 0.0, 1.0]]], device=device
    ).expand(batch, -1, -1).clone()
    depth = torch.full((batch, 1, 5, 5), 2.0, device=device)
    return xyz, visible, provisional, candidate, w2c, intrinsics, depth


@pytest.mark.parametrize("batch", [1, 8])
def test_triton_project_depth_evidence_matches_prevalidated_reference(batch: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = _inputs(device, batch)
    expected = project_depth_evidence(*args, support_threshold=0.1, conflict_threshold=0.1, validate=False)
    actual = project_depth_evidence_triton(*args, support_threshold=0.1, conflict_threshold=0.1, validate=False)

    assert project_depth_evidence_triton.dispatch_metadata["actual"] == (
        "triton" if device.type == "cuda" else "eager_fallback"
    )
    for name in ("eligible", "projected_valid", "sensor_valid", "support", "conflict"):
        assert torch.equal(actual[name], expected[name]), name
    assert torch.equal(actual["depth_residual"], expected["depth_residual"])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton dispatch requires CUDA/HIP")
def test_triton_project_depth_evidence_masks_nonfinite_projection_before_indexing():
    device = torch.device("cuda")
    xyz, visible, provisional, candidate, w2c, intrinsics, depth = _inputs(device, 1)
    xyz = xyz.clone()
    xyz[:, 0, 0] = torch.finfo(xyz.dtype).max
    expected = project_depth_evidence(
        xyz, visible, provisional, candidate, w2c, intrinsics, depth,
        support_threshold=0.1, conflict_threshold=0.1, validate=False,
    )
    actual = project_depth_evidence_triton(
        xyz, visible, provisional, candidate, w2c, intrinsics, depth,
        support_threshold=0.1, conflict_threshold=0.1, validate=False,
    )

    assert not expected["projected_valid"][:, 0].any()
    assert torch.equal(actual["projected_valid"], expected["projected_valid"])
    assert torch.equal(actual["depth_residual"], expected["depth_residual"])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton dispatch requires CUDA/HIP")
def test_triton_project_depth_evidence_uses_input_dtype_epsilon():
    device = torch.device("cuda")
    xyz, visible, provisional, candidate, w2c, intrinsics, depth = _inputs(device, 1)
    dtype = torch.float16
    xyz = xyz.to(dtype)
    xyz[:, 0] = torch.tensor([0.0, 0.0, 5e-4], device=device, dtype=dtype)
    w2c, intrinsics, depth = w2c.to(dtype), intrinsics.to(dtype), depth.to(dtype)
    expected = project_depth_evidence(
        xyz, visible, provisional, candidate, w2c, intrinsics, depth,
        support_threshold=0.1, conflict_threshold=0.1, validate=False,
    )
    actual = project_depth_evidence_triton(
        xyz, visible, provisional, candidate, w2c, intrinsics, depth,
        support_threshold=0.1, conflict_threshold=0.1, validate=False,
    )

    assert not expected["projected_valid"][:, 0].any()
    assert torch.equal(actual["projected_valid"], expected["projected_valid"])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton dispatch requires CUDA/HIP")
def test_triton_project_depth_evidence_keeps_prevalidated_structural_checks():
    args = list(_inputs(torch.device("cuda"), 1))
    args[1] = args[1].to(torch.uint8)
    with pytest.raises(ValueError):
        project_depth_evidence_triton(*args, support_threshold=0.1, conflict_threshold=0.1, validate=False)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton dispatch requires CUDA/HIP")
def test_triton_project_depth_evidence_keeps_prevalidated_threshold_checks():
    args = _inputs(torch.device("cuda"), 1)
    with pytest.raises(ValueError):
        project_depth_evidence_triton(*args, support_threshold=float("nan"), conflict_threshold=0.1, validate=False)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton dispatch requires CUDA/HIP")
def test_triton_project_depth_evidence_handles_unaligned_tail():
    device = torch.device("cuda")
    xyz, visible, provisional, candidate, w2c, intrinsics, depth = _inputs(device, 1)
    xyz = xyz.repeat(1, 65, 1)
    visible = visible.repeat(1, 65)
    provisional = provisional.repeat(1, 65)
    candidate = candidate.repeat(1, 65)
    expected = project_depth_evidence(
        xyz, visible, provisional, candidate, w2c, intrinsics, depth,
        support_threshold=0.1, conflict_threshold=0.1, validate=False,
    )
    actual = project_depth_evidence_triton(
        xyz, visible, provisional, candidate, w2c, intrinsics, depth,
        support_threshold=0.1, conflict_threshold=0.1, validate=False,
    )

    for name in ("eligible", "projected_valid", "sensor_valid", "support", "conflict", "depth_residual"):
        assert torch.equal(actual[name], expected[name]), name


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton dispatch requires CUDA/HIP")
def test_triton_project_depth_evidence_falls_back_for_noncontiguous_input_without_hidden_copy():
    device = torch.device("cuda")
    xyz, visible, provisional, candidate, w2c, intrinsics, depth = _inputs(device, 1)
    xyz = xyz.transpose(1, 2).contiguous().transpose(1, 2)
    assert xyz.shape == (1, 4, 3)
    assert not xyz.is_contiguous()

    expected = project_depth_evidence(
        xyz, visible, provisional, candidate, w2c, intrinsics, depth,
        support_threshold=0.1, conflict_threshold=0.1, validate=False,
    )
    actual = project_depth_evidence_triton(
        xyz, visible, provisional, candidate, w2c, intrinsics, depth,
        support_threshold=0.1, conflict_threshold=0.1, validate=False,
    )

    assert project_depth_evidence_triton.dispatch_metadata == {
        "requested": "triton",
        "actual": "eager_fallback",
        "reason": "noncontiguous_input",
    }
    for name in expected:
        assert torch.equal(actual[name], expected[name]), name


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton dispatch requires CUDA/HIP")
def test_triton_project_depth_evidence_falls_back_for_empty_workload():
    device = torch.device("cuda")
    xyz = torch.empty((1, 0, 3), device=device)
    mask = torch.empty((1, 0), dtype=torch.bool, device=device)
    w2c = torch.eye(4, device=device)[None]
    intrinsics = torch.eye(3, device=device)[None]
    depth = torch.ones((1, 1, 2, 2), device=device)
    args = (xyz, mask, mask.clone(), mask.clone(), w2c, intrinsics, depth)

    expected = project_depth_evidence(*args, support_threshold=0.1, conflict_threshold=0.1, validate=False)
    actual = project_depth_evidence_triton(*args, support_threshold=0.1, conflict_threshold=0.1, validate=False)

    assert project_depth_evidence_triton.dispatch_metadata["reason"] == "empty_workload"
    for name in expected:
        assert actual[name].numel() == 0
        assert torch.equal(actual[name], expected[name]), name


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton dispatch requires CUDA/HIP")
def test_triton_project_depth_evidence_writes_caller_owned_outputs_without_stale_values():
    device = torch.device("cuda")
    args = _inputs(device, 1)
    first = project_depth_evidence_triton(*args, support_threshold=0.1, conflict_threshold=0.1, validate=False)
    outputs = {name: torch.empty_like(value) for name, value in first.items()}
    xyz, visible, provisional, candidate, w2c, intrinsics, depth = args
    visible = visible.clone()
    visible.zero_()

    actual = project_depth_evidence_triton(
        xyz, visible, provisional, candidate, w2c, intrinsics, depth,
        support_threshold=0.1, conflict_threshold=0.1, validate=False, out=outputs,
    )

    assert all(actual[name] is value for name, value in outputs.items())
    assert not actual["eligible"].any()
    assert not actual["projected_valid"].any()
    assert not actual["sensor_valid"].any()
    assert not actual["support"].any()
    assert not actual["conflict"].any()
    assert torch.equal(actual["depth_residual"], torch.zeros_like(actual["depth_residual"]))
