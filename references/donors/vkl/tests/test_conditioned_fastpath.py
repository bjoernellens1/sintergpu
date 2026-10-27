from pathlib import Path

import pytest
import torch

from vkl.data import SyntheticRGBDSource
from vkl.kernels import REGISTRY
from vkl.pipeline import PipelineExecutor, PipelineSpec, load_pipeline_spec

ROOT = Path(__file__).resolve().parents[1]
PIPELINE = ROOT / "recipes/splatograph/configs/pipelines/conditioned_fastpath_v1.yaml"


def test_conditioned_fastpath_composes_camera_validity_through_admission():
    spec = load_pipeline_spec(PIPELINE)
    executor = PipelineExecutor(spec)
    sample = SyntheticRGBDSource(
        batch_size=2, height=48, width=64, ring_size=1, device="cpu", seed=3
    ).build()[0]
    out = executor.run(sample)
    assert executor.max_parallel_nodes == 2
    assert out["canonical.rgb"].shape == (2, 3, 48, 64)
    assert out["canonical.depth"].shape == (2, 1, 48, 64)
    assert out["canonical.rgb_valid"].all()
    assert torch.allclose(out["evidence.frame_rgb_valid_fraction"], torch.ones(2))
    assert (out["evidence.frame_depth_coverage"] > 0).all()
    assert (out["evidence.frame_depth_coverage"] < 1).all()
    assert out["register_depth.valid"].dtype == torch.bool
    assert out["filter.admit_mask"].shape == (2,)
    assert out["filter.quality_gate"].shape == (2,)


def test_conditioned_fastpath_registers_raw_depth_with_independent_resolution():
    spec = load_pipeline_spec(PIPELINE)
    executor = PipelineExecutor(PipelineSpec(spec.name, spec.nodes))
    rgb_intrinsics = torch.tensor(
        [[[50.0, 0.0, 31.5], [0.0, 50.0, 23.5], [0.0, 0.0, 1.0]]]
    )
    depth_intrinsics = torch.tensor(
        [[[25.0, 0.0, 15.5], [0.0, 25.0, 11.5], [0.0, 0.0, 1.0]]]
    )
    pose = torch.eye(4)[None]
    inputs = {
        "rgb": torch.rand(1, 3, 48, 64),
        "depth": torch.full((1, 1, 24, 32), 2.0),
        "intrinsics": rgb_intrinsics,
        "depth_intrinsics": depth_intrinsics,
        "target_intrinsics": rgb_intrinsics.clone(),
        "transform_rgb_depth": torch.eye(4)[None],
        "output_hw": [48, 64],
        "pose": pose,
        "reference_pose": pose.clone(),
        "frame_gap": torch.ones(1, dtype=torch.int32),
    }

    out = executor.run(inputs)

    registered_valid = out["register_depth"]["valid"]
    canonical = out["canonical"]
    evidence = out["evidence"]
    assert canonical["rgb"].shape == (1, 3, 48, 64)
    assert canonical["depth"].shape == (1, 1, 48, 64)
    assert torch.isfinite(canonical["depth"]).all()
    # Registration validity is measurement occupancy. It must not be re-labeled
    # as the geometric conditioning domain consumed by K0/K1.
    assert 0 < registered_valid.float().mean().item() < 1
    assert canonical["depth_valid"].all()
    assert torch.equal(evidence["frame_depth_domain_fraction"], torch.ones(1))
    assert torch.allclose(
        evidence["frame_depth_coverage"],
        registered_valid.float().mean((-2, -1))[:, 0],
    )


def test_conditioned_fastpath_keeps_finite_unregistered_depth_out_of_measurement_evidence(monkeypatch):
    def register_with_finite_hole(*, depth, output_hw, **_):
        registered = torch.full(
            (depth.shape[0], 1, *output_hw), 2.0, dtype=depth.dtype, device=depth.device
        )
        valid = torch.ones_like(registered, dtype=torch.bool)
        valid[..., 8, 12] = False
        return {"depth": registered, "valid": valid}

    monkeypatch.setitem(REGISTRY, "depth_register", register_with_finite_hole)
    spec = load_pipeline_spec(PIPELINE)
    executor = PipelineExecutor(PipelineSpec(spec.name, spec.nodes))
    intrinsics = torch.tensor(
        [[[50.0, 0.0, 31.5], [0.0, 50.0, 23.5], [0.0, 0.0, 1.0]]]
    )
    pose = torch.eye(4)[None]
    out = executor.run(
        {
            "rgb": torch.rand(1, 3, 48, 64),
            "depth": torch.full((1, 1, 24, 32), 2.0),
            "intrinsics": intrinsics,
            "depth_intrinsics": intrinsics,
            "target_intrinsics": intrinsics,
            "transform_rgb_depth": torch.eye(4)[None],
            "output_hw": [48, 64],
            "pose": pose,
            "reference_pose": pose.clone(),
            "frame_gap": torch.ones(1, dtype=torch.int32),
        }
    )

    canonical = out["canonical"]
    evidence = out["evidence"]
    assert canonical["depth_valid"].all()
    assert not canonical["depth_measurement_valid"][0, 0, 8, 12]
    assert canonical["depth"][0, 0, 8, 12] == 0
    assert torch.allclose(
        evidence["frame_depth_measurement_fraction"], torch.tensor([1.0 - 1.0 / (48 * 64)])
    )
    assert torch.equal(
        evidence["frame_depth_coverage"], evidence["frame_depth_measurement_fraction"]
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA/HIP")
def test_conditioned_fastpath_has_real_parallel_frontend_on_gpu():
    spec = load_pipeline_spec(PIPELINE)
    executor = PipelineExecutor(spec)
    sample = SyntheticRGBDSource(
        batch_size=1, height=48, width=64, ring_size=1, device="cuda", seed=4
    ).build()[0]
    executor.run(sample, mode="streams", max_streams=4)
    assert executor.max_parallel_nodes == 2
    assert executor.last_execution_mode == "streams"
    assert executor.last_stream_count == 2
