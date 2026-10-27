from pathlib import Path

import torch

from vkl.data import SyntheticRGBDSource
from vkl.kernels import canonical_rescale
from vkl.pipeline import PipelineExecutor, load_pipeline_spec


ROOT = Path(__file__).resolve().parents[1]
PIPELINE = ROOT / "recipes/splatograph/configs/pipelines/fastpath_v1.yaml"
CONDITIONED_PIPELINE = (
    ROOT / "recipes/splatograph/configs/pipelines/conditioned_fastpath_v1.yaml"
)


def _inputs(batch: int = 2) -> dict[str, torch.Tensor]:
    reference = torch.eye(4).repeat(batch, 1, 1)
    pose = reference.clone()
    if batch > 1:
        pose[1, 0, 3] = 0.12
    intrinsics = torch.eye(3).repeat(batch, 1, 1)
    return {
        "rgb": torch.rand(batch, 3, 480, 640),
        "depth": torch.rand(batch, 1, 480, 640) * 5.0,
        "pose": pose,
        "reference_pose": reference,
        "frame_gap": torch.ones(batch, dtype=torch.int64),
        "intrinsics": intrinsics,
    }


def test_splatograph_fastpath_recipe_runs():
    executor = PipelineExecutor(load_pipeline_spec(PIPELINE))
    out = executor.run(_inputs())

    assert out["filter.admit_mask"].shape == (2,)
    assert out["evidence.tile_score"].shape == (2, 12, 16)
    assert out["canonical.rgb_valid"].shape == (2, 1, 480, 640)
    assert out["canonical.depth_valid"].shape == (2, 1, 480, 640)
    assert out["canonical.depth_measurement_valid"].shape == (2, 1, 480, 640)


def test_fastpath_does_not_turn_invalid_conditioning_into_image_evidence():
    executor = PipelineExecutor(load_pipeline_spec(PIPELINE))
    inputs = _inputs(batch=1)
    inputs["rgb"][0, :, 20:40, 20:40] = float("nan")
    inputs["depth"][0, :, 60:80, 60:80] = float("nan")

    out = executor.run(inputs)

    assert torch.isfinite(out["canonical.rgb"]).all()
    assert not out["canonical.rgb_valid"][0, 0, 20:40, 20:40].any()
    # With no explicit depth-domain mask, the full camera grid remains the
    # conditioning domain. Missing/non-finite samples are occupancy failures.
    assert out["canonical.depth_valid"][0, 0, 60:80, 60:80].all()
    assert not out["canonical.depth_measurement_valid"][0, 0, 60:80, 60:80].any()
    assert out["evidence.frame_rgb_valid_fraction"].item() < 1.0
    assert out["evidence.frame_depth_domain_fraction"].item() == 1.0
    assert out["evidence.frame_depth_measurement_fraction"].item() < 1.0
    assert torch.isfinite(out["evidence.frame_score"]).all()


def test_canonical_rescale_sanitizes_nonfinite_depth_without_shrinking_default_domain():
    rgb = torch.rand(1, 3, 4, 4)
    depth = torch.ones(1, 1, 4, 4)
    depth[0, 0, 1, 2] = float("nan")

    out = canonical_rescale(
        rgb=rgb,
        depth=depth,
        target_pixels=16,
        min_width=1,
        min_height=1,
    )

    assert out["depth"][0, 0, 1, 2].item() == 0.0
    assert out["depth_valid"][0, 0, 1, 2]
    assert not out["depth_measurement_valid"][0, 0, 1, 2]
    assert torch.isfinite(out["depth"]).all()


def test_conditioned_fastpath_keeps_registered_occupancy_separate_from_domain():
    source = SyntheticRGBDSource(
        batch_size=1,
        height=16,
        width=16,
        ring_size=1,
        device="cpu",
        seed=7,
    )
    inputs = source.build()[0]
    executor = PipelineExecutor(load_pipeline_spec(CONDITIONED_PIPELINE))

    out = executor.run(inputs)

    assert out["canonical.depth_valid"].all()
    measurement_fraction = out["evidence.frame_depth_measurement_fraction"].item()
    coverage = out["evidence.frame_depth_coverage"].item()
    assert 0.0 < measurement_fraction < 1.0
    assert abs(measurement_fraction - coverage) < 1e-6
    assert out["evidence.frame_depth_domain_fraction"].item() == 1.0
