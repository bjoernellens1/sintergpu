from pathlib import Path

import torch

from vkl.pipeline import PipelineExecutor, load_pipeline_spec


ROOT = Path(__file__).parents[1]


def test_rgbd_pose_geometry_recipe_runs_shared_canonical_contract():
    spec = load_pipeline_spec(ROOT / "recipes/depth/configs/pipelines/rgbd_pose_geometry_v1.yaml")
    out = PipelineExecutor(spec).run({
        "rgb": torch.rand(1, 3, 48, 64),
        "depth": torch.ones(1, 1, 48, 64),
        "intrinsics": torch.tensor([[[50.0, 0, 31.5], [0, 50.0, 23.5], [0, 0, 1]]]),
        "pose": torch.eye(4)[None],
    })

    assert out["geometry.xyz_camera"].shape == (1, 3, 48, 64)
    assert out["geometry.xyz_world"].shape == (1, 3, 48, 64)
    assert out["convert.valid"].all()
    assert "/2" in out["pyramid"]
