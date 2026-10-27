from pathlib import Path

import torch

from vkl.pipeline import PipelineExecutor, load_pipeline_spec


ROOT = Path(__file__).parents[1]


def test_rgb_pose_direct_depth_recipe_runs():
    h, w, focal, baseline = 24, 32, 30.0, 0.2
    image = torch.rand(1, 3, h, w)
    spec = load_pipeline_spec(ROOT / "recipes/depth/configs/pipelines/rgb_pose_direct_depth_v1.yaml")
    pose = torch.eye(4)[None]
    neighbors = pose[:, None].clone()
    neighbors[:, :, 0, 3] = baseline
    out = PipelineExecutor(spec).run({
        "rgb": image,
        "rgb_neighbors": image[:, None],
        "pose": pose,
        "pose_neighbors": neighbors,
        "intrinsics": torch.tensor([[[focal, 0, (w - 1) / 2], [0, focal, (h - 1) / 2], [0, 0, 1]]]),
    })

    assert out["d6.depth"].shape == (1, 1, h, w)
    assert out["evidence.grid_evidence"].shape[:3] == (1, 8, 8)
