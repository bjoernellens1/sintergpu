import torch

from vkl.bench.trajectory_eval import evaluate_metric_trajectory


def test_metric_trajectory_evaluation_removes_only_first_pose_rigid_frame() -> None:
    estimate = torch.eye(4).repeat(3, 1, 1)
    estimate[:, 0, 3] = torch.tensor([0.0, 1.0, 2.0])
    ground_truth = estimate.clone()
    ground_truth[:, 1, 3] = 3.0

    out = evaluate_metric_trajectory(estimate, ground_truth)

    torch.testing.assert_close(out["translation_rmse_m"], torch.zeros(()))
    torch.testing.assert_close(out["translation_final_m"], torch.zeros(()))
    torch.testing.assert_close(out["rotation_mean_deg"], torch.zeros(()))


def test_metric_trajectory_evaluation_reports_drift_after_rigid_alignment() -> None:
    estimate = torch.eye(4).repeat(3, 1, 1)
    estimate[:, 0, 3] = torch.tensor([0.0, 1.0, 3.0])
    ground_truth = torch.eye(4).repeat(3, 1, 1)
    ground_truth[:, 0, 3] = torch.tensor([5.0, 6.0, 7.0])

    out = evaluate_metric_trajectory(estimate, ground_truth)

    torch.testing.assert_close(out["translation_final_m"], torch.tensor(1.0))
    assert out["translation_rmse_m"] > 0
