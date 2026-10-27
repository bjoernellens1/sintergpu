import numpy as np
import pytest

from caspar_slang.gn import _validate, reference, synthetic_problem


def test_reference_jacobian_matches_finite_difference():
    p, q, normal, indices = synthetic_problem(19)
    p = p[indices].astype(np.float64)
    normal = normal.astype(np.float64)
    expected = reference(*synthetic_problem(19))["jacobian"]
    eps = 1e-6
    columns = []
    for i in range(6):
        xi = np.zeros(6)
        xi[i] = eps

        def residual(sign):
            transformed = p + sign * (np.cross(xi[:3], p) + xi[3:])
            return np.einsum("ij,ij->i", normal, transformed - q)

        columns.append((residual(1) - residual(-1)) / (2 * eps))
    np.testing.assert_allclose(np.array(columns).T, expected, atol=1e-9)


@pytest.mark.parametrize("damping", [-1, np.nan, np.inf])
def test_rejects_invalid_damping(damping):
    with pytest.raises(ValueError, match="damping"):
        _validate(*synthetic_problem(2), damping)


def test_rejects_out_of_bounds_and_empty_correspondences():
    p, q, n, idx = synthetic_problem(2)
    with pytest.raises(ValueError, match="bounds"):
        _validate(p, q, n, np.array([0, len(p)]), 0)
    with pytest.raises(ValueError, match="nonempty"):
        _validate(p, q[:0], n[:0], np.zeros(0, np.uint32), 0)


def test_reference_singular_failure_and_damped_recovery():
    zeros = np.zeros((1, 3), np.float32)
    indices = np.zeros(1, np.uint32)
    assert reference(zeros, zeros, zeros, indices, damping=0)["status"] == 1
    assert reference(zeros, zeros, zeros, indices, damping=1e-3)["status"] == 0


@pytest.mark.skipif(
    __import__("os").environ.get("CASPAR_SLANG_GPU_TESTS") != "1",
    reason="set CASPAR_SLANG_GPU_TESTS=1",
)
@pytest.mark.parametrize("reduction", ["workgroup", "subgroup"])
@pytest.mark.parametrize("n", [1, 65, 4099])
def test_gpu_variants_and_resident_repeated_solve(n, reduction):
    from caspar_slang.gn import GaussNewton

    inputs = synthetic_problem(n, seed=17)
    solver = GaussNewton(reduction=reduction)
    problem = solver.prepare(*inputs)
    for damping in (1e-3, 0.1):
        solver.dispatch(problem, damping)
        result = solver.readback(problem)
        expected = reference(*inputs, damping=damping)
        for name in ("residual", "jacobian", "hessian", "gradient", "delta"):
            np.testing.assert_allclose(getattr(result, name), expected[name], atol=3e-5, rtol=3e-4)
        assert result.status == 0
        np.testing.assert_allclose(result.cost, expected["cost"], atol=3e-5, rtol=3e-4)


def test_reference_accepts_tuple_indices_like_gpu_preparation():
    points, targets, normals, _ = synthetic_problem(2)
    expected = reference(points, targets, normals, np.array([0, 1]))
    actual = reference(points, targets, normals, (0, 1))
    np.testing.assert_array_equal(actual["jacobian"], expected["jacobian"])
