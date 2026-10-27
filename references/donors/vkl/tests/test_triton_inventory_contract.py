import importlib.util
from pathlib import Path

import torch

from vkl.bench.triton_qualification import measure_case
from vkl.kernels import reference

_MODULE_PATH = Path(__file__).parents[1] / "recipes" / "triton" / "benchmark_inventory.py"
_SPEC = importlib.util.spec_from_file_location("benchmark_inventory", _MODULE_PATH)
_MODULE = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(_MODULE)
inventory = _MODULE.inventory
build_report_metadata = _MODULE.build_report_metadata
classifications = _MODULE.classifications


def test_inventory_matches_reference_exports_and_batch_matrix():
    cases = inventory()
    assert {case.name for case in cases} >= set(reference.__all__)
    for case in cases:
        assert case.family
        assert callable(case.factory)
        assert callable(case.reference)
        assert case.parity is not None
        assert case.workload["image_hw"] == (8, 8)
        assert case.workload["algorithm_dimensions"]
        assert case.workload["algorithm_dimensions"]["operation"] == case.name
        assert case.workload["algorithm_dimensions"]["control"]
        assert set(case.workload["batches"]) == {1, 2, 4, 8}


def test_every_public_kernel_has_a_specific_triton_triage_reason():
    decisions = classifications()

    assert set(decisions) == set(reference.__all__)
    for name, decision in decisions.items():
        assert decision["family"], name
        assert decision["disposition"] in {"candidate", "deferred", "retain_compile", "retain_vendor"}, name
        assert decision["reason"], name


def test_inventory_controls_are_valid_and_deterministic_on_cpu():
    for case in inventory():
        args, kwargs = case.factory(torch.device("cpu"), 1)
        output = case.reference(*args, **kwargs)
        assert output is not None, case.name
        args_again, kwargs_again = case.factory(torch.device("cpu"), 1)
        for first, second in zip(args, args_again):
            if isinstance(first, torch.Tensor):
                assert torch.equal(first, second), case.name
        assert kwargs.keys() == kwargs_again.keys()


def test_inventory_controls_execute_at_required_batches():
    for case in inventory():
        report = measure_case(case, ("eager",), device="cpu", warmup=0, iterations=1)
        for batch in case.workload["batches"]:
            entry = report["batches"][str(batch)]
            assert entry["batch"] == batch, case.name
            assert entry["variants"]["eager"]["status"] == "ok", case.name
            assert entry["variants"]["eager"]["timing"]["p50_ms"] >= 0, case.name


def test_cpu_argmin_fallback_is_not_reported_as_triton_candidate():
    case = next(case for case in inventory() if case.name == "select_best_inverse_depth")
    assert case.reference is not case.triton
    args, kwargs = case.factory(torch.device("cpu"), 1)
    assert torch.unique(args[0]).numel() == args[0].numel()
    _expected, indices = args[0].topk(2, dim=-1, largest=False)
    assert torch.equal(case.reference(*args, **kwargs)["best_index"], indices[..., 0])
    report = measure_case(case, ("eager", "triton"), device="cpu", warmup=0, iterations=1)

    assert report["variants"]["triton"]["status"] == "unsupported"
    assert report["variants"]["triton"]["dispatch"]["actual"] == "unsupported"
    assert report["decision"] != "candidate"
    assert report["decision"] != "accepted"


def test_inventory_report_metadata_carries_provenance_and_dimensions(monkeypatch):
    monkeypatch.setenv("VKL_SOURCE_REVISION", "source-sha")
    monkeypatch.setenv("VKL_IMAGE_REVISION", "image-sha")
    monkeypatch.setenv("VKL_IMAGE_REFERENCE", "localhost/vkl:test")

    metadata = build_report_metadata(torch.device("cpu"), (1, 2, 4, 8), "inventory command")

    assert metadata["source_revision"] == "source-sha"
    assert metadata["image_revision"] == "image-sha"
    assert metadata["image_reference"] == "localhost/vkl:test"
    assert metadata["torch_version"]
    assert metadata["triton_version"]
    assert metadata["backend"]["device_name"] == metadata["device_name"]
    assert metadata["command"] == "inventory command"
    assert metadata["algorithm_dimensions"]["batches"] == [1, 2, 4, 8]
