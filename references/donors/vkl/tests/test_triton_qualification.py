import pytest
import torch

from vkl.bench.triton_qualification import (
    KernelCase,
    _stats,
    compare_outputs,
    measure_case,
    serialize_report,
)


def _case():
    def factory(device, batch=1):
        return (torch.arange(batch, device=device, dtype=torch.float32),), {}

    def reference(values):
        return {"value": values + 1}

    return KernelCase(
        name="toy",
        family="unit",
        factory=factory,
        reference=reference,
        triton=None,
        parity=None,
        workload={"batch": 2, "seed": 7},
    )


def test_measure_case_reports_parity_and_unsupported_triton():
    report = measure_case(_case(), ("eager", "triton"), device="cpu", warmup=1, iterations=2)

    assert report["name"] == "toy"
    assert report["variants"]["eager"]["status"] == "ok"
    assert report["variants"]["triton"]["status"] == "unsupported"
    assert report["variants"]["triton"]["reason"]
    assert report["variants"]["eager"]["parity"]["maximum_error"] == 0.0
    assert report["variants"]["eager"]["parity"]["mismatch_count"] == 0
    assert report["decision"] != "accepted"


def test_measure_case_serializes_compilation_failures():
    case = _case()
    report = measure_case(
        case,
        {"eager": case.reference, "compiled": lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("compile failed"))},
        device="cpu",
        warmup=0,
        iterations=1,
    )

    assert report["variants"]["compiled"]["status"] == "error"
    assert "compile failed" in report["variants"]["compiled"]["error"]


def test_compiled_variant_failure_is_not_reported_as_compiled(monkeypatch):
    case = _case()

    def compile_reference(_reference):
        return lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("lowering failed"))

    monkeypatch.setattr(torch, "compile", compile_reference)
    report = measure_case(case, ("compiled",), device="cpu", warmup=0, iterations=1)

    compiled = report["variants"]["compiled"]
    assert compiled["status"] == "error"
    assert compiled["actual_implementation"] == "error"


def test_measure_case_reports_numerical_mismatch():
    case = _case()
    report = measure_case(
        case,
        {"eager": case.reference, "triton": lambda values: {"value": values + 1.25}},
        device="cpu",
        warmup=0,
        iterations=1,
    )

    parity = report["variants"]["triton"]["parity"]
    assert parity["maximum_error"] == 0.25
    assert parity["mismatch_count"] == 2
    assert report["decision"] != "accepted"


def test_measure_case_isolates_variants_and_rejects_input_mutation():
    def factory(device, batch=2):
        values = torch.arange(batch, device=device, dtype=torch.float32)
        scratch = torch.zeros(batch, device=device)
        return (values, scratch), {}

    def reference(values, _scratch):
        return {"value": values + 1}

    def mutating_candidate(values, scratch):
        scratch.add_(1)
        mutating_candidate.dispatch_metadata = {"requested": "triton", "actual": "triton"}
        return {"value": values + 1}

    case = KernelCase(
        name="mutation",
        family="unit",
        factory=factory,
        reference=reference,
        triton=mutating_candidate,
        parity=None,
        workload={"batch": 2},
    )
    report = measure_case(
        case,
        {"triton": mutating_candidate, "eager": reference},
        device="cpu",
        warmup=1,
        iterations=2,
    )

    triton = report["variants"]["triton"]
    assert triton["parity"]["exact"] is True
    assert triton["status"] == "contract_error"
    assert triton["input_state"]["exact"] is False
    assert triton["reason"] == "implementation mutated benchmark inputs"
    assert report["variants"]["eager"]["status"] == "ok"
    assert report["variants"]["eager"]["input_state"]["exact"] is True
    assert report["decision"] != "candidate"


def test_measure_case_rejects_mutating_reference_contract():
    def factory(device, batch=2):
        return (torch.arange(batch, device=device, dtype=torch.float32),), {}

    def mutating_reference(values):
        values.add_(1)
        return {"value": values}

    case = KernelCase(
        name="mutating-reference",
        family="unit",
        factory=factory,
        reference=mutating_reference,
        triton=None,
        parity=None,
        workload={"batch": 2},
    )
    report = measure_case(case, {"eager": mutating_reference}, device="cpu", warmup=0, iterations=1)

    assert report["variants"]["reference"]["status"] == "error"
    assert report["variants"]["reference"]["error"] == "reference mutated benchmark inputs"
    assert report["variants"]["eager"]["status"] == "contract_error"
    assert report["decision"] == "unsupported"


def test_factory_internal_typeerror_is_not_misread_as_one_argument_factory():
    def broken_factory(_device, _batch):
        raise TypeError("factory implementation bug")

    case = KernelCase(**{**_case().__dict__, "factory": broken_factory})
    with pytest.raises(TypeError, match="factory implementation bug"):
        measure_case(case, ("eager",), device="cpu", warmup=0, iterations=1)


def test_one_argument_factory_remains_supported():
    def one_argument_factory(device):
        return (torch.ones(2, device=device),), {}

    case = KernelCase(**{**_case().__dict__, "factory": one_argument_factory})
    report = measure_case(case, ("eager",), device="cpu", warmup=0, iterations=1)
    assert report["variants"]["eager"]["status"] == "ok"


def test_measure_case_measures_every_declared_batch(monkeypatch):
    case = _case()
    case = KernelCase(**{**case.__dict__, "workload": {"batches": (1, 2, 4, 8)}})
    monkeypatch.setattr(torch, "compile", lambda _reference: (_ for _ in ()).throw(RuntimeError("compile failed")))

    report = measure_case(case, device="cpu", warmup=0, iterations=1)

    assert set(report["batches"]) == {"1", "2", "4", "8"}
    assert [report["batches"][str(batch)]["batch"] for batch in (1, 2, 4, 8)] == [1, 2, 4, 8]
    for batch in (1, 2, 4, 8):
        variants = report["batches"][str(batch)]["variants"]
        assert variants["eager"]["status"] == "ok"
        assert variants["compiled"]["status"] == "error"
        assert variants["triton"]["status"] == "unsupported"


def test_measure_case_repeats_each_timed_sample_without_changing_outputs():
    calls = 0

    def reference(values):
        nonlocal calls
        calls += 1
        return {"value": values + 1}

    case = KernelCase(**{**_case().__dict__, "reference": reference})
    report = measure_case(
        case,
        {"eager": reference},
        device="cpu",
        warmup=2,
        iterations=3,
        calls_per_sample=4,
    )

    assert calls == 1 + (2 + 3) * 4
    assert report["calls_per_sample"] == 4
    assert report["timing_contract"] == {
        "layer": "python_wall_amortized",
        "canonical_device_kernel_latency": False,
        "units": "ms_per_call",
        "calls_per_sample": 4,
        "warmup_samples": 2,
        "warmup_calls": 8,
        "timed_samples": 3,
        "timed_calls": 12,
    }
    timing = report["variants"]["eager"]["timing"]
    assert {"min_ms", "p05_ms", "p50_ms", "p95_ms", "max_ms", "mad_ms"} <= set(timing)
    assert report["variants"]["eager"]["parity"]["exact"] is True
    assert report["variants"]["eager"]["input_state"]["exact"] is True


def test_stats_uses_true_even_sample_median_and_mad():
    stats = _stats([1.0, 2.0, 100.0, 101.0])
    assert stats["p50_ms"] == 51.0
    assert stats["mad_ms"] == 49.5


def test_compare_outputs_accepts_matching_signed_infinities_and_nan():
    expected = torch.tensor([float("inf"), float("-inf"), float("nan")])
    same = compare_outputs(expected, expected.clone())
    assert same == {"maximum_error": 0.0, "mismatch_count": 0, "exact": True}

    actual = torch.tensor([float("-inf"), float("-inf"), float("nan")])
    mismatch = compare_outputs(expected, actual)
    assert mismatch["maximum_error"] == float("inf")
    assert mismatch["mismatch_count"] == 1
    assert mismatch["exact"] is False


def test_serialize_report_creates_parent_directory(tmp_path):
    destination = tmp_path / "nested" / "report.json"
    serialize_report({"decision": "retain_compile"}, str(destination))

    assert destination.read_text() == '{\n  "decision": "retain_compile"\n}\n'


def test_serialize_report_emits_strict_json_for_nonfinite_values(tmp_path):
    destination = tmp_path / "strict.json"
    serialize_report({"error": float("inf"), "missing": float("nan")}, str(destination))

    text = destination.read_text()
    assert "Infinity" not in text
    assert "NaN" not in text


def test_compiled_variant_reports_requested_and_actual_implementation(monkeypatch):
    case = _case()

    def compile_reference(reference):
        return lambda *args, **kwargs: reference(*args, **kwargs)

    monkeypatch.setattr(torch, "compile", compile_reference)
    report = measure_case(case, ("compiled",), device="cpu", warmup=0, iterations=1)

    compiled = report["variants"]["compiled"]
    assert compiled["requested_implementation"] == "compiled"
    assert compiled["actual_implementation"] == "compiled_unverified"


def test_eager_fallback_with_exact_parity_is_not_a_triton_candidate():
    case = _case()

    def fallback(values):
        fallback.dispatch_metadata = {"requested": "triton", "actual": "eager_fallback"}
        return {"value": values + 1}

    case = KernelCase(**{**case.__dict__, "triton": fallback})
    report = measure_case(case, ("eager", "triton"), device="cpu", warmup=0, iterations=1)

    assert report["variants"]["triton"]["parity"]["exact"] is True
    assert report["variants"]["triton"]["actual_implementation"] == "eager_fallback"
    assert report["decision"] != "candidate"
