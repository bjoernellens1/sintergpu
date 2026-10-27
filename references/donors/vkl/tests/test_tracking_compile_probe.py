import importlib.util
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location("tracking_compile_probe", ROOT / "recipes/tracking/compile_tracking_transaction.py")
assert _SPEC and _SPEC.loader
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def test_tracking_compile_probe_transaction_is_cpu_callable():
    transaction = _MODULE.make_transaction(batch=1, height=6, width=8, device=torch.device("cpu"))

    out = transaction()

    assert out["delta"].shape == (1, 6)
    assert out["condition_proxy"].shape == (1,)
