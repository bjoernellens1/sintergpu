"""Contract model tests only: no real backend, ABI, or Vulkan evidence."""
import importlib.util
from pathlib import Path
import sys

import pytest

_path = Path(__file__).resolve().parents[1] / 'tools/diagnostics/rhi_receipt_model.py'
_spec = importlib.util.spec_from_file_location('rhi_receipt_model', _path)
_module = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _module
_spec.loader.exec_module(_module)
ReceiptModel = _module.ReceiptModel


def make(**changes):
    args = dict(buffer=11, memory=21, offset=0, size=12, allocation_size=16, ordinary=True)
    args.update(changes)
    return args


def test_lifetime_generation_and_device():
    model = ReceiptModel(bytes(range(16)), 42)
    first = model.register(**make())
    model.check(first)
    model.release(first)
    second = model.register(**make())
    assert second.generation != first.generation
    with pytest.raises(ValueError, match='stale'):
        model.check(first)
    model.close()
    with pytest.raises(ValueError, match='stale'):
        model.check(second)
    with pytest.raises(ValueError, match='closed'):
        model.register(**make())


@pytest.mark.parametrize('change', [dict(ordinary=False), dict(external=True), dict(sparse=True), dict(rebound=True), dict(buffer=0), dict(memory=0), dict(offset=1), dict(size=17), dict(size=0)])
def test_reject_unsupported_or_invalid(change):
    with pytest.raises(ValueError):
        ReceiptModel(bytes(range(16)), 42).register(**make(**change))


def test_reject_duplicate_and_alias():
    model = ReceiptModel(bytes(range(16)), 42)
    model.register(**make())
    for change in (dict(memory=22), dict(buffer=12)):
        with pytest.raises(ValueError, match='duplicate or aliased'):
            model.register(**make(**change))


@pytest.mark.parametrize('uuid,provider', [(bytes(16), 3), (bytes(15), 3), (bytes(range(16)), 0)])
def test_reject_unknown_device(uuid, provider):
    with pytest.raises(ValueError):
        ReceiptModel(uuid, provider)
