"""Opt-in complete D4/T0 parity against NumPy and mandatory portable donors.

The canonical archive is produced explicitly with one pinned CPU Torch build.
GPU qualification never imports Torch or regenerates expected donor outputs.
"""
import importlib
import os
from pathlib import Path

import numpy as np
import pytest

from benchmarks.reprojection_fixture_archive import load_fixture
from benchmarks.reprojection_fixtures import payloads

pytestmark = pytest.mark.skipif(os.getenv('SINTERGPU_GPU_TESTS') != '1', reason='explicit GPU qualification only')

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARCHIVE = ROOT / 'qualification/evidence/vkl-reprojection-frozen-0.2.5'


def compare(actual, expected):
    assert actual.keys() == expected.keys()
    for key, array in expected.items():
        assert actual[key].shape == array.shape
        if array.dtype.kind in 'biu':
            np.testing.assert_array_equal(actual[key], array, err_msg=key)
        else:
            # Identical strict thresholds for NumPy and portable frozen donors.
            np.testing.assert_allclose(actual[key], array, rtol=8e-6, atol=8e-7, equal_nan=True, err_msg=key)


def _case_groups():
    groups = {}
    for payload in payloads():
        groups.setdefault(payload['case_id'], []).append(payload)
    assert len(groups) == 63
    assert sum(map(len, groups.values())) == 113
    return groups


CASE_GROUPS = _case_groups()


@pytest.fixture(scope='module')
def canonical_outputs():
    """Validate every canonical state before the device fixture can allocate."""
    directory = Path(os.environ.get('SINTERGPU_FROZEN_REFERENCE_ARCHIVE', DEFAULT_ARCHIVE))
    return {case_id: [load_fixture(payload, directory) for payload in states]
            for case_id, states in CASE_GROUPS.items()}


@pytest.fixture(scope='module')
def device(canonical_outputs):
    import slangpy as spy
    owned_device = spy.Device(type=spy.DeviceType.vulkan, compiler_options={'floating_point_mode': spy.SlangFloatingPointMode.precise})
    try:
        yield owned_device
    finally:
        owned_device.close()


@pytest.mark.parametrize('case_id', CASE_GROUPS)
def test_full_outputs_and_reused_scratch(device, canonical_outputs, case_id):
    from benchmarks.reprojection_support import PreparedGraph

    states = CASE_GROUPS[case_id]
    donors = canonical_outputs[case_id]
    prepared = PreparedGraph(device, states[0]['graph'], states[0]['inputs'])
    for payload, donor in zip(states, donors):
        assert payload['graph'] == states[0]['graph']
        if payload['iteration']:
            for key, value in payload['inputs'].items():
                prepared.upload(key, value)
        module = ('reprojection_numeric' if payload['oracle_module'] == 'reproject'
                  else 'temporal_reprojection_numeric')
        oracle = getattr(importlib.import_module('sintergpu.references.' + module), payload['oracle_function'])
        expected = oracle(**payload['oracle_kwargs'])
        encoder = device.create_command_encoder()
        prepared.record(encoder)
        device.submit_command_buffer(encoder.finish())
        device.wait_for_idle()
        actual = prepared.outputs(expected)
        compare(actual, expected)
        compare(actual, donor)
        prepared.assert_poison_tails()
