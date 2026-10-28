"""Independent stage outputs preserve the full frozen camera profile."""
from pathlib import Path

import numpy as np
import pytest

from benchmarks.camera_registration_fixture_history import load_canonical
from benchmarks.camera_registration_gpu_fixture import fixture_graph, DTYPES
from sintergpu.references.camera_registration_stage_numeric import run_stage

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / 'qualification/evidence/vkl-camera-registration-frozen-0.2.6-closure-v2'


def _prepared(graph, inputs):
    arrays = {}
    for name, spec in graph['resources'].items():
        arrays[name] = np.full(spec['elements'] + 13, 73, dtype=DTYPES[spec['dtype']])
        if name in inputs:
            arrays[name][:spec['elements']] = inputs[name].reshape(-1)
    return arrays


def test_frozen_composed_graphs_all_buffers_and_tails():
    corpus = load_canonical(ARCHIVE)
    assert len(corpus) == 80
    entries = set()
    for payload, expected in corpus:
        graph, inputs = fixture_graph(payload, max_workgroups_x=65535)
        buffers = _prepared(graph, inputs)
        before = {name: value.copy() for name, value in buffers.items() if name in inputs}
        for stage in graph['stages']:
            bindings = {name: buffers[resource] for name, resource in stage['bindings'].items()}
            before_stage = {name: value.copy() for name, value in buffers.items()}
            result = run_stage(stage['entrypoint'], stage['params'], bindings)
            assert all(result[k] is bindings[k] for k in result)
            written = {stage['bindings'][key] for key in result}
            for name, old in before_stage.items():
                spec = graph['resources'][name]
                if name not in written:
                    np.testing.assert_array_equal(buffers[name].view(np.uint8), old.view(np.uint8), err_msg=stage['entrypoint']+' '+name)
                np.testing.assert_array_equal(buffers[name][spec['elements']:], old[spec['elements']:], err_msg=stage['entrypoint']+' '+name+' tail')
            entries.add(stage['entrypoint'])
        for name, value in before.items():
            np.testing.assert_array_equal(buffers[name].view(np.uint8), value.view(np.uint8), err_msg=name)
        for name, spec in graph['resources'].items():
            np.testing.assert_array_equal(buffers[name][spec['elements']:], 73, err_msg=name)
        for key, expected_value in expected.items():
            output = buffers[graph['outputs']['depth_rays' if key == 'result' else key]][:expected_value.size].reshape(expected_value.shape)
            if expected_value.dtype == np.float32:
                np.testing.assert_allclose(output, expected_value, rtol=8e-6, atol=8e-7, equal_nan=True, err_msg=payload['case_id']+' '+key)
            else:
                np.testing.assert_array_equal(output, expected_value, err_msg=payload['case_id']+' '+key)
    assert entries == {'rectifyGrid','rectifyFromGrid','depthCameraRays','clearRegistration', 'projectRegistration','registrationMin','registrationWinnerCount','finalizeRegistration','finalizeRegistrationDebug'}


def test_collision_atomic_inout_and_replay():
    p = dict(batches=1, points=5, targetHeight=1, targetWidth=2)
    b = dict(minDepthBits=np.array([0,1,73],np.uint32), winnerIndex=np.array([0,0,73],np.uint32), collisionCount=np.array([4,4,73],np.uint32))
    run_stage('clearRegistration', p, b)
    np.testing.assert_array_equal(b['minDepthBits'], [0x7f800000,0x7f800000,73])
    np.testing.assert_array_equal(b['winnerIndex'], [5,5,73])
    np.testing.assert_array_equal(b['collisionCount'], [0,0,73])
    b.update(projectedZOut=np.array([2,1,1,1,1,73],np.float32), sourceValidOut=np.array([1,1,1,0,1,73],np.uint32), destinationIndexOut=np.array([0,0,0,1,1,73],np.uint32))
    run_stage('registrationMin', p, b)
    run_stage('registrationWinnerCount', p, b)
    np.testing.assert_array_equal(b['winnerIndex'], [1,4,73])
    np.testing.assert_array_equal(b['collisionCount'], [3,1,73])
    run_stage('registrationWinnerCount', p, b)
    np.testing.assert_array_equal(b['collisionCount'], [6,2,73])
    run_stage('clearRegistration', p, b)
    np.testing.assert_array_equal(b['collisionCount'], [0,0,73])


def test_debug_finalizer_word_representation_and_invalid_values():
    p = dict(batches=1, targetHeight=1, targetWidth=4)
    bits = np.array([1., np.inf, np.nan, 2.],np.float32).view(np.uint32)
    b = dict(minDepthBits=np.r_[bits,np.uint32(73)], winnerIndex=np.array([3,4,5,0xffffffff,73],np.uint32),
             collisionCount=np.array([2,7,9,0xffffffff,73],np.uint32),
             depthOut=np.full(5,73,np.float32),validOut=np.full(5,73,np.uint32),
             winnerSourceIndexOut=np.full(5,73,np.int64),collisionCountOut=np.full(5,73,np.int64))
    result = run_stage('finalizeRegistrationDebug',p,b)
    assert set(result) == {'depthOut','validOut','winnerSourceIndexOut','collisionCountOut'}
    np.testing.assert_array_equal(b['depthOut'],[1,0,0,2,73])
    np.testing.assert_array_equal(b['validOut'],[1,0,0,1,73])
    np.testing.assert_array_equal(b['winnerSourceIndexOut'],[3,-1,-1,0xffffffff,73])
    np.testing.assert_array_equal(b['collisionCountOut'],[2,7,9,0xffffffff,73])
    np.testing.assert_array_equal(b['winnerSourceIndexOut'].view(np.uint32)[:8], [3,0,0xffffffff,0xffffffff,0xffffffff,0xffffffff,0xffffffff,0])


def test_min_and_winner_preserve_existing_inout_state():
    p = dict(batches=1,points=3,targetHeight=1,targetWidth=1)
    b = dict(projectedZOut=np.array([2,3,1,73],np.float32),sourceValidOut=np.array([1,1,0,73],np.uint32),
             destinationIndexOut=np.array([0,0,0,73],np.uint32),
             minDepthBits=np.array([np.float32(1).view(np.uint32),73],np.uint32),
             winnerIndex=np.array([2,73],np.uint32),collisionCount=np.array([4,73],np.uint32))
    run_stage('registrationMin',p,b)
    run_stage('registrationWinnerCount',p,b)
    np.testing.assert_array_equal(b['minDepthBits'],[np.float32(1).view(np.uint32),73])
    np.testing.assert_array_equal(b['winnerIndex'],[2,73])
    np.testing.assert_array_equal(b['collisionCount'],[6,73])
