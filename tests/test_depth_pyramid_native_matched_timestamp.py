# SPDX-License-Identifier: Apache-2.0
"""Native matched stage source and metadata tests; no Vulkan device execution."""
import shutil
import subprocess
import pytest
from pathlib import Path
from benchmarks import depth_pyramid_native_matched_runner as runner
ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'benchmarks/native/vulkan_timestamp_matched.c'


def test_matched_source_and_runner_contract():
    assert SOURCE.is_file()
    text=SOURCE.read_text()
    assert 'VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT' in text and 'VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT' in text
    assert 'timestampValidBits' in text and 'timestampPeriod' in text
    assert 'vkGetQueryPoolResults' in text and 'VK_QUERY_RESULT_64_BIT' in text
    assert 'expected_family' in text
    assert str(SOURCE.relative_to(ROOT)) in runner.EXTRA


def test_strict_native_compile(tmp_path):
    cc=shutil.which('cc')
    if not cc: pytest.skip('no C compiler')
    result=subprocess.run([cc,'-std=c11','-O2','-Wall','-Wextra','-Werror',str(SOURCE),'-lvulkan','-o',str(tmp_path/'probe')],capture_output=True,text=True)
    assert result.returncode==0,result.stdout+result.stderr


def test_native_matched_metadata_rejects_bad_family_or_pairs():
    obj={'device_uuid':'uuid','vendor_id':4318,'device_id':42,'driver_version':123,'timestamp_period_ns':1.,'timestamp_valid_bits':64,'queue_family':3,'top_pairs':[[1,2]]*50,'bottom_pairs':[[3,4]]*50}
    runner.validate_result(obj,'uuid',4318,42,123,3,50)
    obj['queue_family']=4
    with pytest.raises(ValueError): runner.validate_result(obj,'uuid',4318,42,123,3,50)
    obj['queue_family']=3; obj['bottom_pairs']=[]
    with pytest.raises(ValueError): runner.validate_result(obj,'uuid',4318,42,123,3,50)


def test_four_ordered_queries_and_raw_output_without_wrap_assumptions():
    text=SOURCE.read_text()
    calls=[line.strip() for line in text.splitlines() if 'vkCmdWriteTimestamp(' in line]
    assert calls == [
        'vkCmdWriteTimestamp(cmd,VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,queries,4*i);',
        'vkCmdWriteTimestamp(cmd,VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,queries,4*i+1);',
        'vkCmdWriteTimestamp(cmd,VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,queries,4*i+2);',
        'vkCmdWriteTimestamp(cmd,VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,queries,4*i+3);']
    obj={'device_uuid':'uuid','vendor_id':4318,'device_id':42,'driver_version':123,'timestamp_period_ns':1.,'timestamp_valid_bits':8,'queue_family':3,'top_pairs':[[255,0]]*50,'bottom_pairs':[[0,1]]*50}
    runner.validate_result(obj,'uuid',4318,42,123,3,50)


def test_compiler_dependencies_include_matched_source():
    cc=shutil.which('cc')
    if not cc: pytest.skip('no C compiler')
    dependencies=runner.dependency_hashes(Path(cc).resolve())
    assert str(SOURCE.resolve()) in dependencies
    assert any(path.endswith('/vulkan/vulkan.h') for path in dependencies)
