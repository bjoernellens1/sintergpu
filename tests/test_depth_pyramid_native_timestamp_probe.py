# SPDX-License-Identifier: Apache-2.0
"""Compile-only native Vulkan probe checks. Never opens a device."""
from pathlib import Path
import shutil
import subprocess
import pytest
ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'benchmarks/native/vulkan_timestamp_probe.c'


def test_native_probe_exposes_identity_and_raw_timestamp_contract():
    assert SOURCE.is_file(), 'native Vulkan timestamp source missing'
    source=SOURCE.read_text()
    for token in ('VkPhysicalDeviceIDProperties','timestampPeriod','timestampValidBits','vkGetQueryPoolResults','VK_QUERY_RESULT_64_BIT','VK_QUERY_RESULT_WAIT_BIT'):
        assert token in source
    assert 'VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT' in source


def test_native_probe_compiles_with_strict_warnings(tmp_path):
    cc=shutil.which('cc')
    if not cc: pytest.skip('C compiler unavailable')
    result=subprocess.run([cc,'-std=c11','-O2','-Wall','-Wextra','-Werror',str(SOURCE),'-lvulkan','-o',str(tmp_path/'probe')],capture_output=True,text=True)
    assert result.returncode==0,result.stdout+result.stderr
