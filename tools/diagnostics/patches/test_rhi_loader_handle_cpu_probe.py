# SPDX-License-Identifier: Apache-2.0
"""Local CPU-only dynamic-loader experiment, not ICD/provider authentication."""
import os
from pathlib import Path
import resource
import shutil
import subprocess
import pytest

PROBE = Path('/dev/shm/sintergpu-allocation-correlation-lab/build/rhi_loader_handle_cpu_probe')

def compile_so(path, source, args=()):
    c = path.with_suffix('.c')
    c.write_text(source)
    subprocess.run(['cc', '-shared', '-fPIC', str(c), '-o', str(path), *args], check=True)

def test_actual_rhi_loader_handle_and_dependency_symbol(tmp_path):
    if not PROBE.is_file() or not shutil.which('cc'):
        pytest.skip('local isolated RHI build/compiler unavailable')
    direct = tmp_path / 'direct'; direct.mkdir()
    indirect = tmp_path / 'indirect'; indirect.mkdir()
    direct_loader = direct / 'libvulkan.so.1'
    indirect_loader = indirect / 'libvulkan.so.1'
    dep = indirect / 'libfakeprovider.so'
    compile_so(direct_loader, 'void vkGetInstanceProcAddr(void) {}\n')
    compile_so(dep, 'void vkGetInstanceProcAddr(void) {}\n')
    compile_so(indirect_loader, 'void fake_loader_anchor(void) {}\n',
               ['-L' + str(indirect), '-Wl,--no-as-needed', '-lfakeprovider', '-Wl,-rpath,$ORIGIN'])
    env = dict(os.environ, VK_DRIVER_FILES='/nonexistent/sintergpu-no-vulkan-icd.json',
               VK_ICD_FILENAMES='/nonexistent/sintergpu-no-vulkan-icd.json')
    def run(directory, expected_loader, expected_owner, relation):
        e = dict(env, LD_LIBRARY_PATH=str(directory))
        return subprocess.run([str(PROBE), str(expected_loader), str(expected_owner), relation],
                              env=e, capture_output=True, text=True,
                              preexec_fn=lambda: resource.setrlimit(resource.RLIMIT_CORE, (0, 0)))
    assert run(direct, direct_loader, direct_loader, 'same').returncode == 0
    assert run(indirect, indirect_loader, dep, 'different').returncode == 0
    # A loader handle's identity alone does not establish its symbol's owner.
    assert run(indirect, indirect_loader, indirect_loader, 'same').returncode != 0
    assert run(direct, indirect_loader, direct_loader, 'same').returncode != 0
