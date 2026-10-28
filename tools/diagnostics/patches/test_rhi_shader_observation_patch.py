# SPDX-License-Identifier: Apache-2.0
"""Local-only exact source patch application, no Vulkan execution."""
import hashlib
import os
from pathlib import Path
import subprocess
import tempfile
import pytest

LAB=Path('/dev/shm/sintergpu-rhi-receipt-lab')
HERE=Path(__file__).resolve().parent
CASES=[('rhi-limits','rhi-shader-byte-observation-experimental.patch',[
 ('include/slang-rhi.h','5fdfd45c7bed85d4ab98a9116cc50782a55858c011daee6756f1cf0ba2518903','1bd60ee9a893428e458a68c436b2b69fb14b2d02b0d0e65bf68f6d9dda9a1457'),
 ('src/vulkan/vk-shader-program.h','0838fe7f9dab01ba9f5760ee937c856b2c5a7ce9c8813bdccaa34548f5d75ae9','1cf12226ba236e730c8ccac6a0cf7a2e025cc9c9caae3fc00c92e9db40a8f130'),
 ('src/vulkan/vk-shader-program.cpp','c6f0c94979f05015033f74f4d3887285f427b440e8f60d474399db4ad4eb8832','bf854f78cc08305ba345a100c3b756ee2522e06c2791a6fb7107f667d2329bd4')]),
 ('slangpy-limits','slangpy-shader-byte-observation-experimental.patch',[
 ('src/slangpy_ext/device/shader.cpp','6a87423a770b9a5fbde764d60dad1b9b3530880240e2e8e9d6ac2464b824245f','30bdd602064d674e771c8c509e9cc296381f0531fdc30cf348bb3f13661b7189')])]

@pytest.mark.parametrize('source,patch,files',CASES)
def test_apply(source,patch,files):
    if not (LAB/source).is_dir(): pytest.skip('local audited source unavailable; not CI proof')
    env=os.environ.copy()
    for k in ('GIT_DIR','GIT_WORK_TREE','GIT_INDEX_FILE'): env.pop(k,None)
    env['GIT_CEILING_DIRECTORIES']='/tmp'
    with tempfile.TemporaryDirectory(dir='/tmp') as tmp:
        for name,sha,_ in files:
            raw=(LAB/source/name).read_bytes(); assert hashlib.sha256(raw).hexdigest()==sha
            p=Path(tmp)/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw)
        subprocess.run(['git','apply','--check',str(HERE/patch)],cwd=tmp,env=env,check=True)
        subprocess.run(['git','apply',str(HERE/patch)],cwd=tmp,env=env,check=True)
        for name,_,sha in files:
            raw=(Path(tmp)/name).read_bytes()
            assert hashlib.sha256(raw).hexdigest()==sha
