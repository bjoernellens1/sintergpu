# SPDX-License-Identifier: Apache-2.0
"""Local-only exact layered source checks; no GPU."""
import hashlib
import os
from pathlib import Path
import subprocess
import tempfile
import pytest

LAB=Path('/dev/shm/sintergpu-rhi-receipt-lab'); HERE=Path(__file__).resolve().parent
CASES=[('rhi-shader-observe','rhi-dispatch-association-experimental.patch',[
 ('include/slang-rhi.h','1bd60ee9a893428e458a68c436b2b69fb14b2d02b0d0e65bf68f6d9dda9a1457','96d11d39a837041c8793d1d82019f63f9cfddde73b8d7b7a7e5c4ec04e563600'),
 ('src/vulkan/vk-command.h','1d2694b9b79a3dda98f8c32be425c5bbdd298ca724bb23a80ca9942eb341889a','c0a54f6269091cc5242921f1ad0aca90616bbf9320bf4a0ac770e3b7e2703745'),
 ('src/vulkan/vk-command.cpp','8510182fac821f0bb750c54ff910c61d6815106500addd80728eef7ca5da11df','f91745314ea9dcdeb56f6263b71bb33988cf52a769d15251a4913d9fb0880da2'),
 ('src/vulkan/vk-pipeline.h','ac7890493ef8124c3c6022412740adf196c21c4ca937b0fb6afc09a989a47634','1005bc9a8398aeb3547c803b2e8eb9e45e2a87aab6fda97f655989b1778cbc49'),
 ('src/vulkan/vk-pipeline.cpp','b759c270f1f13dae433d4066c39459217e31b1730d0290ee0ec841d3b3b9205d','1d5ef0607c3d133211f117a0c84a7283f8116aab061601a4c0829c686a1509f1')]),
 ('slangpy-shader-observe','slangpy-dispatch-association-experimental.patch',[
 ('src/slangpy_ext/device/command.cpp','ae8ca0f5b2e2e38ab7aa85ff4678d9c25b2b3b6fd10434780f1657ff670b9e62','d134fac2c52040f063791af01cd9ae7cdd4ef2403ebfb4a73ac6f2c267c7148e')])]

@pytest.mark.parametrize('source,patch,files',CASES)
def test_exact_source_application(source,patch,files):
    if not (LAB/source).is_dir(): pytest.skip('local audited source unavailable; not CI proof')
    env=os.environ.copy()
    for k in ('GIT_DIR','GIT_WORK_TREE','GIT_INDEX_FILE'): env.pop(k,None)
    env['GIT_CEILING_DIRECTORIES']='/tmp'
    with tempfile.TemporaryDirectory(dir='/tmp') as tmp:
        for name,before,_ in files:
            data=(LAB/source/name).read_bytes();assert hashlib.sha256(data).hexdigest()==before
            dest=Path(tmp)/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(data)
        subprocess.run(['git','apply','--check',str(HERE/patch)],cwd=tmp,env=env,check=True)
        subprocess.run(['git','apply',str(HERE/patch)],cwd=tmp,env=env,check=True)
        for name,_,after in files:
            assert hashlib.sha256((Path(tmp)/name).read_bytes()).hexdigest()==after
