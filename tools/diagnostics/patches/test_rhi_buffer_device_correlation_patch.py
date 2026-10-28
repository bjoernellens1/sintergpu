# SPDX-License-Identifier: Apache-2.0
"""Local source-layer checks; skips without audited inputs and does not prove CI or GPU."""
import hashlib
import os
from pathlib import Path
import subprocess
import tempfile
import pytest
LAB=Path('/dev/shm/sintergpu-dispatch-observe-lab')
HERE=Path(__file__).resolve().parent
CASES = [('rhi-source', 'rhi-buffer-device-correlation-experimental.patch', [('include/slang-rhi.h', '96d11d39a837041c8793d1d82019f63f9cfddde73b8d7b7a7e5c4ec04e563600', '05ba08cef8ca63f4b0268046849dc56a4c46d07f0d4c8077b803bc6e51428dac'), ('src/vulkan/vk-device.h', 'f66664d01a31b014858e2c23e8759faf89af9210308bb4607659eda2abdeb566', '5f23dc7abf4e92af80e1c8a69031d9da2fb1565a6fc763c1800c5625619a5d4e'), ('src/vulkan/vk-buffer.h', '18f7c6603d0e9fb4fc217a66e4e94738df71dd5718b8e29858d03236eef7e381', '558dce714401bd462d341edcb197c5edfe258360e503c33ba88052889dd3ad0a'), ('src/vulkan/vk-buffer.cpp', 'bbb2e85eb9efab04675a7bbd04f681740a5c8979bf7ac3e1b7af9bf212aa787e', 'd25031a4bdc7b609dfc4e0bfae6980b44402579ed6d3577e0e5700caed89fb55')]), ('slangpy-source', 'slangpy-buffer-device-correlation-experimental.patch', [('src/slangpy_ext/device/resource.cpp', '571d212d21adf5361eeb76a80e46eb1872af78eab092e028de5bf6c0be225081', 'f0d0dfdfecd7c17b323e80b0f61c0fd890fbf667ab913a62ba92d2bfaf3e6d0a')])]
@pytest.mark.parametrize('source,patch,files',CASES)
def test_exact_layer(source,patch,files):
    if not (LAB/source).is_dir(): pytest.skip('local audited source unavailable')
    env=os.environ.copy()
    for key in ('GIT_DIR','GIT_WORK_TREE','GIT_INDEX_FILE'): env.pop(key,None)
    env['GIT_CEILING_DIRECTORIES']='/tmp'
    with tempfile.TemporaryDirectory(dir='/tmp') as tmp:
        for name,before,_ in files:
            data=(LAB/source/name).read_bytes()
            assert hashlib.sha256(data).hexdigest()==before
            target=Path(tmp)/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
        subprocess.run(['git','apply','--check',str(HERE/patch)],cwd=tmp,env=env,check=True)
        subprocess.run(['git','apply',str(HERE/patch)],cwd=tmp,env=env,check=True)
        for name,_,after in files:
            assert hashlib.sha256((Path(tmp)/name).read_bytes()).hexdigest()==after
