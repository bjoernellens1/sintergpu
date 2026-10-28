# SPDX-License-Identifier: Apache-2.0
"""Local audited-source patch checks only; no build or Vulkan evidence."""
import hashlib
import os
from pathlib import Path
import subprocess
import tempfile
import pytest

LAB=Path('/dev/shm/sintergpu-rhi-receipt-lab')
HERE=Path(__file__).resolve().parent
CASES=[
 ('rhi','rhi-device-limit-observation-experimental.patch',[
  ('include/slang-rhi.h','861508fe63a9d496f299bd7d5f697e5a4bb3947fe7672182bfe016d6e36d18cb','5fdfd45c7bed85d4ab98a9116cc50782a55858c011daee6756f1cf0ba2518903'),
  ('src/vulkan/vk-device.cpp','5ce3c4d5dfd53a4b06cc14ee93a3232edde0f6504836927f6ca9803f6c67d813','52a57fc7b08bba8f06402a987da3e535b714785a22018e8a42dc17172baf222b')]),
 ('slangpy','slangpy-device-limit-observation-experimental.patch',[
  ('src/sgl/device/device.cpp','1fdd14494516445a6fcbb0212713b56e5ae09dda6956e64563e3a7b26a5066bb','020948db4622a3e16135791376215e39629241ba7b9d48fab0aff8f78ce9a04d'),
  ('src/slangpy_ext/device/device.cpp','0367c195fd70e11e32ec5136dc97616efdeaf2c7110978ff6e9aeef829bb8fd3','39917999f068d437a7b6714110458c3d95e7a9a532aab57e83d3bbd295d53721')])]

@pytest.mark.parametrize('source,patch,files',CASES)
def test_local_source_apply_is_exact(source,patch,files):
    if not (LAB/source).is_dir(): pytest.skip('local audit source unavailable; not CI proof')
    env=os.environ.copy()
    for name in ('GIT_DIR','GIT_WORK_TREE','GIT_INDEX_FILE'): env.pop(name,None)
    env['GIT_CEILING_DIRECTORIES']='/tmp'
    with tempfile.TemporaryDirectory(dir='/tmp',prefix='sinter-limit-patch-') as tmp:
        for name,expected,_ in files:
            data=(LAB/source/name).read_bytes()
            assert hashlib.sha256(data).hexdigest()==expected
            dest=Path(tmp)/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(data)
        subprocess.run(['git','apply','--check',str(HERE/patch)],cwd=tmp,env=env,check=True)
        subprocess.run(['git','apply',str(HERE/patch)],cwd=tmp,env=env,check=True)
        for name,_,expected in files:
            assert hashlib.sha256((Path(tmp)/name).read_bytes()).hexdigest()==expected
