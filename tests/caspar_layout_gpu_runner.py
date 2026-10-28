# SPDX-License-Identifier: Apache-2.0
"""Fail-closed Caspar layout GPU runner scaffold.

The installed SlangPy API does not yet provide a reviewed proof that its
selected VkPhysicalDevice UUID was obtained through its own Vulkan loader.
`run` therefore refuses before device creation. Source inspection is not GPU
qualification and does not verify binary/compiler runtime equivalence.
"""
import argparse
import hashlib
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
CANDIDATE_COMMIT='019110e750540c21071f63375d68e578df1bf9d4'
FILES=('tests/caspar_layout_gpu_runner.py','tests/caspar_layout_gpu_fixture.py','tests/test_caspar_layout_gpu_fixture.py',
       'src/sintergpu/__init__.py','src/sintergpu/data/VERSION','src/sintergpu/caspar_layout_plan.py',
       'src/sintergpu/references/caspar_layout_numeric.py','src/sintergpu/kernels/experimental/caspar_layout.slang',
       'references/donors/caspar-slang/src/caspar_slang/layout.py')
SHADER='src/sintergpu/kernels/experimental/caspar_layout.slang'


def closure(commit):
    if not isinstance(commit,str) or len(commit)!=40 or any(c not in '0123456789abcdef' for c in commit): raise ValueError('full commit SHA required')
    head=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()
    if head!=commit: raise ValueError('checkout HEAD mismatch')
    result={}
    for name in FILES:
        path=ROOT/name
        if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(ROOT.resolve()): raise ValueError('missing, symlinked, or outside source: '+name)
        data=path.read_bytes()
        if data!=subprocess.check_output(['git','-C',str(ROOT),'show',commit+':'+name]): raise ValueError('source differs from commit: '+name)
        result[name]=hashlib.sha256(data).hexdigest()
    if (ROOT/SHADER).read_bytes()!=subprocess.check_output(['git','-C',str(ROOT),'show',CANDIDATE_COMMIT+':'+SHADER]):
        raise ValueError('shader differs from reviewed candidate')
    for name in ('sintergpu','sintergpu.caspar_layout_plan','sintergpu.references.caspar_layout_numeric','caspar_layout_gpu_fixture'):
        module=sys.modules.get(name)
        if module is None: continue
        path=Path(module.__file__).resolve()
        if not path.is_relative_to(ROOT.resolve()): raise ValueError('loaded module outside closure: '+name)
        relative=str(path.relative_to(ROOT.resolve()))
        if result.get(relative)!=hashlib.sha256(path.read_bytes()).hexdigest(): raise ValueError('loaded module differs from closure: '+name)
    return result


def run(args):
    if not args.execute: raise ValueError('explicit --execute required')
    closure(args.commit)
    # No environment variable, caller flag, GPU name or vulkaninfo snapshot can
    # establish that a SlangPy device's borrowed physical handle belongs to a
    # particular Vulkan loader/UUID. A reviewed direct identity bridge is needed.
    raise RuntimeError('selected SlangPy physical-device UUID is not independently verifiable; GPU execution refused')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--execute',action='store_true');p.add_argument('--commit',required=True)
    run(p.parse_args())

if __name__=='__main__': main()
