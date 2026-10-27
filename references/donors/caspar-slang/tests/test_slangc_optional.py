from pathlib import Path

import pytest

from caspar_slang.compiler import compile_slang, find_slangc
from caspar_slang.demo import demo_kernel
from caspar_slang.emitter import emit_slang


@pytest.mark.skipif(find_slangc() is None, reason="slangc not installed")
def test_demo_compiles_to_spirv(tmp_path: Path):
    src = tmp_path / "demo.slang"
    spv = tmp_path / "demo.spv"
    src.write_text(emit_slang(demo_kernel()))
    compile_slang(src, spv, target="spirv")
    assert spv.exists()
    assert spv.stat().st_size > 0
