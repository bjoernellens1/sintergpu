import importlib.util
import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.mark.skipif(
    shutil.which("slangc") is None or importlib.util.find_spec("symforce") is None,
    reason="requires slangc and SymForce",
)
def test_generated_gn_compiles_to_valid_spirv(tmp_path):
    from caspar_slang.compiler import compile_slang
    from caspar_slang.emitter import emit_slang
    from caspar_slang.gn import residual_kernel

    source = tmp_path / "gn.slang"
    source.write_text(emit_slang(residual_kernel()))
    output = tmp_path / "gn.spv"
    compile_slang(source, output)
    assert output.read_bytes()[:4] == b"\x03\x02\x23\x07"
    if shutil.which("spirv-val"):
        subprocess.run(["spirv-val", str(output)], check=True, capture_output=True, text=True)


@pytest.mark.skipif(shutil.which("slangc") is None, reason="requires slangc")
@pytest.mark.parametrize(
    "entry", ["reduceResidual", "reduceResidualSubgroup", "reducePartials", "solveSystem"]
)
def test_reduction_and_solve_spirv(tmp_path, entry):
    from caspar_slang.compiler import compile_slang

    source = Path(__file__).parents[1] / "src" / "caspar_slang" / "shaders" / "gn.slang"
    output = tmp_path / (entry + ".spv")
    compile_slang(source, output, entry=entry)
    assert output.read_bytes()[:4] == b"\x03\x02\x23\x07"
    if shutil.which("spirv-val"):
        subprocess.run(["spirv-val", str(output)], check=True, capture_output=True, text=True)
