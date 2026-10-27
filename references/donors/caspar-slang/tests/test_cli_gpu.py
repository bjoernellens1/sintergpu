import os
import subprocess
import sys

import pytest


@pytest.mark.skipif(os.environ.get("CASPAR_SLANG_GPU_TESTS") != "1", reason="GPU opt-in")
def test_validate_cli_reports_real_backend():
    proc = subprocess.run(
        [sys.executable, "-m", "caspar_slang.cli", "validate", "--size", "257"],
        text=True,
        capture_output=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "vulkan" in proc.stdout.lower()
    assert "PASS" in proc.stdout
