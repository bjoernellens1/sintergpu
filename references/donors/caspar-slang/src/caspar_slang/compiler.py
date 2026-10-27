from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

TARGET_ARGS = {
    "spirv": ["-target", "spirv", "-profile", "glsl_450"],
    "cuda": ["-target", "cuda"],
    "ptx": ["-target", "ptx"],
    "metal": ["-target", "metal"],
    "cpp": ["-target", "cpp"],
}


def find_slangc() -> str | None:
    return shutil.which("slangc")


def compile_slang(
    source: str | Path,
    output: str | Path,
    *,
    target: str = "spirv",
    entry: str = "computeMain",
) -> subprocess.CompletedProcess[str]:
    slangc = find_slangc()
    if slangc is None:
        raise RuntimeError(
            "slangc was not found. Install Slang (or a Vulkan SDK that includes it) "
            "and ensure slangc is on PATH."
        )
    if target not in TARGET_ARGS:
        raise ValueError(f"unsupported target {target!r}; choose from {sorted(TARGET_ARGS)}")

    cmd = [
        slangc,
        str(source),
        "-entry",
        entry,
        "-stage",
        "compute",
        *TARGET_ARGS[target],
        "-o",
        str(output),
    ]
    return subprocess.run(cmd, check=True, text=True, capture_output=True)
