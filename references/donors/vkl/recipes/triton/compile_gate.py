"""Run one compiler qualification command with bounded process ownership."""

from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
import signal
import subprocess
import tempfile
import time
from pathlib import Path


def _acquire_lock(stream, timeout_s: float) -> float | None:
    start = time.monotonic()
    while True:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            return time.monotonic() - start
        except BlockingIOError:
            if time.monotonic() - start >= timeout_s:
                return None
            time.sleep(min(0.05, timeout_s))


def _terminate_group(process: subprocess.Popen[str]) -> tuple[str, str, bool]:
    terminated = False
    if process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGTERM)
            terminated = True
        except ProcessLookupError:
            pass
    try:
        stdout, stderr = process.communicate(timeout=2.0)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
            terminated = True
        except ProcessLookupError:
            pass
        stdout, stderr = process.communicate()
    return stdout, stderr, terminated


def _has_files(path: Path) -> bool:
    return path.exists() and any(entry.is_file() for entry in path.rglob("*"))


def _has_compiler_cache(root: Path) -> bool:
    return _has_files(root / "torchinductor") or _has_files(root / "triton")


def _prepare_cache(
    state: str,
    requested: str | None,
) -> tuple[tempfile.TemporaryDirectory[str] | None, Path, dict[str, str]]:
    temporary = None
    if state == "cold_empty" and requested is None:
        temporary = tempfile.TemporaryDirectory(prefix="vkl-compile-cache-")
        root = Path(temporary.name)
    elif requested is None:
        raise ValueError("warm_persistent requires --cache-dir")
    else:
        root = Path(requested).expanduser().resolve()

    if state == "cold_empty":
        root.mkdir(parents=True, exist_ok=True)
        if any(root.iterdir()):
            raise ValueError("cold_empty cache directory must be empty before qualification")
    elif not _has_compiler_cache(root):
        raise ValueError(
            "warm_persistent cache directory must contain TorchInductor or Triton cache files"
        )

    torch_cache = root / "torchinductor"
    triton_cache = root / "triton"
    torch_cache.mkdir(parents=True, exist_ok=True)
    triton_cache.mkdir(parents=True, exist_ok=True)
    return temporary, root, {
        "TORCHINDUCTOR_CACHE_DIR": str(torch_cache),
        "TRITON_CACHE_DIR": str(triton_cache),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeout-s", type=float, default=300.0)
    parser.add_argument("--lock-timeout-s", type=float, default=5.0)
    parser.add_argument("--lock-file", default="results/triton/.compile_gate.lock")
    parser.add_argument(
        "--cache-state",
        default="cold_empty",
        choices=("cold_empty", "warm_persistent"),
    )
    parser.add_argument("--cache-dir")
    parser.add_argument("--output", required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if (
        not math.isfinite(args.timeout_s)
        or args.timeout_s <= 0
        or not math.isfinite(args.lock_timeout_s)
        or args.lock_timeout_s <= 0
        or not command
    ):
        raise ValueError("finite positive timeouts and a command after -- are required")

    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    lock_path = Path(args.lock_file)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "compiler_cache_state": args.cache_state,
        "cache_state_verified": False,
        "cache_root": None,
        "cache_ephemeral": False,
        "cache_environment": None,
        "command": command,
        "compile_status": "failed",
        "qualification_wall_s": None,
        "cold_qualification_wall_s": None,
        "lock_wait_s": None,
        "warm_execution_us": None,
        "measurement_status": "unavailable",
        "timeout_s": args.timeout_s,
        "process_group_terminated": False,
    }
    temporary_cache: tempfile.TemporaryDirectory[str] | None = None

    try:
        with lock_path.open("a+", encoding="utf-8") as lock_stream:
            lock_wait = _acquire_lock(lock_stream, args.lock_timeout_s)
            result["lock_wait_s"] = lock_wait
            if lock_wait is None:
                result["compile_status"] = "busy"
                result["reason"] = f"compiler gate lock exceeded {args.lock_timeout_s:g}s"
            else:
                try:
                    temporary_cache, cache_root, cache_environment = _prepare_cache(
                        args.cache_state,
                        args.cache_dir,
                    )
                except ValueError as error:
                    result["reason"] = str(error)
                else:
                    result["cache_state_verified"] = True
                    result["cache_root"] = str(cache_root)
                    result["cache_ephemeral"] = temporary_cache is not None
                    result["cache_environment"] = cache_environment
                    environment = os.environ.copy()
                    environment.update(cache_environment)
                    start = time.monotonic()
                    process = subprocess.Popen(
                        command,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                        start_new_session=True,
                        env=environment,
                    )
                    try:
                        stdout, stderr = process.communicate(timeout=args.timeout_s)
                        elapsed = time.monotonic() - start
                        result["qualification_wall_s"] = elapsed
                        if args.cache_state == "cold_empty":
                            result["cold_qualification_wall_s"] = elapsed
                        result["returncode"] = process.returncode
                        result["stdout"] = stdout
                        result["stderr"] = stderr
                        if process.returncode == 0:
                            result["compile_status"] = "completed"
                            result["measurement_status"] = "not_measured_by_gate"
                        else:
                            result["reason"] = "compiler command returned nonzero"
                    except subprocess.TimeoutExpired:
                        stdout, stderr, terminated = _terminate_group(process)
                        elapsed = time.monotonic() - start
                        result["compile_status"] = "timeout"
                        result["qualification_wall_s"] = elapsed
                        if args.cache_state == "cold_empty":
                            result["cold_qualification_wall_s"] = elapsed
                        result["reason"] = f"compiler command exceeded {args.timeout_s:g}s"
                        result["stdout"] = stdout
                        result["stderr"] = stderr
                        result["returncode"] = process.returncode
                        result["process_group_terminated"] = terminated
    finally:
        destination.write_text(json.dumps(result, indent=2, default=str) + "\n")
        if temporary_cache is not None:
            temporary_cache.cleanup()


if __name__ == "__main__":
    main()
