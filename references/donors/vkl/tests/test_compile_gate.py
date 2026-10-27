import fcntl
import json
import subprocess
import sys
import time
from pathlib import Path


def _script():
    return Path(__file__).resolve().parents[1] / "recipes/triton/compile_gate.py"


def _base(output: Path, lock: Path):
    return [
        sys.executable,
        str(_script()),
        "--lock-file",
        str(lock),
        "--output",
        str(output),
    ]


def test_compile_gate_records_completed_cold_compile_command(tmp_path):
    output = tmp_path / "compile.json"
    command = (
        "import os; "
        "print(os.environ['TORCHINDUCTOR_CACHE_DIR']); "
        "print(os.environ['TRITON_CACHE_DIR'])"
    )
    completed = subprocess.run(
        [
            *_base(output, tmp_path / "gate.lock"),
            "--timeout-s",
            "5",
            "--cache-state",
            "cold_empty",
            "--",
            sys.executable,
            "-c",
            command,
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    result = json.loads(output.read_text())
    assert result["compile_status"] == "completed"
    assert result["measurement_status"] == "not_measured_by_gate"
    assert result["compiler_cache_state"] == "cold_empty"
    assert result["cache_state_verified"] is True
    assert result["cache_ephemeral"] is True
    assert result["cache_environment"]["TORCHINDUCTOR_CACHE_DIR"] in result["stdout"]
    assert result["cache_environment"]["TRITON_CACHE_DIR"] in result["stdout"]
    assert result["qualification_wall_s"] >= 0
    assert result["cold_qualification_wall_s"] == result["qualification_wall_s"]
    assert result["lock_wait_s"] >= 0


def test_compile_gate_rejects_nonempty_cold_cache(tmp_path):
    output = tmp_path / "invalid-cold.json"
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "existing.bin").write_text("cached")

    completed = subprocess.run(
        [
            *_base(output, tmp_path / "gate.lock"),
            "--cache-state",
            "cold_empty",
            "--cache-dir",
            str(cache),
            "--",
            sys.executable,
            "-c",
            "raise SystemExit('must not run')",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    result = json.loads(output.read_text())
    assert result["compile_status"] == "failed"
    assert result["cache_state_verified"] is False
    assert "must be empty" in result["reason"]
    assert result["qualification_wall_s"] is None


def test_compile_gate_rejects_unrelated_file_as_warm_cache(tmp_path):
    output = tmp_path / "invalid-warm.json"
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "unrelated.txt").write_text("not a compiler cache")

    completed = subprocess.run(
        [
            *_base(output, tmp_path / "gate.lock"),
            "--cache-state",
            "warm_persistent",
            "--cache-dir",
            str(cache),
            "--",
            sys.executable,
            "-c",
            "raise SystemExit('must not run')",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    result = json.loads(output.read_text())
    assert result["compile_status"] == "failed"
    assert result["cache_state_verified"] is False
    assert "TorchInductor or Triton cache files" in result["reason"]
    assert result["qualification_wall_s"] is None


def test_compile_gate_reuses_explicit_warm_cache(tmp_path):
    output = tmp_path / "warm.json"
    cache = tmp_path / "cache"
    torch_cache = cache / "torchinductor"
    torch_cache.mkdir(parents=True)
    (torch_cache / "prior.bin").write_text("cached")
    command = "import os; print(os.environ['TORCHINDUCTOR_CACHE_DIR'])"

    completed = subprocess.run(
        [
            *_base(output, tmp_path / "gate.lock"),
            "--cache-state",
            "warm_persistent",
            "--cache-dir",
            str(cache),
            "--",
            sys.executable,
            "-c",
            command,
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    result = json.loads(output.read_text())
    assert result["compile_status"] == "completed"
    assert result["compiler_cache_state"] == "warm_persistent"
    assert result["cache_state_verified"] is True
    assert result["cache_ephemeral"] is False
    assert Path(result["cache_root"]) == cache.resolve()
    assert result["cold_qualification_wall_s"] is None
    assert result["cache_environment"]["TORCHINDUCTOR_CACHE_DIR"] in result["stdout"]


def test_compile_gate_records_timeout_and_terminates_child_process_group(tmp_path):
    output = tmp_path / "timeout.json"
    orphan = tmp_path / "orphan.txt"
    child = (
        "import subprocess,sys,time; "
        f"subprocess.Popen([sys.executable,'-c',\"import time,pathlib; time.sleep(0.25); "
        f"pathlib.Path({str(orphan)!r}).write_text('orphan')\"]); "
        "time.sleep(5)"
    )
    completed = subprocess.run(
        [
            *_base(output, tmp_path / "gate.lock"),
            "--timeout-s",
            "0.05",
            "--",
            sys.executable,
            "-c",
            child,
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    result = json.loads(output.read_text())
    assert result["compile_status"] == "timeout"
    assert result["measurement_status"] == "unavailable"
    assert result["warm_execution_us"] is None
    assert result["cache_state_verified"] is True
    assert result["cold_qualification_wall_s"] >= 0.05
    assert result["process_group_terminated"] is True
    time.sleep(0.3)
    assert not orphan.exists()


def test_compile_gate_does_not_start_second_compiler_while_lock_is_owned(tmp_path):
    output = tmp_path / "busy.json"
    lock_path = tmp_path / "gate.lock"
    with lock_path.open("a+", encoding="utf-8") as lock_stream:
        fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        completed = subprocess.run(
            [
                *_base(output, lock_path),
                "--lock-timeout-s",
                "0.05",
                "--",
                sys.executable,
                "-c",
                "raise SystemExit('must not run')",
            ],
            check=False,
            capture_output=True,
            text=True,
        )

    assert completed.returncode == 0, completed.stderr
    result = json.loads(output.read_text())
    assert result["compile_status"] == "busy"
    assert result["qualification_wall_s"] is None
    assert result["measurement_status"] == "unavailable"
    assert result["cache_state_verified"] is False
