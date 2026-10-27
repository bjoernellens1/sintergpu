import subprocess
import sys


def test_run_requires_calibration_and_positive_limit():
    proc = subprocess.run(
        [sys.executable, "-m", "slang_rgbd_slam.cli", "run", "/not/a/dataset", "--max-frames", "0"],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 2
    assert "positive" in proc.stderr


def test_compare_forwards_matching_hybrid_and_map_capacity(monkeypatch, tmp_path, capsys):
    from slang_rgbd_slam import cli, hybrid_comparison

    captured = {}

    def compare(frames, camera, output, **options):
        captured.update(options)
        assert len(frames) == 2
        assert output == tmp_path
        return {"ok": True}

    monkeypatch.setattr(hybrid_comparison, "compare_hybrid_backends", compare)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "slang-rgbd-slam",
            "compare",
            "--frames",
            "2",
            "--max-blocks",
            "1024",
            "--max-keyframes",
            "4",
            "--keyframe-interval",
            "2",
            "--ba-every-keyframes",
            "2",
            "--disable-loops",
            "--output",
            str(tmp_path),
        ],
    )
    assert cli.main() == 0
    assert captured["map_options"] == {"max_blocks": 1024}
    assert captured["hybrid_options"] == dict(
        max_keyframes=4,
        keyframe_interval=2,
        ba_every_keyframes=2,
        enable_loops=False,
        enable_local_ba=True,
    )
    assert '"ok": true' in capsys.readouterr().out


def test_tune_passes_keyword_only_iterations(monkeypatch, tmp_path, capsys):
    import types

    from slang_rgbd_slam import cli, hardware

    runtime = object()
    monkeypatch.setitem(
        sys.modules, "caspar_slang.runtime", types.SimpleNamespace(VulkanRuntime=lambda: runtime)
    )

    def tune(device, cache_path, *, iterations):
        assert device is runtime
        assert cache_path == tmp_path / "profile.json"
        assert iterations == 3
        return {"validated": True}

    monkeypatch.setattr(hardware, "tune", tune)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "slang-rgbd-slam",
            "tune",
            "--iterations",
            "3",
            "--output",
            str(tmp_path / "profile.json"),
        ],
    )
    assert cli.main() == 0
    assert '"validated": true' in capsys.readouterr().out
