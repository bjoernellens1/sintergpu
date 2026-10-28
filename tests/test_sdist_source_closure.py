"""Exercise the release source check with staged-like additional evidence."""
import importlib.util
from pathlib import Path
import shutil
import subprocess
import sys


def test_sdist_contains_tracked_and_new_source_types(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[1]
    tracked = subprocess.check_output(['git', 'ls-files', '-z'], cwd=root).decode().strip('\0').split('\0')
    tree = tmp_path / 'tree'
    for relative in tracked:
        source = root / relative
        if source.is_file():
            destination = tree / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
    # Simulate newly staged files, including extensionless evidence, without
    # touching the shared Git index or depending on current peer edits.
    extras = {
        'qualification/evidence/sdist-test/exit-status': b'0\n',
        'qualification/evidence/sdist-test/detail.txt': b'evidence\n',
        'benchmarks/shaders/sdist-test.slang': b'// test shader\n',
        'benchmarks/native/sdist-test.c': b'/* test source */\n',
    }
    for relative, data in extras.items():
        target = tree / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    output = tmp_path / 'dist'
    # All build intermediates and the build's temp directory remain isolated.
    temp = tmp_path / 'temp'
    temp.mkdir()
    import os
    env = dict(os.environ, TMPDIR=str(temp))
    subprocess.run([sys.executable, '-m', 'build', '--sdist', '--no-isolation', '--outdir', str(output)],
                   cwd=tree, env=env, check=True, capture_output=True, text=True)
    archives = list(output.glob('*.tar.gz'))
    assert len(archives) == 1
    spec = importlib.util.spec_from_file_location('sinter_release_sdist_test', tree / 'tools/release.py')
    release = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(release)
    selected = [p for p in tracked if (tree / p).is_file()] + list(extras)
    monkeypatch.setattr(release, 'run', lambda *args, **kwargs: ('\0'.join(selected) + '\0').encode())
    release.verify_package_sources(tree, archives[0], 'sdist')
