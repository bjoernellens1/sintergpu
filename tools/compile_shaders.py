#!/usr/bin/env python3
"""Compile every packaged catalog entry and retain reflection/compiler evidence."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from sintergpu import __version__, kernel_path
from sintergpu.catalog import list_operations, validate_catalog


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--slangc', default='slangc')
    parser.add_argument('--output', type=Path, default=ROOT/'build/shaders')
    parser.add_argument('--operation')
    args = parser.parse_args()
    compiler = shutil.which(args.slangc)
    if compiler is None:
        parser.error('Slang compiler unavailable; supply pinned --slangc')
    errors = validate_catalog()
    if errors:
        raise RuntimeError(errors)
    args.output.mkdir(parents=True, exist_ok=True)
    version = subprocess.run([compiler, '-version'], capture_output=True, text=True)
    record = {'version': __version__, 'compiler_version': (version.stdout+version.stderr).strip(),
              'compiler_sha256': hashlib.sha256(Path(compiler).read_bytes()).hexdigest(),
              'target': 'spirv_1_5', 'entries': [], 'gpu_execution': 'not performed'}
    for op in list_operations():
        if args.operation and op['id'] != args.operation:
            continue
        spv = args.output / (op['id']+'.spv')
        reflection = spv.with_suffix('.reflection.json')
        cmd = [compiler, str(kernel_path(op['id'])), '-target', 'spirv', '-profile', 'spirv_1_5', '-entry', op['entrypoint'], '-o', str(spv), '-reflection-json', str(reflection)]
        result = subprocess.run(cmd, capture_output=True, text=True)
        item = {'operation': op['id'], 'source_sha256': op['sha256'], 'returncode': result.returncode,
                'diagnostics': result.stdout+result.stderr}
        if result.returncode == 0:
            item['spirv_sha256'] = hashlib.sha256(spv.read_bytes()).hexdigest()
            item['reflection'] = json.loads(reflection.read_text())
        record['entries'].append(item)
    record['status'] = 'passed' if record['entries'] and all(e['returncode']==0 for e in record['entries']) else 'failed'
    (args.output/'results.json').write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps({'status': record['status'], 'entries': len(record['entries']), 'compiler': record['compiler_version']}))
    return 0 if record['status']=='passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
