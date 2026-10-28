#!/usr/bin/env python3
"""Snapshot an isolated RHI build, without claiming build-time attestation.

Capture before and after a build separately and compare inputs, then verify the
post-build snapshot before use. This records file bytes and symlink identities;
it does not establish dynamic linker mappings or unrecorded system dependencies.
Manifests are untrusted: validation does not authenticate capture time or provenance.
"""
import argparse
import hashlib
import json
import os
import re
from pathlib import Path

ROOTS = ('parent-source', 'rhi-source', 'slangpy-source', 'dependency-source', 'slang-sdk', 'slang-sdk/lib/slang-standard-module-2026.12')
INPUT_FILES = ('probe.cpp', 'edit.py', 'build/CMakeCache.txt', 'build/compile_commands.json', 'build/build.ninja')
OUTPUT_FILES = ('build/libsgl.so', 'build/libslang-compiler.so.0.2026.12',
         'build/libslang-glsl-module-2026.12.so', 'build/libslang-glslang-2026.12.so',
         'build/slang-rhi/libslang-rhi.a', 'build/slang-rhi/libslang-rhi-resources.a',
         'build/slang-rhi/libslang-rhi-vma.a', 'build/slangpy/libsgl_data.a',
         'build/rhi_receipt_cpu_probe')
SCOPE = 'filesystem snapshot only; no complete dependency or loaded-library attestation'
TOOLS = ('CMAKE_C_COMPILER', 'CMAKE_CXX_COMPILER', 'CMAKE_MAKE_PROGRAM', 'CMAKE_AR', 'CMAKE_LINKER', 'CMAKE_COMMAND',
         'CMAKE_C_COMPILER_AR', 'CMAKE_CXX_COMPILER_AR', 'CMAKE_RANLIB', 'CMAKE_STRIP')


def digest_file(path):
    before = path.stat()
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    after = path.stat()
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
        raise ValueError(f'file changed while hashing: {path}')
    return h.hexdigest()


def file_identity(path):
    link = os.readlink(path) if path.is_symlink() else None
    target = path.resolve(strict=True)
    if not target.is_file():
        raise ValueError(f'not a file: {path}')
    return {'link': link, 'resolved': str(target), 'sha256': digest_file(target)}


def snapshot(lab, phase):
    if phase not in ('pre', 'post'):
        raise ValueError('phase must be pre or post')
    lab = Path(lab).resolve(strict=True)
    roots = {name: (lab / name).resolve(strict=True) for name in ROOTS}
    if any(not path.is_dir() for path in roots.values()):
        raise ValueError('all source roots must be directories')
    items = {}
    for name, root in roots.items():
        alias = lab / name
        items['root:' + name] = {'resolved': str(root), 'link': os.readlink(alias) if alias.is_symlink() else None}
        for directory, dirs, files in os.walk(root, followlinks=False):
            for entry in sorted(dirs + files):
                path = Path(directory) / entry
                relative = path.relative_to(root).as_posix()
                # Build emits these in the source package; they are post outputs.
                if name == 'slangpy-source' and (relative == 'slangpy/.build_dir' or (relative.startswith('slangpy/slangpy_ext') and relative.endswith('.so'))):
                    continue
                key = name + '/' + relative
                if path.is_symlink():
                    target = path.resolve(strict=True)
                    if target.is_dir() and not any(target == base or target.is_relative_to(base) for base in roots.values()):
                        raise ValueError(f'directory link outside recorded roots: {path} -> {target}')
                    items[key] = {'link': os.readlink(path), 'resolved': str(target)}
                    if target.is_file():
                        items[key]['target_sha256'] = digest_file(target)
                elif path.is_file():
                    items[key] = {'sha256': digest_file(path)}
                elif not path.is_dir():
                    raise ValueError(f'unsupported source entry: {path}')
    for name in INPUT_FILES:
        path = lab / name
        if not path.is_file():
            raise ValueError(f'missing required build input: {path}')
        items['file:' + name] = file_identity(path)
    outputs = {}
    if phase == 'post':
        for name in OUTPUT_FILES:
            path = lab / name
            if not path.is_file():
                raise ValueError(f'missing required build output: {path}')
            outputs[name] = file_identity(path)
        extensions = sorted((roots['slangpy-source'] / 'slangpy').glob('slangpy_ext*.so'))
        if len(extensions) != 1:
            raise ValueError('expected exactly one SlangPy extension binary')
        outputs['extension'] = file_identity(extensions[0])
    cache = (lab / 'build/CMakeCache.txt').read_text().splitlines()
    for name in TOOLS:
        entries = [line.split('=', 1)[1] for line in cache if line.startswith(name + ':') and '=' in line]
        if len(entries) != 1:
            raise ValueError(f'missing or ambiguous configured tool: {name}')
        tool = Path(entries[0])
        if not tool.is_file() or not os.access(tool, os.X_OK):
            raise ValueError(f'configured tool is not executable: {tool}')
        items['tool:' + name] = file_identity(tool)
    return {'schema_version': 2, 'lab': str(lab), 'phase': phase, 'scope': SCOPE, 'inputs': items, 'outputs': outputs}



def validate_manifest(record, phase):
    if not isinstance(record, dict) or set(record) != {'schema_version', 'lab', 'phase', 'scope', 'inputs', 'outputs'}:
        raise ValueError('invalid manifest structure')
    lab = record['lab']
    if (record['schema_version'] != 2 or record['phase'] != phase or record['scope'] != SCOPE or
        not isinstance(lab, str) or not Path(lab).is_absolute() or str(Path(lab)) != lab or
        not isinstance(record['inputs'], dict) or not isinstance(record['outputs'], dict)):
        raise ValueError('invalid manifest identity')
    required = ({'root:' + x for x in ROOTS} | {'file:' + x for x in INPUT_FILES} |
                {'tool:' + x for x in TOOLS})
    if not required <= record['inputs'].keys():
        raise ValueError('missing required input identities')
    expected_outputs = set(OUTPUT_FILES) | {'extension'} if phase == 'post' else set()
    if set(record['outputs']) != expected_outputs:
        raise ValueError('missing or unexpected output identities')
    def valid_file(value):
        return (isinstance(value, dict) and set(value) == {'link', 'resolved', 'sha256'} and
                (value['link'] is None or isinstance(value['link'], str) and bool(value['link'])) and
                isinstance(value['resolved'], str) and Path(value['resolved']).is_absolute() and
                isinstance(value['sha256'], str) and re.fullmatch('[0-9a-f]{64}', value['sha256']))
    for key in {'file:' + x for x in INPUT_FILES} | {'tool:' + x for x in TOOLS}:
        if not valid_file(record['inputs'][key]): raise ValueError(f'invalid input identity: {key}')
    for key, value in record['outputs'].items():
        if not valid_file(value): raise ValueError(f'invalid output identity: {key}')
    for name in ROOTS:
        root = record['inputs']['root:' + name]
        if (not isinstance(root, dict) or set(root) != {'resolved', 'link'} or
            not isinstance(root['resolved'], str) or not Path(root['resolved']).is_absolute() or
            not (root['link'] is None or isinstance(root['link'], str) and root['link'])):
            raise ValueError(f'invalid root identity: {name}')
    # The live post-build snapshot below validates all remaining tree entries,
    # their exact key set and their file/link/hash structures.


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('capture', 'check', 'compare-inputs'))
    parser.add_argument('--lab', type=Path)
    parser.add_argument('--phase', choices=('pre', 'post'))
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--post-manifest', type=Path)
    args = parser.parse_args()
    if args.mode == 'compare-inputs':
        pre = json.loads(args.manifest.read_text())
        post = json.loads(args.post_manifest.read_text()) if args.post_manifest else {}
        try:
            validate_manifest(pre, 'pre')
            validate_manifest(post, 'post')
            if pre['lab'] != post['lab'] or pre['inputs'] != post['inputs']:
                raise ValueError('pre/post inputs mismatch')
            if snapshot(post['lab'], 'post') != post:
                raise ValueError('post manifest differs from live lab')
        except (ValueError, OSError, TypeError, KeyError) as exc:
            raise SystemExit(f'pre/post inputs mismatch or invalid manifest: {exc}') from exc
        print('pre/post recorded inputs match; complete build dependency and runtime identity remain unproven')
        return
    if not args.lab or not args.phase:
        parser.error('--lab and --phase are required')
    actual = snapshot(args.lab, args.phase)
    if args.mode == 'capture':
        with args.manifest.open('x') as f:
            json.dump(actual, f, sort_keys=True, indent=2)
            f.write('\n')
    elif actual != json.loads(args.manifest.read_text()):
        raise SystemExit('snapshot mismatch; refuse to use this build')
    print(f'{args.mode} {args.phase}: {len(actual["inputs"])} inputs, {len(actual["outputs"])} outputs; {actual["scope"]}')


if __name__ == '__main__':
    main()
