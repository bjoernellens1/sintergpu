#!/usr/bin/env python3
"""Import approved frozen sources from git blobs, never donor working files.

Run from the repository root. Existing imported destinations are replaced only
when their bytes still match their previous imported digest.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / 'src/sintergpu'
DONORS = {
    'locus': ('66f67d38e2e34ceec626dd13857ca5de76deafba', 'MIT'),
    'caspar-slang': ('cab53390aa111cd5526cb30c9aef484e73521186', 'Apache-2.0'),
    'slang-rgbd-slam': ('e35e52b35f692b8337696ea3c9d6721d1cc2bc6b', 'Apache-2.0'),
    'vkl': ('3384023a311651ea4260cf12ddab1567972f89f3', 'Apache-2.0 (owner authorization 2026-10-27)'),
    'eligsir-sim': ('57a6fc17edfb523f90647dcbccf182db1245a1db', 'Apache-2.0'),
    'xsplat-vk': ('5c41a47c9e6ec1ad666d06abd239f1c02424624b', 'Apache-2.0'),
}


def blob(repo, revision, path):
    return subprocess.check_output(['git', '-C', str(repo), 'show', f'{revision}:{path}'])


def write_imports(root, pending, previous):
    """Preflight every destination before writing any imported source."""
    for destination, data in pending.items():
        relative = str(destination.relative_to(root))
        if not destination.exists() or destination.read_bytes() == data:
            continue
        record = previous.get(relative)
        if record is None:
            raise RuntimeError(f'refusing untracked import destination: {relative}')
        if hashlib.sha256(destination.read_bytes()).hexdigest() != record['imported_sha256']:
            raise RuntimeError(f'refusing modified import destination: {relative}')
        if record.get('changes'):
            raise RuntimeError(f'refusing to replace adapted import: {relative}')
    for destination, data in pending.items():
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('donor', choices=DONORS)
    parser.add_argument('repository', type=Path)
    args = parser.parse_args()
    revision, license_id = DONORS[args.donor]
    manifest_path = ROOT / 'provenance' / (args.donor + '.json')
    previous = {r['destination']: r for r in json.loads(manifest_path.read_text())['files']} if manifest_path.exists() else {}
    pending = {}
    paths = subprocess.check_output(['git', '-C', str(args.repository), 'ls-tree', '-rz', '--name-only', revision]).decode().split('\0')
    records = []
    for path in filter(None, paths):
        name = args.donor
        is_license = Path(path).name.upper().startswith(('LICENSE', 'NOTICE', 'COPYING', 'THIRD_PARTY'))
        if name == 'locus':
            selected = path.startswith(('gpu/', 'tests/gpu/')) or is_license or path == 'docs/IMPORT_PROVENANCE.json'
            selected = selected and not path.endswith(('.swift', '.metal'))
        elif name == 'vkl':
            selected = path.startswith(('src/vkl/kernels/', 'tests/test_', 'recipes/triton/', 'recipes/splatograph/bench_project_depth', 'docs/TRITON', 'docs/KERNEL_LATENCY')) or is_license or path == 'src/vkl/__init__.py'
        elif name == 'eligsir-sim':
            selected = path.startswith(('src/eligsir_sim/video/colour.py', 'src/eligsir_sim/video/_colour_hip_ext.py', 'tests/video/test_colour')) or is_license
        elif name == 'xsplat-vk':
            selected = path.startswith(('shaders/', 'src/reference', 'include/xsplat/reference', 'tests/reference', 'tests/passes')) or is_license
        else:
            selected = path.startswith(('src/', 'tests/', 'docs/VALIDATION', 'docs/PERFORMANCE')) or is_license or path == 'pyproject.toml'
        if not selected:
            continue
        data = blob(args.repository, revision, path)
        destination = ROOT / 'references/donors' / name / path
        pending[destination] = data
        digest = hashlib.sha256(data).hexdigest()
        records.append({'source_path': path, 'destination': str(destination.relative_to(ROOT)), 'source_sha256': digest, 'imported_sha256': digest, 'changes': []})
        shader_path = None
        if path.endswith('.slang'):
            if name == 'locus' and path.startswith('gpu/'):
                shader_path = Path('kernels/locus') / path.removeprefix('gpu/')
            elif '/shaders/' in path or path.startswith('shaders/'):
                shader_path = Path('kernels') / name.replace('-', '_') / path.split('shaders/', 1)[1]
        if shader_path:
            target = PACKAGE / shader_path
            pending[target] = data
            records.append({'source_path': path, 'destination': str(target.relative_to(ROOT)), 'source_sha256': digest, 'imported_sha256': digest, 'changes': []})
        if name == 'locus' and path in ('tests/gpu/reference.py', 'tests/gpu/cases.py'):
            target = PACKAGE / 'fixtures/locus' / Path(path).name
            adapted = data.replace(b'from reference import', b'from .reference import')
            pending[target] = adapted
            records.append({'source_path': path, 'destination': str(target.relative_to(ROOT)), 'source_sha256': digest, 'imported_sha256': hashlib.sha256(adapted).hexdigest(), 'changes': ['package-relative reference import'] if adapted != data else []})
    manifest = {'schema_version': 1, 'repository': f'bjoernellens1/{args.donor}', 'revision': revision, 'license': license_id, 'author': 'see retained source copyright and notices', 'authorization': 'Original owner code may be Apache-2.0 per direct user authorization on 2026-10-27; third-party terms unchanged.', 'files': records, 'qualification': 'Imported source and historical evidence; requires new SinterGPU verification.'}
    write_imports(ROOT, pending, previous)
    destination = ROOT / 'provenance' / (args.donor + '.json')
    destination.parent.mkdir(exist_ok=True)
    destination.write_text(json.dumps(manifest, indent=2) + '\n')
    print(f'{args.donor}: {len(records)} frozen source records')


if __name__ == '__main__':
    main()
