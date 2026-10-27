#!/usr/bin/env python3
"""Prepare, verify and publish immutable, attested SinterGPU releases."""
import argparse
import gzip
import email
import zipfile
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import tarfile

REPO = 'bjoernellens1/sintergpu'
WORKFLOW = REPO + '/.github/workflows/release.yml'
ROOT = Path(__file__).resolve().parents[1]


def run(args, **kwargs):
    return subprocess.run(args, check=True, capture_output=True, **kwargs).stdout


def gh(*args):
    return run(['gh', *args], text=True)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_version(tag, version):
    if not re.fullmatch(r'v(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)', tag) or tag != 'v'+version:
        raise ValueError('strict SemVer tag must equal VERSION')
    return version


def safe_name(name):
    if not name or Path(name).name != name or name in ('.', '..') or '\\' in name or not re.fullmatch(r'[A-Za-z0-9_.+-]+', name):
        raise ValueError('unsafe asset name')
    return name


def write_checksums(directory, names):
    directory = Path(directory)
    (directory/'SHA256SUMS').write_text(''.join(f'{sha256(directory/safe_name(n))}  {n}\n' for n in sorted(names)))


def verify_checksums(directory):
    directory = Path(directory)
    names = set()
    for line in (directory/'SHA256SUMS').read_text().splitlines():
        digest, name = line.split('  ', 1)
        safe_name(name)
        if name in names or not re.fullmatch('[a-f0-9]{64}', digest) or sha256(directory/name) != digest:
            raise ValueError('checksum mismatch or duplicate: '+name)
        names.add(name)
    if not names:
        raise ValueError('empty checksums')
    return names


def validate_qualification(record):
    if record.get('status') not in ('experimental', 'qualified') or not isinstance(record.get('qualifications'), list) or not isinstance(record.get('pending_gates'), list):
        raise ValueError('qualification record must declare status, qualifications and pending_gates')
    if record['status'] == 'experimental' and not record['pending_gates']:
        raise ValueError('experimental qualification must declare pending gates')
    if record['status'] == 'qualified' and (record['pending_gates'] or not record['qualifications']):
        raise ValueError('qualified release cannot have missing evidence')
    for item in record['qualifications']:
        if not all(item.get(k) for k in ('operation', 'adapter', 'compiler', 'source_sha256', 'driver', 'device', 'status')):
            raise ValueError('incomplete hardware qualification')


def shader_digest(root):
    h = hashlib.sha256()
    for path in sorted((root/'src/sintergpu/kernels').rglob('*.slang')):
        h.update(path.relative_to(root/'src/sintergpu').as_posix().encode()+b'\0'+path.read_bytes()+b'\0')
    return h.hexdigest()


def verify_package_sources(root, package, kind):
    """Bind distributed sources to tracked release bytes, including references."""
    tracked = run(['git', 'ls-files', '-z'], cwd=root).decode().split('\0')
    if kind == 'wheel':
        expected = {p.removeprefix('src/'): (root/p).read_bytes() for p in tracked
                    if p.startswith('src/sintergpu/')}
        with zipfile.ZipFile(package) as archive:
            names = archive.namelist()
            actual = {p: archive.read(p) for p in names if p.startswith('sintergpu/')}
    else:
        selected = ('src/sintergpu/', 'include/', 'cmake/', 'third_party/', 'references/',
                    'provenance/', 'docs/', 'tools/', 'benchmarks/', 'tests/', 'changes/', 'LICENSES/', 'examples/', 'qualification/')
        singles = {'VERSION', 'LICENSE', 'NOTICE', 'README.md', 'pyproject.toml',
                   'MANIFEST.in', 'CMakeLists.txt', 'consumers.json', 'CHANGELOG.md', 'CONTRIBUTING.md'}
        expected = {p: (root/p).read_bytes() for p in tracked if p.startswith(selected) or p in singles}
        with tarfile.open(package) as archive:
            members = [m for m in archive.getmembers() if m.isfile()]
            names = [m.name for m in members]
            actual = {m.name.partition('/')[2]: archive.extractfile(m).read() for m in members
                      if m.name.partition('/')[2] in expected}
    if len(names) != len(set(names)) or actual != expected:
        raise ValueError(kind+' source bytes do not match tracked release sources')


def prepare(tag, output, qualification, compilers, root=ROOT):
    version = validate_version(tag, (root/'VERSION').read_text().strip())
    if (root/'src/sintergpu/data/VERSION').read_text().strip() != version:
        raise ValueError('packaged VERSION mismatch')
    changelog = (root/'CHANGELOG.md').read_text()
    section = re.search(r'^## '+re.escape(version)+r'\b([^\n]*)\n(.*?)(?=^## |\Z)', changelog, re.M | re.S)
    if not section or 'unreleased' in section.group(1).lower() or not re.search(r'^- .+', section.group(2), re.M):
        raise ValueError('commit a released changelog section before tagging')
    source = run(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
    if run(['git', 'rev-list', '-n', '1', tag], cwd=root, text=True).strip() != source:
        raise ValueError('checkout is not tag commit')
    # Never package a checkout whose source differs from the tagged commit.
    if run(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=root, text=True).strip():
        raise ValueError('tracked checkout must be clean')
    validate_qualification(qualification)
    if len(compilers) != 2 or {c.get('pinned_release') for c in compilers} != {'2026.12', '2026.19'}:
        raise ValueError('both pinned compiler records required')
    catalog = json.loads((root/'src/sintergpu/data/catalog.json').read_text())
    operations = catalog['operations']
    expected_sources = {op['id']: op['sha256'] for op in operations}
    if len(expected_sources) != len(operations):
        raise ValueError('duplicate catalog operations')
    for item in qualification['qualifications']:
        if expected_sources.get(item['operation']) != item['source_sha256']:
            raise ValueError('hardware evidence source does not match catalog')
    for name, digest in qualification.get('evidence_sha256', {}).items():
        path = (root/name).resolve()
        if not path.is_relative_to(root.resolve()) or sha256(path) != digest:
            raise ValueError('qualification evidence digest mismatch')
    package_root = (root/'src/sintergpu').resolve()
    for op in operations:
        path = (package_root/op['path']).resolve()
        if not path.is_relative_to(package_root) or sha256(path) != op['sha256']:
            raise ValueError('catalog source digest mismatch')
    for compiler in compilers:
        actual_sources = {entry['operation']: entry['source_sha256'] for entry in compiler.get('entries', [])}
        if actual_sources != expected_sources or len(actual_sources) != len(compiler.get('entries', [])):
            raise ValueError('compiler source coverage does not match catalog')
        if compiler['pinned_release'] not in compiler.get('compiler_version', ''):
            raise ValueError('actual compiler version does not match pin')
        if compiler.get('status') != 'passed' or not compiler.get('compiler_version') or not compiler.get('compiler_sha256') or not compiler.get('entries') or any(e.get('returncode') != 0 or not e.get('reflection') for e in compiler['entries']):
            raise ValueError('compiler/reflection qualification missing')
    output.mkdir(parents=True, exist_ok=True)
    archive = run(['git', 'archive', '--format=tar', '--prefix=sintergpu-'+version+'/', source], cwd=root)
    (output/f'sintergpu-{version}-source.tar.gz').write_bytes(gzip.compress(archive, mtime=0))
    package_versions = []
    for wheel in sorted(output.glob('*.whl')):
        with zipfile.ZipFile(wheel) as archive:
            metadata_names = [n for n in archive.namelist() if n.endswith('.dist-info/METADATA')]
            if len(metadata_names) != 1:
                raise ValueError('wheel metadata is ambiguous')
            metadata = email.message_from_bytes(archive.read(metadata_names[0]))
            if metadata['Name'] != 'sintergpu' or metadata['Version'] != version:
                raise ValueError('wheel package/version mismatch')
            verify_package_sources(root, wheel, 'wheel')
            package_versions.append({'asset': wheel.name, 'name': metadata['Name'], 'version': metadata['Version']})
    sdists = [p for p in output.glob('*.tar.gz') if not p.name.endswith('-source.tar.gz')]
    for sdist in sdists:
        verify_package_sources(root, sdist, 'sdist')
    if not package_versions or not sdists:
        raise ValueError('wheel and sdist packages required')
    assets = [{'name': p.name, 'size': p.stat().st_size, 'sha256': sha256(p)} for p in sorted(output.iterdir()) if p.is_file() and p.name != 'SHA256SUMS' and not p.name.endswith('-manifest.json')]
    manifest = {'schema_version': 1, 'repository': REPO, 'version': version, 'tag': tag, 'source_commit': source,
                'workflow': WORKFLOW, 'run_url': os.environ.get('GITHUB_SERVER_URL', 'https://github.com')+'/'+REPO+'/actions/runs/'+os.environ.get('GITHUB_RUN_ID', 'local'),
                'shader_tree_sha256': shader_digest(root), 'assets': assets, 'package_versions': package_versions, 'compilers': compilers, **qualification}
    name = f'sintergpu-{version}-manifest.json'
    (output/name).write_text(json.dumps(manifest, indent=2, sort_keys=True)+'\n')
    write_checksums(output, [a['name'] for a in assets]+[name])
    return manifest


def check_existing_assets(assets, directory):
    existing = set()
    for asset in assets:
        name = safe_name(asset['name'])
        path = Path(directory)/name
        if not path.is_file() or asset.get('size') != path.stat().st_size or asset.get('digest') != 'sha256:'+sha256(path):
            raise ValueError('existing release asset differs: '+name)
        existing.add(name)
    return existing


def verify_download(tag, directory):
    release = json.loads(gh('release', 'view', tag, '-R', REPO, '--json', 'tagName,isDraft,isImmutable,assets,url'))
    if release['isDraft'] or not release.get('isImmutable') or release['tagName'] != tag:
        raise ValueError('published immutable release required')
    directory.mkdir(parents=True, exist_ok=True)
    gh('release', 'download', tag, '-R', REPO, '--dir', str(directory))
    names = verify_checksums(directory)
    manifest_path = directory/f'sintergpu-{tag[1:]}-manifest.json'
    manifest = json.loads(manifest_path.read_text())
    validate_version(tag, manifest['version'])
    if manifest.get('repository') != REPO or manifest.get('workflow') != WORKFLOW or not re.fullmatch('[0-9a-f]{40}', manifest['source_commit']):
        raise ValueError('manifest provenance mismatch')
    remote = json.loads(gh('api', f'repos/{REPO}/commits/{tag}'))['sha']
    if remote != manifest['source_commit']:
        raise ValueError('remote tag source mismatch')
    expected = {manifest_path.name} | {safe_name(a['name']) for a in manifest['assets']}
    if names != expected:
        raise ValueError('manifest/checksum asset set mismatch')
    for asset in manifest['assets']:
        path = directory/asset['name']
        if sha256(path) != asset['sha256'] or path.stat().st_size != asset['size']:
            raise ValueError('manifest asset mismatch')
    gh('release', 'verify', tag, '-R', REPO)
    for name in sorted(names | {'SHA256SUMS'}):
        gh('release', 'verify-asset', tag, str(directory/name), '-R', REPO)
        gh('attestation', 'verify', str(directory/name), '--repo', REPO, '--signer-workflow', WORKFLOW,
           '--source-ref', 'refs/tags/'+tag, '--source-digest', remote, '--deny-self-hosted-runners')
    return manifest


def publish(tag, directory):
    verify_checksums(directory)
    manifest = json.loads((directory/f'sintergpu-{tag[1:]}-manifest.json').read_text())
    validate_version(tag, manifest['version'])
    if manifest.get('repository') != REPO or manifest.get('workflow') != WORKFLOW:
        raise ValueError('prepared manifest provenance mismatch')
    remote = json.loads(gh('api', f'repos/{REPO}/commits/{tag}'))['sha']
    if remote != manifest.get('source_commit'):
        raise ValueError('prepared source does not match remote tag')
    settings = json.loads(gh('api', f'repos/{REPO}/immutable-releases'))
    if not settings.get('enabled'):
        raise ValueError('enable repository immutable releases first')
    # A missing release is distinguished from auth/server failures through the API list.
    releases = json.loads(gh('api', '--paginate', '--slurp', f'repos/{REPO}/releases?per_page=100'))
    match = [r for page in releases for r in page if r['tag_name'] == tag]
    if match and not match[0]['draft']:
        with tempfile.TemporaryDirectory() as temp:
            return verify_download(tag, Path(temp))
    if not match:
        gh('release', 'create', tag, '-R', REPO, '--draft', '--verify-tag', '--title', tag, '--notes', 'Experimental kernels; see manifest qualification and pending gates.')
    details = json.loads(gh('api', f'repos/{REPO}/releases/tags/{tag}'))
    existing = check_existing_assets(details['assets'], directory)
    for path in sorted(directory.iterdir()):
        if path.is_file() and path.name not in existing:
            gh('release', 'upload', tag, str(path), '-R', REPO)
    details = json.loads(gh('api', f'repos/{REPO}/releases/tags/{tag}'))
    check_existing_assets(details['assets'], directory)
    if {a['name'] for a in details['assets']} != {p.name for p in directory.iterdir() if p.is_file()}:
        raise ValueError('release upload incomplete')
    for name in verify_checksums(directory) | {'SHA256SUMS'}:
        gh('attestation', 'verify', str(directory/name), '--repo', REPO, '--signer-workflow', WORKFLOW,
           '--source-ref', 'refs/tags/'+tag, '--source-digest', manifest['source_commit'], '--deny-self-hosted-runners')
    if not (directory/'attestation.jsonl').is_file():
        raise ValueError('attestation bundle required')
    remote = json.loads(gh('api', f'repos/{REPO}/commits/{tag}'))['sha']
    if remote != manifest['source_commit']:
        raise ValueError('remote tag changed before publication')
    gh('release', 'edit', tag, '-R', REPO, '--draft=false')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'publish', 'verify'])
    parser.add_argument('--tag', required=True)
    parser.add_argument('--output', type=Path, default=Path('dist'))
    parser.add_argument('--qualification', type=Path)
    parser.add_argument('--compiler', type=Path, action='append', default=[])
    args = parser.parse_args()
    validate_version(args.tag, args.tag[1:])
    if args.command == 'prepare':
        if not args.qualification:
            parser.error('--qualification required')
        prepare(args.tag, args.output, json.loads(args.qualification.read_text()), [json.loads(p.read_text()) for p in args.compiler])
    elif args.command == 'publish':
        publish(args.tag, args.output)
    else:
        verify_download(args.tag, args.output)


if __name__ == '__main__':
    main()
