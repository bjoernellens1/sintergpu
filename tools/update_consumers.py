#!/usr/bin/env python3
"""Deliver verified release pins through reviewable, idempotent consumer PRs."""
import argparse
import base64
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import release

ALLOWED = {f'bjoernellens1/{name}' for name in ('stereoforge', 'eligsir-sim', 'eligsir-native', 'vkl', 'slang-rgbd-slam', 'caspar-slang', 'locus')}
VERIFY_ARGV = ['python3', '.sintergpu/verify_lock.py', '--lock', '.sintergpu/lock.json']
LOCK_PATH = '.sintergpu/lock.json'


def validate_registry(path):
    entries = json.loads(Path(path).read_text())['consumers']
    if {x['repository'] for x in entries} != ALLOWED or len(entries) != 7:
        raise ValueError('registry must contain exactly the seven approved repositories')
    for item in entries:
        if item.get('lock_path') != LOCK_PATH or item.get('verification_argv') != VERIFY_ARGV or item.get('integration_state') not in ('registered', 'integrated'):
            raise ValueError('invalid registered consumer schema')
        if item['integration_state'] == 'registered' and (item.get('consumed_modules') != [] or item.get('integration_argv')):
            raise ValueError('registration cannot claim kernel consumption')
        if item['integration_state'] == 'integrated' and (not item.get('consumed_modules') or not item.get('integration_argv')):
            raise ValueError('integrated consumer requires real modules and test argv')
        for key in ('verification_argv', 'integration_argv'):
            if not isinstance(item.get(key), list) or any(not isinstance(arg, str) or not arg for arg in item[key]):
                raise ValueError('commands must be argv arrays')
    return entries


def classify_pr(prs):
    if len(prs) > 1:
        raise ValueError('multiple matching PRs require investigation')
    if not prs:
        return 'new'
    return 'merged' if prs[0].get('merged_at') else ('open' if prs[0]['state'] == 'open' else 'declined')


def validate_lock(lock):
    if lock.get('schema_version') != 1 or lock.get('repository') != release.REPO:
        raise ValueError('invalid lock schema/repository')
    release.validate_version(lock['tag'], lock['version'])
    if not re.fullmatch('[a-f0-9]{40}', lock.get('source_commit', '')):
        raise ValueError('invalid source commit')
    prefix = f'https://github.com/{release.REPO}/releases/download/{lock["tag"]}/'
    for field in ('manifest', 'artifact'):
        value = lock[field]
        release.safe_name(value['name'])
        if value['url'] != prefix+value['name'] or not re.fullmatch('[a-f0-9]{64}', value['sha256']):
            raise ValueError('invalid immutable asset pin')
    if not re.fullmatch('[a-f0-9]{64}', lock.get('shader_tree_sha256', '')):
        raise ValueError('missing shader digest')
    if lock.get('integration_state') == 'registered' and (lock.get('consumed_modules') != [] or lock.get('migration') != 'pending'):
        raise ValueError('registration is not migration')
    if lock.get('integration_state') not in ('registered', 'integrated'):
        raise ValueError('invalid integration state')


# This verifier becomes downstream-owned source, never downloaded executable code.
VERIFIER = '''#!/usr/bin/env python3
"""Verify release integrity only; kernel migration and native checks are separate."""
import argparse, hashlib, json, pathlib, re, subprocess, tarfile, tempfile, urllib.request
REPO = "bjoernellens1/sintergpu"
WORKFLOW = REPO + "/.github/workflows/release.yml"
def require(value, message):
    if not value: raise ValueError(message)
def gh(*args):
    return subprocess.run(["gh", *args], check=True, capture_output=True, text=True).stdout
def main():
    p = argparse.ArgumentParser(); p.add_argument("--lock", required=True); args = p.parse_args()
    lock = json.loads(pathlib.Path(args.lock).read_text())
    require(lock.get("schema_version") == 1 and lock.get("repository") == REPO, "schema/repository")
    require(re.fullmatch(r"v(0|[1-9][0-9]*)[.](0|[1-9][0-9]*)[.](0|[1-9][0-9]*)", lock["tag"]) and lock["tag"] == "v"+lock["version"], "version/tag")
    require(re.fullmatch("[a-f0-9]{40}", lock["source_commit"]), "source commit")
    info = json.loads(gh("release", "view", lock["tag"], "-R", REPO, "--json", "isDraft,isImmutable"))
    require(not info["isDraft"] and info.get("isImmutable"), "immutable publication required")
    gh("release", "verify", lock["tag"], "-R", REPO)
    with tempfile.TemporaryDirectory() as tmp:
        paths = {}
        for key in ("manifest", "artifact"):
            pin = lock[key]; name = pin["name"]
            require(re.fullmatch("[A-Za-z0-9_.+-]+", name) and name not in (".", ".."), "unsafe asset path")
            require(pin["url"] == "https://github.com/"+REPO+"/releases/download/"+lock["tag"]+"/"+name, "asset URL")
            target = pathlib.Path(tmp)/name
            with urllib.request.urlopen(pin["url"], timeout=60) as response: target.write_bytes(response.read())
            require(hashlib.sha256(target.read_bytes()).hexdigest() == pin["sha256"], "asset hash")
            gh("release", "verify-asset", lock["tag"], str(target), "-R", REPO)
            gh("attestation", "verify", str(target), "--repo", REPO, "--signer-workflow", WORKFLOW, "--source-ref", "refs/tags/"+lock["tag"], "--source-digest", lock["source_commit"], "--deny-self-hosted-runners")
            paths[key] = target
        manifest = json.loads(paths["manifest"].read_text())
        for key in ("repository", "version", "tag", "source_commit", "shader_tree_sha256"):
            require(manifest.get(key) == lock[key], "manifest mismatch: "+key)
        require(manifest.get("workflow") == WORKFLOW, "workflow mismatch")
        require(any(a["name"] == lock["artifact"]["name"] and a["sha256"] == lock["artifact"]["sha256"] and a["size"] == paths["artifact"].stat().st_size for a in manifest["assets"]), "artifact not bound by manifest")
        with tarfile.open(paths["artifact"]) as archive:
            for member in archive:
                path = pathlib.PurePosixPath(member.name)
                require(not path.is_absolute() and ".." not in path.parts and not member.issym() and not member.islnk() and (member.isfile() or member.isdir()), "unsafe archive member")
    print("Release integrity verified; consumer/native qualification remains separate.")
if __name__ == "__main__": main()
'''

DEPENDENCY_WORKFLOW = '''name: SinterGPU release integrity
on:
  pull_request:
    paths: ['.sintergpu/**', '.github/workflows/sintergpu-dependency.yml']
  workflow_dispatch:
permissions:
  contents: read
  attestations: read
jobs:
  verify:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@11bd71901bbe5b1630ceea73d27597364c9af683
        with:
          persist-credentials: false
      - run: python3 .sintergpu/verify_lock.py --lock .sintergpu/lock.json
        env:
          GH_TOKEN: ${{ github.token }}
'''


class API:
    def __init__(self):
        self.attempts = 0

    def __call__(self, endpoint, method='GET', data=None, missing=False):
        args = ['gh', 'api', endpoint, '--method', method, '--include']
        if data is not None:
            args += ['--input', '-']
        for attempt in range(4):
            self.attempts += 1
            result = subprocess.run(args, input=json.dumps(data) if data is not None else None, text=True, capture_output=True)
            if result.returncode == 0:
                body = result.stdout
                if body.startswith('HTTP/'):
                    body = re.split(r'\r?\n\r?\n', body, maxsplit=1)[-1]
                return json.loads(body) if body.strip() else None
            error = result.stderr
            if missing and '(HTTP 404)' in error:
                return None
            if attempt < 3 and (re.search(r'HTTP 429', error) or (method == 'GET' and re.search(r'HTTP 5\d\d', error))):
                wait = re.search(r'Retry-After:\s*(\d+)', result.stdout+'\n'+error, re.I)
                time.sleep(min(60, int(wait[1]) if wait else 2**attempt))
                continue
            # Do not echo credentials, request bodies or environment values.
            code = re.search(r'HTTP (\d+)', error)
            raise RuntimeError('GitHub API failure '+(code[1] if code else 'transport')+' '+method+' '+endpoint.split('?')[0])


def content(api, repo, path, ref):
    item = api(f'repos/{repo}/contents/{path}?ref={ref}', missing=True)
    return None if item is None else base64.b64decode(item['content']).decode()


def make_lock(manifest, directory, consumer):
    name = f'sintergpu-{manifest["version"]}-manifest.json'
    source = next(a for a in manifest['assets'] if a['name'] == f'sintergpu-{manifest["version"]}-source.tar.gz')
    prefix = f'https://github.com/{release.REPO}/releases/download/{manifest["tag"]}/'
    lock = {'schema_version': 1, 'repository': release.REPO, 'version': manifest['version'], 'tag': manifest['tag'],
            'source_commit': manifest['source_commit'], 'shader_tree_sha256': manifest['shader_tree_sha256'],
            'manifest': {'name': name, 'url': prefix+name, 'sha256': release.sha256(directory/name)},
            'artifact': {'name': source['name'], 'url': prefix+source['name'], 'sha256': source['sha256']},
            'integration_state': consumer['integration_state'], 'consumed_modules': consumer['consumed_modules'],
            'migration': 'pending' if consumer['integration_state'] == 'registered' else 'integrated'}
    validate_lock(lock)
    return json.dumps(lock, indent=2, sort_keys=True)+'\n'


def deliver_one(api, item, lock, register=False, dry_run=False):
    repo = item['repository']; tag = json.loads(lock)['tag']; head = 'sintergpu/'+tag
    marker = f'<!-- sintergpu-release:{release.REPO}:{tag} -->'
    base = api(f'repos/{repo}')['default_branch']
    if item.get('base_branch') and item['base_branch'] != base:
        raise ValueError('registered base branch changed')
    base_sha = api(f'repos/{repo}/git/ref/heads/{base}')['object']['sha']
    prs = []
    page = 1
    while True:
        batch = api(f'repos/{repo}/pulls?state=all&head=bjoernellens1:{head}&per_page=100&page={page}')
        prs.extend(batch)
        if len(batch) < 100:
            break
        page += 1
    for pr in prs:
        if marker not in (pr.get('body') or '') or pr['head']['repo']['full_name'] != repo:
            raise ValueError('human PR uses machine branch')
    state = classify_pr(prs)
    if state == 'declined':
        return {'repository': repo, 'status': 'declined', 'url': prs[0]['html_url']}
    if state == 'merged':
        if content(api, repo, LOCK_PATH, prs[0]['merge_commit_sha']) != lock:
            raise ValueError('merged PR does not contain expected release lock')
        return {'repository': repo, 'status': 'delivered', 'url': prs[0]['html_url'], 'consumer_checks': 'inspect downstream CI; integrity delivery is not native qualification'}
    files = {LOCK_PATH: lock}
    current = api(f'repos/{repo}/git/ref/heads/{head}', missing=True)
    if register:
        files.update({'.sintergpu/verify_lock.py': VERIFIER, '.github/workflows/sintergpu-dependency.yml': DEPENDENCY_WORKFLOW})
    elif content(api, repo, '.sintergpu/verify_lock.py', base) is None:
        if not current or state != 'open':
            raise ValueError('initial registration required; use explicit --register with local workflow-authorized gh')
        # A normal retry may inspect and reuse an exact pending registration.
        # It cannot create its workflow: the existing branch must match below.
        files.update({'.sintergpu/verify_lock.py': VERIFIER, '.github/workflows/sintergpu-dependency.yml': DEPENDENCY_WORKFLOW})
    old = current['object']['sha'] if current else base_sha
    if current:
        commit = api(f'repos/{repo}/commits/{old}')
        if commit['commit']['message'] != f'sintergpu: pin {tag}' or any(f['filename'] not in files for f in commit['files']):
            raise ValueError('human-modified machine branch conflict')
        if any(content(api, repo, name, old) != text for name, text in files.items()):
            raise ValueError('existing machine branch bytes differ; investigate instead of overwriting')
    elif any(content(api, repo, name, base) is not None for name in files if register):
        raise ValueError('registration would overwrite existing consumer-owned files')
    if dry_run:
        return {'repository': repo, 'status': 'dry-run', 'action': 'reuse PR' if state == 'open' else 'create PR', 'base_sha': base_sha, 'head_sha': old}
    # Verify precisely staged generated content before writing Git objects.
    with tempfile.TemporaryDirectory() as temp:
        stage = Path(temp); (stage/'.sintergpu').mkdir()
        (stage/LOCK_PATH).write_text(lock)
        verifier = files.get('.sintergpu/verify_lock.py') or content(api, repo, '.sintergpu/verify_lock.py', base)
        # Do not execute human-modified downstream code with delivery credentials.
        if verifier != VERIFIER:
            raise ValueError('consumer-owned verifier changed; review registration before updating')
        (stage/'.sintergpu/verify_lock.py').write_text(verifier)
        subprocess.run(VERIFY_ARGV, cwd=stage, check=True)
    if not current:
        tree = api(f'repos/{repo}/git/trees', 'POST', {'base_tree': base_sha, 'tree': [{'path': path, 'mode': '100644', 'type': 'blob', 'content': text} for path, text in files.items()]})
        commit = api(f'repos/{repo}/git/commits', 'POST', {'message': f'sintergpu: pin {tag}', 'tree': tree['sha'], 'parents': [base_sha]})
        try:
            api(f'repos/{repo}/git/refs', 'POST', {'ref': 'refs/heads/'+head, 'sha': commit['sha']})
        except RuntimeError:
            found = api(f'repos/{repo}/git/ref/heads/{head}', missing=True)
            if not found or found['object']['sha'] != commit['sha']:
                raise
        old = commit['sha']
    title = f'Register SinterGPU {tag} release metadata (kernel migration pending)' if item['integration_state'] == 'registered' else f'Update SinterGPU to {tag}'
    body = f'{marker}\n\nPins immutable release source, manifest and artifact hashes.\n\nIntegration state: {item["integration_state"]}. Kernel migration pending: {item["integration_state"] == "registered"}.\nRelease integrity verification is separate from consumer native tests and GPU qualification.\n\nVerification argv: `{json.dumps(item["verification_argv"])}`\nConsumer integration argv: `{json.dumps(item["integration_argv"])}`\nNo automatic merge.\n'
    prefix = f'https://github.com/{release.REPO}/blob/{tag}/'
    body += f'\n[Changelog]({prefix}CHANGELOG.md) · [Migration instructions]({prefix}docs/integration.md) · [Correctness/performance scope]({prefix}docs/status.md) · [Qualification metadata](https://github.com/{release.REPO}/releases/download/{tag}/sintergpu-{tag[1:]}-manifest.json).\n'
    if state == 'open':
        if prs[0].get('body') != body or prs[0]['head']['sha'] != old:
            raise ValueError('human-modified PR body or stale head conflict')
        pr = prs[0]
    else:
        try:
            pr = api(f'repos/{repo}/pulls', 'POST', {'title': title, 'body': body, 'head': head, 'base': base})
        except RuntimeError:
            # Ambiguous create response: query instead of issuing another create.
            found = api(f'repos/{repo}/pulls?state=all&head=bjoernellens1:{head}')
            if len(found) != 1 or found[0].get('body') != body:
                raise
            pr = found[0]
    return {'repository': repo, 'status': 'delivered', 'url': pr['html_url'], 'head_sha': old, 'base_sha': base_sha,
            'checks_url': f'https://github.com/{repo}/pull/{pr["number"]}/checks', 'consumer_checks': 'pending downstream CI; native qualification not inferred'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tag', required=True); p.add_argument('--registry', type=Path, default=Path('consumers.json'))
    mode = p.add_mutually_exclusive_group(required=True); mode.add_argument('--dry-run', action='store_true'); mode.add_argument('--deliver', action='store_true')
    p.add_argument('--register', action='store_true', help='explicit initial registration; requires workflow write authorization')
    p.add_argument('--repository', choices=sorted(ALLOWED)); p.add_argument('--results', type=Path, default=Path('delivery-results.json'))
    args = p.parse_args(); entries = validate_registry(args.registry)
    results = []; api = API()
    try:
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            manifest = release.verify_download(args.tag, directory)
            for item in entries:
                if args.repository and item['repository'] != args.repository:
                    continue
                try:
                    result = deliver_one(api, item, make_lock(manifest, directory, item), args.register, args.dry_run)
                except Exception as error:
                    result = {'repository': item['repository'], 'status': 'failed', 'error': str(error)}
                results.append(result)
    except Exception as error:
        results = [{'repository': item['repository'], 'status': 'failed', 'error': str(error)} for item in entries if not args.repository or args.repository == item['repository']]
    args.results.write_text(json.dumps({'tag': args.tag, 'results': results, 'api_attempts': api.attempts}, indent=2)+'\n')
    print(json.dumps(results, indent=2))
    return int(any(x['status'] in ('failed', 'declined') for x in results))


if __name__ == '__main__':
    raise SystemExit(main())
