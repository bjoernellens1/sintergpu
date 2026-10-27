import importlib.util
import json
from pathlib import Path
import pytest

spec = importlib.util.spec_from_file_location('release', Path(__file__).resolve().parents[1]/'tools/release.py')
r = importlib.util.module_from_spec(spec)
if spec.loader and Path(spec.origin).exists():
    spec.loader.exec_module(r)


def test_strict_version():
    assert hasattr(r, 'validate_version'), 'release validation is missing'
    assert r.validate_version('v0.1.0', '0.1.0') == '0.1.0'
    for tag in ['v01.1.0', '0.1.0', 'v0.1.0-rc1', 'v0.2.0']:
        with pytest.raises(ValueError):
            r.validate_version(tag, '0.1.0')


def test_checksums_detect_tamper(tmp_path):
    assert hasattr(r, 'write_checksums'), 'artifact checksums are missing'
    asset = tmp_path/'payload.tar.gz'
    asset.write_bytes(b'payload')
    r.write_checksums(tmp_path, [asset.name])
    r.verify_checksums(tmp_path)
    asset.write_bytes(b'tampered')
    with pytest.raises(ValueError):
        r.verify_checksums(tmp_path)


def test_reject_checksum_path_traversal(tmp_path):
    assert hasattr(r, 'verify_checksums'), 'checksum path validation is missing'
    (tmp_path/'SHA256SUMS').write_text('0'*64+'  ../secret\n')
    with pytest.raises(ValueError):
        r.verify_checksums(tmp_path)


def test_qualification_requires_explicit_pending():
    assert hasattr(r, 'validate_qualification'), 'qualification validation is missing'
    with pytest.raises(ValueError):
        r.validate_qualification({})
    r.validate_qualification({'status': 'experimental', 'qualifications': [], 'pending_gates': ['NVIDIA execution']})


def test_draft_resume_only_matching_assets(tmp_path):
    assert hasattr(r, 'check_existing_assets'), 'immutable draft checks are missing'
    p = tmp_path/'a'; p.write_bytes(b'a')
    asset = {'name': 'a', 'size': 1, 'digest': 'sha256:'+r.sha256(p)}
    assert r.check_existing_assets([asset], tmp_path) == {'a'}
    asset['digest'] = 'sha256:'+'0'*64
    with pytest.raises(ValueError):
        r.check_existing_assets([asset], tmp_path)


def compiler(version):
    return {'pinned_release':version,'status':'passed','compiler_version':version,'compiler_sha256':'a'*64,
            'entries':[{'operation':'a','source_sha256':r.hashlib.sha256(b'shader').hexdigest(),'returncode':0,'reflection':{'entryPoints':[{}]}}]}


def source_repo(tmp_path, changelog='## 0.1.0 — 2026-10-27\n\n- Initial release.\n'):
    import subprocess
    (tmp_path/'src/sintergpu/data').mkdir(parents=True)
    (tmp_path/'src/sintergpu/kernels').mkdir()
    (tmp_path/'src/sintergpu/kernels/a.slang').write_text('shader')
    (tmp_path/'VERSION').write_text('0.1.0\n')
    (tmp_path/'src/sintergpu/data/VERSION').write_text('0.1.0\n')
    (tmp_path/'CHANGELOG.md').write_text(changelog)
    (tmp_path/'src/sintergpu/data/catalog.json').write_text(json.dumps({'operations':[{'id':'a','path':'kernels/a.slang','sha256':r.hashlib.sha256(b'shader').hexdigest()}]}))
    for args in [['init'],['add','.'],['-c','user.name=Test','-c','user.email=test@example.invalid','commit','-m','fixture'],['tag','v0.1.0']]:
        subprocess.run(['git',*args], cwd=tmp_path, check=True, capture_output=True)
    return tmp_path


def test_source_bundle_is_deterministic_and_bound(tmp_path):
    (tmp_path/'repo').mkdir()
    root = source_repo(tmp_path/'repo')
    qualification = {'status':'experimental','qualifications':[],'pending_gates':['NVIDIA']}
    outputs = [tmp_path/'first',tmp_path/'second']
    for output in outputs:
        import zipfile, tarfile
        output.mkdir()
        tracked = r.run(['git','ls-files','-z'],cwd=root).decode().strip('\0').split('\0')
        with zipfile.ZipFile(output/'sintergpu-0.1.0-py3-none-any.whl','w') as archive:
            archive.writestr('sintergpu-0.1.0.dist-info/METADATA','Name: sintergpu\nVersion: 0.1.0\n')
            for p in tracked:
                if p.startswith('src/sintergpu/'):
                    archive.write(root/p,p.removeprefix('src/'))
        with tarfile.open(output/'sintergpu-0.1.0.tar.gz','w:gz') as archive:
            for p in tracked:
                archive.add(root/p,'sintergpu-0.1.0/'+p)
        manifest = r.prepare('v0.1.0', output, qualification, [compiler('2026.12'),compiler('2026.19')], root)
        assert manifest['source_commit']
        assert manifest['shader_tree_sha256'] == r.shader_digest(root)
        r.verify_checksums(output)
    assert (outputs[0]/'sintergpu-0.1.0-source.tar.gz').read_bytes() == (outputs[1]/'sintergpu-0.1.0-source.tar.gz').read_bytes()


def test_unreleased_changelog_blocks_release(tmp_path):
    root = source_repo(tmp_path, '## 0.1.0 — unreleased\n\n- Pending.\n')
    with pytest.raises(ValueError, match='changelog'):
        r.prepare('v0.1.0',tmp_path/'dist',{},[],root)


def test_missing_compiler_record_blocks_release(tmp_path):
    root = source_repo(tmp_path)
    with pytest.raises(ValueError, match='compiler'):
        r.prepare('v0.1.0',tmp_path/'dist',{'status':'experimental','qualifications':[],'pending_gates':['NVIDIA']},[compiler('2026.12')],root)


def test_published_retry_only_verifies_no_upload(tmp_path, monkeypatch):
    (tmp_path/'asset').write_bytes(b'payload')
    (tmp_path/'sintergpu-0.1.0-manifest.json').write_text(json.dumps({'version':'0.1.0','repository':r.REPO,'workflow':r.WORKFLOW,'source_commit':'a'*40}))
    r.write_checksums(tmp_path,['asset','sintergpu-0.1.0-manifest.json'])
    calls=[]
    def gh(*args):
        calls.append(args)
        if args[0]=='api' and '/commits/' in args[1]: return '{"sha":"'+'a'*40+'"}'
        if args[0]=='api' and args[1].endswith('immutable-releases'): return '{"enabled":true}'
        if args[0]=='api': return '[[{"tag_name":"v0.1.0","draft":false}]]'
        pytest.fail('published retry attempted mutation')
    monkeypatch.setattr(r,'gh',gh)
    monkeypatch.setattr(r,'verify_download',lambda tag,path:{'verified':tag})
    assert r.publish('v0.1.0',tmp_path)=={'verified':'v0.1.0'}


def test_wheel_version_cannot_disagree_with_source(tmp_path):
    import zipfile
    (tmp_path/'repo').mkdir()
    root=source_repo(tmp_path/'repo')
    output=tmp_path/'dist'; output.mkdir()
    with zipfile.ZipFile(output/'sintergpu-9.0.0-py3-none-any.whl','w') as archive:
        archive.writestr('sintergpu-9.0.0.dist-info/METADATA','Name: sintergpu\nVersion: 9.0.0\n')
    with pytest.raises(ValueError,match='wheel package/version mismatch'):
        r.prepare('v0.1.0',output,{'status':'experimental','qualifications':[],'pending_gates':['NVIDIA']},[compiler('2026.12'),compiler('2026.19')],root)


def test_same_version_stale_package_bytes_are_rejected(tmp_path):
    import zipfile
    root=source_repo(tmp_path)
    wheel=tmp_path/'bad.whl'
    with zipfile.ZipFile(wheel,'w') as archive:
        archive.writestr('sintergpu/kernels/a.slang',b'old shader')
    with pytest.raises(ValueError,match='wheel source'):
        r.verify_package_sources(root,wheel,'wheel')


def test_missing_and_extra_package_sources_are_rejected(tmp_path):
    import zipfile
    root=source_repo(tmp_path)
    wheel=tmp_path/'bad.whl'
    with zipfile.ZipFile(wheel,'w') as archive:
        archive.writestr('sintergpu/kernels/a.slang',b'shader')
    with pytest.raises(ValueError,match='wheel source'):
        r.verify_package_sources(root,wheel,'wheel')


def test_sdist_stale_reference_is_rejected(tmp_path):
    import io,tarfile
    root=source_repo(tmp_path)
    sdist=tmp_path/'bad.tar.gz'
    with tarfile.open(sdist,'w:gz') as archive:
        member=tarfile.TarInfo('sintergpu-0.1.0/src/sintergpu/kernels/a.slang')
        member.size=3; archive.addfile(member,io.BytesIO(b'old'))
    with pytest.raises(ValueError,match='sdist source'):
        r.verify_package_sources(root,sdist,'sdist')


def test_draft_is_addressed_by_id_not_published_tag_endpoint(tmp_path,monkeypatch):
    manifest={'version':'0.1.0','repository':r.REPO,'workflow':r.WORKFLOW,'source_commit':'a'*40}
    name='sintergpu-0.1.0-manifest.json'
    (tmp_path/name).write_text(json.dumps(manifest));(tmp_path/'payload').write_bytes(b'data')
    (tmp_path/'attestation.jsonl').write_text('signed bundle')
    r.write_checksums(tmp_path,[name,'payload'])
    assets=[]; exists=False; published=False
    def gh(*args):
        nonlocal exists,published
        if args[0]=='api':
            endpoint=args[-1]
            assert '/releases/tags/' not in endpoint,'drafts are not returned by published-tag lookup'
            if endpoint.endswith('immutable-releases'):return '{"enabled":true}'
            if '/commits/' in endpoint:return json.dumps({'sha':'a'*40})
            if endpoint.endswith('releases?per_page=100'):return json.dumps([[{'id':42,'tag_name':'v0.1.0','draft':True}]] if exists else [[]])
            if endpoint.endswith('/releases/42'):return json.dumps({'assets':assets})
        if args[:2]==('release','create'):exists=True;return ''
        if args[:2]==('release','upload'):
            p=Path(args[3]);assets.append({'name':p.name,'size':p.stat().st_size,'digest':'sha256:'+r.sha256(p)});return ''
        if args[:2]==('attestation','verify'):return ''
        if args[:2]==('release','edit'):published=True;return ''
        raise AssertionError(args)
    monkeypatch.setattr(r,'gh',gh)
    r.publish('v0.1.0',tmp_path)
    assert published
