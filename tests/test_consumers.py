import importlib.util
from pathlib import Path
import pytest

spec = importlib.util.spec_from_file_location('update_consumers', Path(__file__).resolve().parents[1]/'tools/update_consumers.py')
u = importlib.util.module_from_spec(spec)
if spec.loader and Path(spec.origin).exists():
    spec.loader.exec_module(u)


def test_registry_exact_allowlist():
    assert hasattr(u, 'validate_registry'), 'consumer registry validation is missing'
    registry = u.validate_registry(Path(__file__).resolve().parents[1]/'consumers.json')
    assert len(registry) == 7
    assert all(x['consumed_modules'] == [] for x in registry)


def test_closed_merged_open_idempotency():
    assert hasattr(u, 'classify_pr'), 'PR idempotency is missing'
    assert u.classify_pr([]) == 'new'
    assert u.classify_pr([{'state':'closed','merged_at':None}]) == 'declined'
    assert u.classify_pr([{'state':'closed','merged_at':'today'}]) == 'merged'
    assert u.classify_pr([{'state':'open','merged_at':None}]) == 'open'
    with pytest.raises(ValueError):
        u.classify_pr([{'state':'open'}, {'state':'open'}])


def test_lock_rejects_mutable_or_wrong_source():
    assert hasattr(u, 'validate_lock'), 'consumer lock validation is missing'
    with pytest.raises(ValueError):
        u.validate_lock({'version':'0.1.0', 'tag':'main'})


def consumer():
    return {'repository':'bjoernellens1/locus','base_branch':'main','integration_state':'registered','consumed_modules':[],
            'verification_argv':u.VERIFY_ARGV,'integration_argv':[]}


def api_fixture(prs=None, branch=None, lock=None, modified=False):
    def api(endpoint, method='GET', data=None, missing=False):
        assert method == 'GET', 'test must never mutate remote state'
        if endpoint.endswith('/locus'): return {'default_branch':'main'}
        if '/pulls?' in endpoint: return prs or []
        if endpoint.endswith('/git/ref/heads/main'): return {'object':{'sha':'base'}}
        if '/git/ref/heads/sintergpu/' in endpoint: return branch
        if '/commits/' in endpoint: return {'commit':{'message':'human edit' if modified else 'sintergpu: pin v0.1.0'},'files':[{'filename':u.LOCK_PATH}]}
        if '/contents/' in endpoint:
            if u.LOCK_PATH in endpoint and lock: return {'content':u.base64.b64encode(lock.encode()).decode()}
            return None
        raise AssertionError(endpoint)
    return api


def test_declined_pr_never_recreated():
    pr={'state':'closed','merged_at':None,'body':'<!-- sintergpu-release:bjoernellens1/sintergpu:v0.1.0 -->',
        'head':{'repo':{'full_name':'bjoernellens1/locus'}},'html_url':'https://github.com/test/1'}
    result=u.deliver_one(api_fixture([pr]),consumer(),'{"tag":"v0.1.0"}',register=True)
    assert result['status']=='declined'


def test_merged_pr_checks_recorded_lock_without_mutation():
    lock='{"tag":"v0.1.0"}'
    pr={'state':'closed','merged_at':'today','merge_commit_sha':'merge','body':'<!-- sintergpu-release:bjoernellens1/sintergpu:v0.1.0 -->',
        'head':{'repo':{'full_name':'bjoernellens1/locus'}},'html_url':'https://github.com/test/1'}
    assert u.deliver_one(api_fixture([pr],lock=lock),consumer(),lock,register=True)['status']=='delivered'
    with pytest.raises(ValueError, match='expected release lock'):
        u.deliver_one(api_fixture([pr],lock='changed'),consumer(),lock,register=True)


def test_dry_run_registration_has_no_writes():
    result=u.deliver_one(api_fixture(),consumer(),'{"tag":"v0.1.0"}',register=True,dry_run=True)
    assert result['status']=='dry-run' and result['action']=='create PR'


def test_human_modified_branch_is_conflict():
    with pytest.raises(ValueError, match='human-modified'):
        u.deliver_one(api_fixture(branch={'object':{'sha':'head'}},modified=True),consumer(),'{"tag":"v0.1.0"}',register=True,dry_run=True)


def test_normal_delivery_requires_explicit_registration():
    with pytest.raises(ValueError, match='initial registration required'):
        u.deliver_one(api_fixture(),consumer(),'{"tag":"v0.1.0"}',dry_run=True)


def test_403_configuration_failure_never_retried(monkeypatch):
    calls=[]
    def run(*args,**kwargs):
        calls.append(args)
        return u.subprocess.CompletedProcess(args,1,'','gh: Forbidden (HTTP 403)')
    monkeypatch.setattr(u.subprocess,'run',run)
    with pytest.raises(RuntimeError,match='403'):
        u.API()('repos/bjoernellens1/locus')
    assert len(calls)==1


def test_429_and_5xx_retry_bounded(monkeypatch):
    outcomes=iter([429,500,200]); sleeps=[]
    def run(*args,**kwargs):
        code=next(outcomes)
        return u.subprocess.CompletedProcess(args,0 if code==200 else 1,'{}' if code==200 else '',f'gh: Retry-After: 3 (HTTP {code})')
    monkeypatch.setattr(u.subprocess,'run',run)
    monkeypatch.setattr(u.time,'sleep',sleeps.append)
    api=u.API(); assert api('repos/bjoernellens1/locus')=={}
    assert api.attempts==3 and sleeps==[3,3]


def test_generated_verifier_rejects_bad_schema_without_network(tmp_path):
    script=tmp_path/'verify.py'; script.write_text(u.VERIFIER)
    lock=tmp_path/'lock.json'; lock.write_text('{}')
    result=u.subprocess.run(['python3',str(script),'--lock',str(lock)],capture_output=True,text=True)
    assert result.returncode != 0 and 'schema/repository' in result.stderr


class FakeGitHub:
    """Small in-memory Git database; no network or credentials."""
    def __init__(self, ambiguous=False):
        self.branch=None; self.pr=None; self.files={}; self.writes=[]; self.ambiguous=ambiguous

    def __call__(self, endpoint, method='GET', data=None, missing=False):
        if method != 'GET': self.writes.append((method,endpoint))
        if endpoint.endswith('/locus'): return {'default_branch':'main'}
        if '/pulls?' in endpoint: return [self.pr] if self.pr else []
        if endpoint.endswith('/git/ref/heads/main'): return {'object':{'sha':'base'}}
        if '/git/ref/heads/sintergpu/' in endpoint: return {'object':{'sha':self.branch}} if self.branch else None
        if '/contents/' in endpoint:
            name=endpoint.split('/contents/')[1].split('?')[0]
            value=self.files.get(name) if '?ref=commit' in endpoint else None
            return {'content':u.base64.b64encode(value.encode()).decode()} if value else None
        if '/commits/commit' in endpoint:
            return {'commit':{'message':'sintergpu: pin v0.1.0'},'files':[{'filename':k} for k in self.files]}
        if endpoint.endswith('/git/trees'):
            self.files={entry['path']:entry['content'] for entry in data['tree']}; return {'sha':'tree'}
        if endpoint.endswith('/git/commits'): return {'sha':'commit'}
        if endpoint.endswith('/git/refs'): self.branch=data['sha']; return {'object':{'sha':self.branch}}
        if endpoint.endswith('/pulls'):
            self.pr={'state':'open','merged_at':None,'body':data['body'],'head':{'sha':'commit','repo':{'full_name':'bjoernellens1/locus'}},'html_url':'https://github.com/bjoernellens1/locus/pull/1','number':1}
            if self.ambiguous: raise RuntimeError('ambiguous create response')
            return self.pr
        raise AssertionError((endpoint,method))


@pytest.mark.parametrize('ambiguous',[False,True])
def test_delivery_retry_creates_zero_extra_prs_or_commits(monkeypatch, ambiguous):
    # Release integrity is verified independently; isolate only PR/Git lifecycle.
    monkeypatch.setattr(u.subprocess,'run',lambda *args,**kwargs: None)
    api=FakeGitHub(ambiguous)
    lock='{"tag":"v0.1.0"}'
    first=u.deliver_one(api,consumer(),lock,register=True)
    writes=len(api.writes)
    second=u.deliver_one(api,consumer(),lock,register=True)
    assert first['url']==second['url']
    assert len(api.writes)==writes
    assert sum(endpoint.endswith('/pulls') for method,endpoint in api.writes)==1
    assert sum(endpoint.endswith('/git/commits') for method,endpoint in api.writes)==1


def test_normal_retry_reuses_pending_registration_without_writes(monkeypatch):
    monkeypatch.setattr(u.subprocess,'run',lambda *args,**kwargs: None)
    api=FakeGitHub()
    first=u.deliver_one(api,consumer(),'{"tag":"v0.1.0"}',register=True)
    writes=len(api.writes)
    second=u.deliver_one(api,consumer(),'{"tag":"v0.1.0"}')
    assert first['url']==second['url'] and len(api.writes)==writes


def test_ambiguous_write_is_not_blindly_retried(monkeypatch):
    calls=[]
    def run(*args,**kwargs):
        calls.append(args)
        return u.subprocess.CompletedProcess(args,1,'','gh: server error (HTTP 500)')
    monkeypatch.setattr(u.subprocess,'run',run)
    with pytest.raises(RuntimeError): u.API()('repos/bjoernellens1/locus/pulls','POST',{})
    assert len(calls)==1


def test_registry_cannot_add_arbitrary_destination(tmp_path):
    import json
    data=json.loads((Path(__file__).resolve().parents[1]/'consumers.json').read_text())
    data['consumers'][0]['repository']='attacker/unapproved'
    path=tmp_path/'registry.json'; path.write_text(json.dumps(data))
    with pytest.raises(ValueError,match='seven approved'): u.validate_registry(path)


def valid_lock():
    tag='v0.1.0'; prefix='https://github.com/bjoernellens1/sintergpu/releases/download/'+tag+'/'
    return {'schema_version':1,'repository':'bjoernellens1/sintergpu','tag':tag,'version':'0.1.0','source_commit':'a'*40,
            'shader_tree_sha256':'b'*64,'manifest':{'name':'manifest.json','url':prefix+'manifest.json','sha256':'c'*64},
            'artifact':{'name':'source.tar.gz','url':prefix+'source.tar.gz','sha256':'d'*64},
            'integration_state':'registered','consumed_modules':[],'migration':'pending'}


@pytest.mark.parametrize('field,value', [('source_commit','main'),('version','0.2.0'),('shader_tree_sha256','bad')])
def test_lock_rejects_invalid_source_version_or_hash(field,value):
    lock=valid_lock(); u.validate_lock(lock); lock[field]=value
    with pytest.raises(ValueError): u.validate_lock(lock)


@pytest.mark.parametrize('field,value', [('name','../source.tar.gz'),('url','https://attacker.invalid/source.tar.gz'),('sha256','bad')])
def test_lock_rejects_unsafe_artifact_pin(field,value):
    lock=valid_lock(); lock['artifact'][field]=value
    with pytest.raises(ValueError): u.validate_lock(lock)


def test_delivery_aggregates_all_seven_after_failure(tmp_path,monkeypatch):
    import json
    results=tmp_path/'results.json'
    registry=Path(__file__).resolve().parents[1]/'consumers.json'
    monkeypatch.setattr(u.sys,'argv',['update_consumers.py','--tag','v0.1.0','--registry',str(registry),'--deliver','--results',str(results)])
    monkeypatch.setattr(u.release,'verify_download',lambda tag,path:{'tag':tag})
    monkeypatch.setattr(u,'make_lock',lambda manifest,path,item:'lock')
    monkeypatch.setattr(u,'release_context',lambda manifest,path:'- Verified release notes.')
    visited=[]
    def deliver(api,item,*args):
        visited.append(item['repository'])
        if item['repository'].endswith('/stereoforge'): raise RuntimeError('App authorization unavailable')
        return {'repository':item['repository'],'status':'delivered','consumer_checks':'pending'}
    monkeypatch.setattr(u,'deliver_one',deliver)
    assert u.main()==1
    assert set(visited)==u.ALLOWED
    report=json.loads(results.read_text())
    assert len(report['results'])==7
    assert sum(x['status']=='failed' for x in report['results'])==1


def release_bundle(tmp_path, changelog=None):
    import io,tarfile,json
    version='0.1.4';path=tmp_path/f'sintergpu-{version}-source.tar.gz'
    text=changelog or '# Changelog\n\n## 0.1.4 — 2026-10-27\n\n- Add conservative depth filling.\n- Preserve initial direct masks.\n\n## 0.1.3 — 2026-10-27\n\n- Older unrelated change.\n'
    with tarfile.open(path,'w:gz') as archive:
        data=text.encode();entry=tarfile.TarInfo(f'sintergpu-{version}/CHANGELOG.md');entry.size=len(data);archive.addfile(entry,io.BytesIO(data))
    return {'tag':'v'+version,'version':version,'assets':[{'name':path.name,'sha256':u.release.sha256(path),'size':path.stat().st_size}]}


def test_release_context_uses_hash_verified_exact_version_changelog(tmp_path):
    manifest=release_bundle(tmp_path)
    context=u.release_context(manifest,tmp_path)
    assert 'Add conservative depth filling.' in context
    assert 'Older unrelated change' not in context
    (tmp_path/manifest['assets'][0]['name']).write_bytes(b'tampered')
    with pytest.raises(ValueError,match='source'):
        u.release_context(manifest,tmp_path)


@pytest.mark.parametrize('section',['## 0.1.4 — unreleased\n\n- Not published.\n','## 0.1.3 — 2026-10-27\n\n- Wrong version.\n'])
def test_release_context_rejects_missing_or_unreleased_notes(tmp_path,section):
    with pytest.raises(ValueError,match='changelog'):
        u.release_context(release_bundle(tmp_path,section),tmp_path)


def test_release_body_gives_registered_and_integrated_consumers_actionable_scope():
    item=consumer();notes='- Add conservative depth filling.'
    body=u.release_body(item,'v0.1.4',notes)
    assert notes in body and 'Registration only' in body
    assert 'No kernel code is migrated by this PR' in body
    assert 'docs/vkl' not in body # No invented per-consumer operation claims.
    item.update(integration_state='integrated',consumed_modules=['reductions/float_sum_256.slang'],integration_argv=['python','-m','pytest','tests/test_reductions.py'])
    body=u.release_body(item,'v0.1.4',notes)
    assert 'reductions/float_sum_256.slang' in body and 'tests/test_reductions.py' in body
    assert 'No automatic merge' in body and 'GPU qualification' in body


def test_delivery_enriched_notes_idempotent_without_extra_commit(monkeypatch):
    monkeypatch.setattr(u.subprocess,'run',lambda *args,**kwargs:None)
    api=FakeGitHub();lock='{"tag":"v0.1.0"}'
    first=u.deliver_one(api,consumer(),lock,register=True,release_notes='- New release detail.')
    writes=list(api.writes)
    second=u.deliver_one(api,consumer(),lock,register=True,release_notes='- New release detail.')
    assert first['head_sha']==second['head_sha'] and api.writes==writes
    assert '- New release detail.' in api.pr['body']


def test_exact_legacy_body_can_upgrade_once_but_human_edits_are_preserved(monkeypatch):
    monkeypatch.setattr(u.subprocess,'run',lambda *args,**kwargs:None)
    api=FakeGitHub();lock='{"tag":"v0.1.0"}'
    u.deliver_one(api,consumer(),lock,register=True)
    original=api.pr['body'];writes=list(api.writes)
    def update(endpoint,method='GET',data=None,missing=False):
        if method=='PATCH':
            assert endpoint.endswith('/pulls/1') and list(data)==['body']
            api.writes.append((method,endpoint));api.pr['body']=data['body'];return api.pr
        return api(endpoint,method,data,missing)
    u.deliver_one(update,consumer(),lock,register=True,release_notes='- New release detail.')
    assert api.writes==writes+[('PATCH','repos/bjoernellens1/locus/pulls/1')]
    u.deliver_one(update,consumer(),lock,register=True,release_notes='- New release detail.')
    assert len(api.writes)==len(writes)+1
    api.pr['body']=original+'\nHuman comment: retain me.\n';before=list(api.writes)
    with pytest.raises(ValueError,match='human-modified'):
        u.deliver_one(update,consumer(),lock,register=True,release_notes='- New release detail.')
    assert api.pr['body'].endswith('Human comment: retain me.\n') and api.writes==before


def test_ambiguous_legacy_body_upgrade_reads_confirmation_without_retry(monkeypatch):
    monkeypatch.setattr(u.subprocess,'run',lambda *args,**kwargs:None)
    api=FakeGitHub();lock='{"tag":"v0.1.0"}';u.deliver_one(api,consumer(),lock,register=True)
    updates=[]
    def ambiguous(endpoint,method='GET',data=None,missing=False):
        if endpoint.endswith('/pulls/1'):
            updates.append(method)
            if method=='PATCH':
                api.pr['body']=data['body'];raise RuntimeError('response lost after applying')
            return api.pr
        return api(endpoint,method,data,missing)
    result=u.deliver_one(ambiguous,consumer(),lock,register=True,release_notes='- Verified detail.')
    assert result['status']=='delivered' and updates==['PATCH','GET']


def test_unverified_release_notes_prevent_all_delivery_writes(tmp_path,monkeypatch):
    import json
    result=tmp_path/'delivery.json'
    monkeypatch.setattr(u.sys,'argv',['update_consumers.py','--tag','v0.1.0','--registry',str(Path(__file__).resolve().parents[1]/'consumers.json'),'--deliver','--results',str(result)])
    monkeypatch.setattr(u.release,'verify_download',lambda *args:{'tag':'v0.1.0'})
    def fail(*args):raise ValueError('verified release changelog unavailable')
    monkeypatch.setattr(u,'release_context',fail)
    monkeypatch.setattr(u,'deliver_one',lambda *args:pytest.fail('delivery before notes verification'))
    assert u.main()==1
    records=json.loads(result.read_text())['results']
    assert len(records)==7 and all(r['status']=='failed' and 'changelog' in r['error'] for r in records)
