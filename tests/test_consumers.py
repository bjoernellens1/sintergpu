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
