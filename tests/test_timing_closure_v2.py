# SPDX-License-Identifier: Apache-2.0
"""Provisional v2 closure policies; does not qualify measurements."""
import copy
from pathlib import Path
import shutil
import types
import pytest
import sintergpu.timing_closure_v2 as closure
from sintergpu.timing_closure_v2 import POLICIES,build_manifest,validate_manifest,require_collection_ready

ROOT=Path(__file__).resolve().parents[1]
FAMILY='vkl.depth_convert.depthConvert'
POLICY=POLICIES[FAMILY]

@pytest.fixture
def sandbox(tmp_path,monkeypatch):
    for name in POLICY['files'].values():
        src=ROOT/name; dest=tmp_path/name; dest.parent.mkdir(parents=True,exist_ok=True); shutil.copyfile(src,dest)
    modules={name:types.SimpleNamespace(__file__=str(tmp_path/path)) for name,path in POLICY['required_modules'].items()}
    monkeypatch.setattr(closure,'sys',types.SimpleNamespace(modules=modules))
    return tmp_path,modules


def make(sandbox):
    root,modules=sandbox
    return build_manifest(root,FAMILY,compiler=dict(POLICY['compiler']))


def test_deterministic_manifest_and_unrelated_mutation(sandbox):
    root,modules=sandbox; a=make(sandbox); b=make(sandbox)
    assert a==b and validate_manifest(root,a)==a['sha256']
    assert a['schema_version']==2 and len(a['sha256'])==64
    assert a['status']=='provisional_not_for_collection'
    with pytest.raises(ValueError,match='cannot authorize'): require_collection_ready(a)
    extra=root/'unrelated'/'file'; extra.parent.mkdir(); extra.write_text('changed')
    assert make(sandbox)==a
    unrelated=root/'src/sintergpu/references/vkl/brief.py'
    unrelated.write_text('unused module changed')
    assert make(sandbox)==a

@pytest.mark.parametrize('role',['shader','runner','fixture','reference.vkl.numpy','adapter','setup','version','catalog_data'])
def test_required_source_mutation_invalidates_only_affected_identity(sandbox,role):
    root,modules=sandbox; before=make(sandbox); path=root/POLICY['files'][role]
    path.write_bytes(path.read_bytes()+b'\n')
    with pytest.raises(ValueError,match='current bytes|pinned policy'): validate_manifest(root,before)
    if role=='shader':
        with pytest.raises(ValueError,match='pinned policy'): make(sandbox)
        return
    after=make(sandbox); assert before['sha256']!=after['sha256']
    changed=[(x['role']) for x,y in zip(before['files'],after['files']) if x!=y]
    assert changed==[role]

@pytest.mark.parametrize('mutation',['missing','duplicate','extra','badrole','outofroot','digest','order','version','schema','flags'])
def test_rejects_invalid_or_incomplete_manifests(sandbox,mutation):
    root,modules=sandbox; obj=copy.deepcopy(make(sandbox))
    if mutation=='missing': obj['files'].pop()
    elif mutation=='duplicate': obj['files'].append(dict(obj['files'][0]))
    elif mutation=='extra': obj['files'].append({'role':'unknown','path':'extra','sha256':'0'*64})
    elif mutation=='badrole': obj['files'][0]['role']='unknown'
    elif mutation=='outofroot': obj['files'][0]['path']='../outside'
    elif mutation=='digest': obj['sha256']='0'*64
    elif mutation=='order': obj['files'].reverse()
    elif mutation=='version': obj['policy_version']=2
    elif mutation=='schema': obj['schema_version']=1
    elif mutation=='flags': obj['compiler_claim']['floating_point_mode']='fast'
    with pytest.raises(ValueError): validate_manifest(root,obj)


def test_missing_or_symlinked_sources_fail_closed(sandbox,tmp_path):
    root,modules=sandbox; path=root/POLICY['files']['shader']; data=path.read_bytes(); path.unlink()
    with pytest.raises(ValueError,match='missing'): make(sandbox)
    outside=tmp_path.parent/'outside-shader'; outside.write_bytes(data); path.symlink_to(outside)
    with pytest.raises(ValueError,match='symlink'): make(sandbox)


def test_new_shader_import_requires_policy_review(sandbox):
    root,modules=sandbox; path=root/POLICY['files']['shader']
    path.write_bytes(path.read_bytes()+b'\n#include "unknown.slang"\n')
    with pytest.raises(ValueError,match='transitive shader dependency'): make(sandbox)


def test_unregistered_family_or_compiler_flags_fail(sandbox):
    root,modules=sandbox
    with pytest.raises(ValueError,match='unregistered'): build_manifest(root,'invented',compiler={})
    with pytest.raises(ValueError,match='compiler'): build_manifest(root,FAMILY,compiler={'backend':'vulkan'})


def test_loaded_modules_required_and_shadow_rejected(sandbox,tmp_path):
    root,modules=sandbox; absent=dict(modules); absent.pop('sintergpu.vkl')
    closure.sys.modules=absent
    with pytest.raises(ValueError,match='module missing'): build_manifest(root,FAMILY,compiler=POLICY['compiler'])
    shadow=dict(modules); shadow['sintergpu.vkl']=types.SimpleNamespace(__file__=str(tmp_path.parent/'vkl.py'))
    closure.sys.modules=shadow
    with pytest.raises(ValueError,match='path mismatch'): build_manifest(root,FAMILY,compiler=POLICY['compiler'])
    unknown=dict(modules); unknown['sintergpu.references.vkl.unexpected']=types.SimpleNamespace(__file__='unexpected.py')
    closure.sys.modules=unknown
    with pytest.raises(ValueError,match='unknown loaded local module'): build_manifest(root,FAMILY,compiler=POLICY['compiler'])

@pytest.mark.parametrize('name',['benchmarks.unexpected','sintergpu.other_module','sintergpu.adapters.unknown','arbitrary_local_alias'])
def test_unexpected_local_namespace_module_rejected(sandbox,name):
    root,modules=sandbox
    modules[name]=types.SimpleNamespace(__file__=str(root/'unregistered.py'))
    with pytest.raises(ValueError,match='unknown loaded local module'): make(sandbox)


def test_imported_source_mutation_is_provisional_only(sandbox):
    import importlib.util
    root,modules=sandbox; path=root/POLICY['files']['setup']
    path.write_text("VALUE = 'before'\n")
    spec=importlib.util.spec_from_file_location('stale_setup_fixture',path)
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    assert module.VALUE=='before'
    path.write_text("VALUE = 'after!'\n")
    modules['sintergpu.vkl']=module
    manifest=make(sandbox)
    assert module.VALUE=='before' and "after!" in path.read_text()
    assert manifest['status']=='provisional_not_for_collection'
    with pytest.raises(ValueError,match='cannot authorize'): require_collection_ready(manifest)


def test_stale_timestamp_pyc_cannot_authorize_collection(sandbox):
    import importlib.util
    import os
    root,modules=sandbox; path=root/POLICY['files']['setup']
    path.write_text("VALUE = 'before'\n")
    old=path.stat()
    spec=importlib.util.spec_from_file_location('pyc_setup_first',path)
    first=importlib.util.module_from_spec(spec); spec.loader.exec_module(first)
    path.write_text("VALUE = 'after!'\n"); os.utime(path,ns=(old.st_atime_ns,old.st_mtime_ns))
    spec=importlib.util.spec_from_file_location('pyc_setup_second',path)
    stale=importlib.util.module_from_spec(spec); spec.loader.exec_module(stale)
    assert stale.VALUE=='before' and 'after!' in path.read_text()
    modules['sintergpu.vkl']=stale
    manifest=make(sandbox)
    with pytest.raises(ValueError,match='cannot authorize'): require_collection_ready(manifest)
