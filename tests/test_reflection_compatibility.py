"""The compiler matrix must agree on reflected bindings and numerical layouts."""
import copy
import importlib.util
from pathlib import Path
import pytest

ROOT=Path(__file__).resolve().parents[1]


def checker():
    path=ROOT/'tools/compare_reflection.py'
    assert path.is_file(),'cross-compiler reflection/layout gate missing'
    spec=importlib.util.spec_from_file_location('reflection_compatibility',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def records():
    reflection={'parameters':[{'name':'params','binding':{'kind':'descriptorTableSlot','index':0},'type':{'kind':'constantBuffer','elementType':{'kind':'struct','fields':[{'name':'count','type':{'kind':'scalar','scalarType':'uint32'},'binding':{'kind':'uniform','offset':0,'size':4}},{'name':'scale','type':{'kind':'scalar','scalarType':'float32'},'binding':{'kind':'uniform','offset':4,'size':4}}]}}}], 'entryPoints':[{'name':'main','stage':'compute','threadGroupSize':[256,1,1]}]}
    def record(version):
        return {'status':'passed','version':'0.1.5','target':'spirv_1_5','pinned_release':version,'compiler_version':version,'compiler_sha256':'a'*64,'entries':[{'operation':'test.op','source_sha256':'b'*64,'returncode':0,'reflection':copy.deepcopy(reflection)}]}
    return record('2026.12'),record('2026.19'),{'test.op':'b'*64}


def test_reflection_new_metadata_is_explicitly_scoped():
    left,right,catalog=records();r=right['entries'][0]['reflection']
    r['version']='1.1';r['globalScope']={};r['entryPoints'][0]['scope']={'kind':'none','parameters':[]}
    r['parameters'][0]['type']['elementType']['fields'][0]['type']['sizes']=[{'kind':'uniform','value':4,'alignment':4}]
    result=checker().compare_records(left,right,catalog)
    assert result['status']=='passed' and result['entries']==1
    assert result['scope'].startswith('All 2026.12 reflection fields')


@pytest.mark.parametrize('mutation',['offset','size','scalar','binding','order','threads','missing','unexpected','entrycount','source','compiler','target','version'])
def test_reflection_matrix_rejects_abi_and_identity_mutations(mutation):
    left,right,catalog=records();r=right['entries'][0]['reflection'];fields=r['parameters'][0]['type']['elementType']['fields']
    if mutation=='offset':fields[1]['binding']['offset']=8
    elif mutation=='size':fields[1]['binding']['size']=8
    elif mutation=='scalar':fields[1]['type']['scalarType']='float64'
    elif mutation=='binding':r['parameters'][0]['binding']['index']=1
    elif mutation=='order':fields.reverse()
    elif mutation=='threads':r['entryPoints'][0]['threadGroupSize']=[128,1,1]
    elif mutation=='missing':del fields[0]['binding']['offset']
    elif mutation=='unexpected':r['parameters'][0]['binding']['newLayoutRule']=1
    elif mutation=='entrycount':right['entries'].append(copy.deepcopy(right['entries'][0]))
    elif mutation=='source':right['entries'][0]['source_sha256']='c'*64
    elif mutation=='compiler':right['compiler_version']='2026.12'
    elif mutation=='target':right['target']='spirv_1_6'
    elif mutation=='version':right['version']='0.1.6'
    with pytest.raises(ValueError):checker().compare_records(left,right,catalog)


def test_ci_requires_both_compiler_artifacts_for_layout_gate():
    workflow=(ROOT/'.github/workflows/ci.yml').read_text()
    assert 'reflection-compatibility:' in workflow
    job=workflow.split('  reflection-compatibility:',1)[1]
    assert 'needs: compiler' in job
    for pin in ('2026.12','2026.19'):assert 'name: compiler-'+pin in job
    assert 'python tools/compare_reflection.py --left' in job
    assert 'name: reflection-compatibility' in job
