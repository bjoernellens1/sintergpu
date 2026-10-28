# SPDX-License-Identifier: Apache-2.0
"""Packaged experimental compact assets; compilation is not GPU qualification."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import pytest
from sintergpu import kernel_path,module_path
from sintergpu.catalog import get_operation,validate_catalog

ROOT=Path(__file__).resolve().parents[1]
FORWARD='compact_batched'; BACKWARD='compact_backward_torch_cpu_profile'
ENTRIES={
 FORWARD:{'compactInit':{'params','slotToSource','indicesOut','validOut','actionsOut','countsOut'},
 'compactFlags':{'params','maskWords','flags','segmentIds'},
 'compactScatterMetadata':{'params','maskWords','actionWords','prefix','slotToSource','indicesOut','validOut','actionsOut'},
 'compactFinalizeCounts':{'params','prefix','countsOut'},
 'compactGatherWords':{'params','valueWords','shapeStrides','slotToSource','valuesOut'}},
 BACKWARD:{'backwardProfileInit':{'params','inverse'},'backwardProfileInverse':{'params','slotSource','inverse'},
 'backwardProfileGather':{'params','inverse','gradWords','outputWords'}}}

@pytest.mark.parametrize('module,candidate',[(FORWARD,'vkl_compact_candidate.slang'),(BACKWARD,'vkl_compact_backward_torch_cpu_profile.slang')])
def test_packaged_bytes_catalog_and_provenance(module,candidate):
    packaged=module_path('vkl.'+module); source=ROOT/'benchmarks/shaders'/candidate
    assert packaged.read_bytes()==source.read_bytes()
    manifest=json.loads((ROOT/'provenance/vkl-compact-packaged.json').read_text())
    record=next(x for x in manifest['files'] if x['destination'].endswith('/'+module+'.slang'))
    assert record['imported_sha256']==record['candidate_sha256']==hashlib.sha256(packaged.read_bytes()).hexdigest()
    assert set(record['entry_points'])==set(ENTRIES[module])
    assert manifest['dependencies'][0]['license']=='MIT'
    for notice in manifest['dependencies'][0]['notices']: assert (ROOT/notice).is_file()
    for entry in ENTRIES[module]:
        op=get_operation(f'vkl.{module}.{entry}')
        assert kernel_path(op['id'])==packaged
        assert op['source']['revision']=='3384023a311651ea4260cf12ddab1567972f89f3'
        assert op['source']['paths']==['src/vkl/kernels/reference/compact.py']
        assert op['qualification']=={'status':'experimental','records':[]}
    assert not validate_catalog()

@pytest.mark.parametrize('module,entry',[(module,entry) for module,entries in ENTRIES.items() for entry in entries])
@pytest.mark.parametrize('version',['2026.12','2026.19'])
def test_packaged_shader_compiles_and_reflects(module,entry,version,tmp_path):
    key='SINTERGPU_SLANGC_'+version.replace('.','_')
    compiler=Path(os.environ.get(key,ROOT/'.tools'/('slang-'+version)/'bin/slangc'))
    if not compiler.is_file():
        if key in os.environ: pytest.fail('explicit compiler missing')
        pytest.skip('compiler unavailable')
    identity=subprocess.run([str(compiler),'-version'],capture_output=True,text=True,check=True)
    assert (identity.stdout+identity.stderr).strip()==version
    output=tmp_path/'shader.spv'; refl=tmp_path/'reflection.json'
    run=subprocess.run([str(compiler),str(module_path('vkl.'+module)),'-entry',entry,'-target','spirv','-profile','glsl_460','-o',str(output),'-reflection-json',str(refl)],capture_output=True,text=True)
    assert run.returncode==0,run.stderr
    data=json.loads(refl.read_text()); ep=data['entryPoints'][0]
    assert ep['threadGroupSize']==[256,1,1]
    assert {x['name'] for x in ep['bindings'] if x['binding'].get('used')==1}==ENTRIES[module][entry]
    fields=next(x for x in data['parameters'] if x['name']=='params')['type']['elementType']['fields']
    names=(['batches','items','capacity','rank','itemBytes','payloadItems','logicalValueBytes','actionBytes','valueBase','maskBase','maskBatchStride','maskItemStride','actionBase','actionBatchStride','actionItemStride'] if module==FORWARD else ['batches','items','capacity','bytesPerItem','logicalBytes','gradBase','componentBytes'])
    assert [x['name'] for x in fields]==names
    assert [x['binding']['offset'] for x in fields]==list(range(0,len(names)*4,4))
    assert all(x['type']['scalarType']=='uint32' for x in fields) and output.stat().st_size>0
