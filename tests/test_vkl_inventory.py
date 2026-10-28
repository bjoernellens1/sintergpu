"""The candidate inventory must cover the actual frozen public source surface."""
import ast
import hashlib
import json
from pathlib import Path


def test_every_frozen_public_candidate_has_explicit_disposition_and_lineage():
    from sintergpu.catalog import get_operation
    root=Path(__file__).resolve().parents[1]
    inventory=json.loads((root/'provenance/vkl-candidates.json').read_text())
    modules={m['module']:m for m in inventory['modules']}
    paths=list((root/'references/donors/vkl/src/vkl/kernels/reference').glob('*.py'))
    assert set(modules)=={p.stem for p in paths if p.stem!='__init__'}
    for path in paths:
        if path.stem=='__init__':continue
        module=modules[path.stem]
        assert hashlib.sha256(path.read_bytes()).hexdigest()==module['source_sha256']
        symbols={n.name:n.lineno for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and not n.name.startswith('_')}
        assert {c['symbol']:c['source_line'] for c in module['candidates']}==symbols
        for c in module['candidates']:
            assert c['disposition'] in {'consolidated','specialized_variant','dependency','reference-only','deferred'}
            assert c['reason'] and c['numerical_contract']
            if c['disposition']=='specialized_variant':
                assert c['slang_operations']
                for op in c['slang_operations']:
                    record=get_operation(op)
                    assert record['source']['revision']==inventory['revision']


def test_documented_candidate_dispositions_match_the_provenance_inventory():
    root=Path(__file__).resolve().parents[1]
    inventory=json.loads((root/'provenance/vkl-candidates.json').read_text())
    rows={}
    for line in (root/'docs/vkl-source-inventory.md').read_text().splitlines():
        if line.startswith('| `'):
            fields=line.split('|')
            rows[fields[1].strip().strip('`')]=fields[2].strip()
    expected={m['module']+'.'+c['symbol']:c['disposition'] for m in inventory['modules'] for c in m['candidates']}
    assert rows==expected


def test_camera_registration_catalog_profiles_and_provenance():
    from sintergpu.catalog import get_operation
    from sintergpu import module_path
    root = Path(__file__).resolve().parents[1]
    ledger = json.loads((root / 'provenance/vkl-camera-registration.json').read_text())
    shader = next(f for f in ledger['files'] if f['destination'].endswith('/vkl/camera_registration.slang'))
    assert ledger['coverage']['public_functions'] == 5
    assert ledger['coverage']['entrypoints'] == len(shader['entry_points']) == 9
    assert len(set(shader['entry_points'])) == 9
    assert module_path('common.brown_conrady').is_file()
    for entry in shader['entry_points']:
        operation = get_operation('vkl.camera_registration.' + entry)
        contract = operation['contract']
        assert operation['qualification']['status'] == 'experimental'
        assert operation['source']['revision'] == ledger['revision']
        assert operation['sha256'] == shader['imported_sha256']
        assert contract['profile'] == 'vkl-camera-registration-eager-float32'
        assert contract['documentation'].endswith('/docs/vkl-camera-registration.md')
        for term in ('float32', 'uint32', 'capacities', 'little-endian', '8-byte'):
            assert term in contract['numerical_behavior']
