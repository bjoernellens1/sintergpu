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
