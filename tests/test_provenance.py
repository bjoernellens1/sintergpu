import importlib.util
import json
from pathlib import Path

spec=importlib.util.spec_from_file_location('check_provenance',Path(__file__).resolve().parents[1]/'tools/check_provenance.py')
p=importlib.util.module_from_spec(spec); spec.loader.exec_module(p)


def test_repository_source_ledger_is_current():
    assert p.check()==[]


def test_generated_and_audit_inputs_cannot_silently_change(tmp_path):
    (tmp_path/'provenance').mkdir()
    (tmp_path/'shader.slang').write_bytes(b'changed')
    (tmp_path/'provenance/generated.json').write_text(json.dumps({'generation':{'outputs':[{'path':'shader.slang','sha256':'0'*64}]},'audit_inputs_sha256':{'shader.slang':'0'*64}}))
    errors=p.check(tmp_path)
    assert len(errors)==2 and all('modified shader.slang' in e for e in errors)
