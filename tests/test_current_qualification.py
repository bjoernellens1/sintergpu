"""Current-version qualification cannot inherit a different kernel source hash."""
import json
import hashlib
from pathlib import Path
from sintergpu import __version__
from sintergpu.catalog import get_operation


def test_current_qualification_records_match_the_packaged_operation_sources():
    root=Path(__file__).resolve().parents[1]
    record=json.loads((root/'qualification'/(__version__+'.json')).read_text())
    assert record['version']==__version__
    for entry in record['qualifications']:
        assert entry['source_sha256']==get_operation(entry['operation'])['sha256'],entry['operation']
        for name,digest in entry.get('source_dependencies_sha256',{}).items():
            assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest,name
