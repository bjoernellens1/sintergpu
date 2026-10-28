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


def test_current_qualification_timing_coverage_matches_source_bound_database():
    root=Path(__file__).resolve().parents[1]
    qualification=json.loads((root/'qualification'/(__version__+'.json')).read_text())
    timing=qualification['timing_reference']
    database=json.loads((root/timing['database']).read_text())
    assert timing['catalog_entries']==database['coverage']['catalog_entries']
    assert timing['paired_cpu_gpu_entries']==database['coverage']['both_measured']
    assert timing['pending_portable_entries']==database['coverage']['pending']
    assert timing['workloads']==sum(row['current_source'] for row in database['measurements'])
    assert timing['retained_historical_workloads']==sum(not row['current_source'] for row in database['measurements'])
    for name in (timing['database'],timing['table'],*timing['raw_samples']):
        assert qualification['evidence_sha256'][name]==hashlib.sha256((root/name).read_bytes()).hexdigest(),name
