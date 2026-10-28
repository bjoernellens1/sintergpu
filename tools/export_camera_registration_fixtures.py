#!/usr/bin/env python3
"""Export an immutable, CPU-only camera corpus from the unchanged pinned donor."""
import argparse
import hashlib
import importlib
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np

from benchmarks.camera_registration_fixture_archive import PROFILE, REVISION, ROOT, payload_identity, source_identities
from benchmarks.camera_registration_fixtures import output_schema, payloads
from tools.export_reprojection_fixtures import _runtime_profile, _frozen_module, _file_sha, _publish


def _outputs(payload, torch, root):
    kw = {k: torch.from_numpy(v.copy()) if isinstance(v, np.ndarray) else v for k,v in payload['kwargs'].items()}
    result = getattr(_frozen_module('camera_registration',root),payload['function'])(**kw)
    if not isinstance(result, dict):
        result = {'result': result}
    return {k:v.detach().cpu().numpy().copy() for k,v in result.items()}


def _reference(payload):
    module = importlib.import_module('sintergpu.references.camera_registration_numeric')
    result = getattr(module,payload['function'])(**payload['kwargs'])
    return result if isinstance(result,dict) else {'result':result}


def _validate(payload, result, reference):
    schema = output_schema(payload)
    if set(result) != set(schema) or set(reference) != set(schema):
        raise ValueError('incomplete output schema')
    for name, (dtype, shape) in schema.items():
        actual, expected = result[name], reference[name]
        if actual.dtype.str != dtype or expected.dtype.str != dtype or actual.shape != shape or expected.shape != shape:
            raise ValueError('output dtype or shape mismatch: '+name)
        if actual.dtype.kind == 'f':
            np.testing.assert_allclose(actual,expected,rtol=8e-6,atol=8e-7,equal_nan=True,err_msg=name)
        else:
            np.testing.assert_array_equal(actual,expected,err_msg=name)


def export_archive(destination, *, root=ROOT, torch_module=None):
    destination=Path(destination).absolute()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError('refusing to overwrite fixture archive')
    torch = torch_module or importlib.import_module('torch')
    runtime=_runtime_profile(torch)
    source=source_identities(root)
    destination.parent.mkdir(parents=True,exist_ok=True)
    staging=Path(tempfile.mkdtemp(prefix='.'+destination.name+'-',dir=destination.parent))
    try:
        records=[]; identities=set(); case_ids=set()
        for payload in payloads():
            try:
                result=_outputs(payload,torch,root)
                _validate(payload,result,_reference(payload))
            except Exception as exc:
                raise RuntimeError('fixture export failed: '+payload['case_id']) from exc
            identity=payload_identity(payload)
            if identity['sha256'] in identities or payload['case_id'] in case_ids:
                raise ValueError('duplicate fixture identity or case ID')
            identities.add(identity['sha256']); case_ids.add(payload['case_id'])
            path=staging/(identity['sha256']+'.npz')
            np.savez_compressed(path,**result)
            outputs={k:dict(dtype=v.dtype.str,shape=list(v.shape),sha256=hashlib.sha256(v.tobytes(order='C')).hexdigest()) for k,v in result.items()}
            records.append(dict(identity=identity,file=path.name,sha256=_file_sha(path),outputs=outputs))
        if not records or source_identities(root) != source:
            raise ValueError('empty corpus or changed source closure')
        manifest=dict(schema_version=1,counts=dict(cases=len(case_ids),states=len(records)),profile=dict(name=PROFILE,frozen_revision=REVISION,source_sha256=source,runtime=runtime),records=records)
        (staging/'manifest.json').write_text(json.dumps(manifest,sort_keys=True,indent=2,allow_nan=False)+'\n')
        _publish(staging,destination)
        return manifest
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path)
    args=parser.parse_args()
    result=export_archive(args.output)
    print(f"Exported {len(result['records'])} camera fixture states to {args.output}")


if __name__=='__main__':
    main()
