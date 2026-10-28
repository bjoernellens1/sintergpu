# SPDX-License-Identifier: Apache-2.0
"""Compile every candidate profile and baseline in the two pinned Slang lanes."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess

import family_support as support


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sources():
    # Harness construction is intentionally separable from GPU allocation.
    original = support.make_case
    support.make_case = lambda device, code, *a, **kw: dict(sources=code)
    result = []
    try:
        for n, schur in ((6,False),(6,True),(12,True),(18,True)):
            case = support.solve_case(None,n,schur=schur)
            for variant, source in case['sources'].items():
                result.append((f'solve{n}-'+('schur' if schur else 'caspar')+'-'+variant,source,'main'))
        for profile in ('aosoa','scalar'):
            case = support.point_case(None,1025,profile)
            for variant, source in case['sources'].items(): result.append(('point-'+profile+'-'+variant,source,'main'))
    finally: support.make_case = original
    fused=(support.CANDIDATE/'kernels/optimization/point_plane_gn.slang').read_text()
    for entry in ('pointPlaneGNWorkgroup','pointPlaneGNSubgroup'): result.append((entry,fused,entry))
    result.append(('import-helpers',(support.CANDIDATE/'examples/import_helpers.slang').read_text(),'main'))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--toolchains',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    records=[]; catalog={}
    for version in ('2026.12','2026.19'):
        compiler=args.toolchains/version/'bin/slangc'
        identity=subprocess.run([str(compiler),'-version'],text=True,capture_output=True,check=True)
        record=dict(version='unpromoted-candidate',pinned_release=version,
            compiler_version=(identity.stdout+identity.stderr).strip(),compiler_sha256=sha(compiler),
            target='spirv_1_5',entries=[],gpu_execution='not performed by this standalone compiler lane')
        directory=args.output/version;directory.mkdir(parents=True,exist_ok=True)
        for name,source,entry in sources():
            path=directory/(name+'.slang');path.write_text(source)
            spv=path.with_suffix('.spv');reflect=path.with_suffix('.reflection.json')
            command=[str(compiler),str(path),'-entry',entry,'-target','spirv','-profile','spirv_1_5',
                '-fp-mode','precise','-O3','-I',str(support.CANDIDATE/'kernels'),
                '-I',str(support.CANDIDATE/'kernels/optimization'),'-o',str(spv),'-reflection-json',str(reflect)]
            result=subprocess.run(command,capture_output=True,text=True)
            item=dict(operation=name,source_sha256=sha(path),entry=entry,command=command,
                returncode=result.returncode,diagnostics=result.stdout+result.stderr)
            if result.returncode==0:
                item.update(spirv_sha256=sha(spv),reflection=json.loads(reflect.read_text()))
            record['entries'].append(item);catalog[name]=sha(path)
        record['status']='passed' if all(item['returncode']==0 for item in record['entries']) else 'failed'
        (directory/'results.json').write_text(json.dumps(record,indent=2)+'\n');records.append(record)
        print(version,record['status'],len(record['entries']),flush=True)
    # Apply the existing library's strict schema-addition allowlist, without
    # changing its catalog, records, release gates or installed code.
    spec=importlib.util.spec_from_file_location('compare_reflection',support.ROOT/'tools/compare_reflection.py')
    compare=importlib.util.module_from_spec(spec);spec.loader.exec_module(compare)
    additions=compare.compare_records(*records,catalog)
    (args.output/'compatibility.json').write_text(json.dumps(dict(status='passed',schema_additions=additions),indent=2)+'\n')


if __name__=='__main__': main()
