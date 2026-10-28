# SPDX-License-Identifier: Apache-2.0
"""Compile the ABI-identical inlined point family and compare emitted instructions."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

from inlined_point_support import point_case,CANDIDATE


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def normalized_instructions(text):
    """Ignore only debug names/source and consistently rename SSA IDs.

    Keep executable/type/constant instruction order and every operand. IDs are
    named by definition order, not by annotation emission order. Retain and sort
    all OpDecorate/OpMemberDecorate annotations (their ordering has no semantics).
    Bindings, memory/NoContraction decorations and control flow remain compared.
    """
    ids={};lines=[]
    for line in text.splitlines():
        line=line.split(';')[0].strip()
        if not line or any(op in line for op in ('OpName','OpMemberName','OpSource ')):continue
        lines.append(line)
        match=re.match(r'^(%[A-Za-z_0-9]+)\s*=',line)
        if match:ids[match[1]]='%'+str(len(ids)+1)
    code=[];annotations=[]
    for line in lines:
        def rename(match):
            if match[0] not in ids:raise ValueError('undefined SPIR-V ID '+match[0])
            return ids[match[0]]
        normalized=re.sub(r'%[A-Za-z_0-9]+',rename,line)
        (annotations if line.startswith(('OpDecorate ','OpMemberDecorate ')) else code).append(normalized)
    return '\n'.join(sorted(annotations)+code)+'\n'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--toolchains',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();record=dict(status='passed',comparisons=[])
    for version in ('2026.12','2026.19'):
        compiler=args.toolchains/version/'bin/slangc'
        actual=subprocess.run([str(compiler),'-version'],check=True,capture_output=True,text=True)
        assert (actual.stdout+actual.stderr).strip()==version
        directory=args.output/version;directory.mkdir(parents=True,exist_ok=True)
        for profile in ('aosoa','scalar'):
            case=point_case(None,1025,profile);normalized={};resources={};artifacts={}
            for variant,source in case['sources'].items():
                path=directory/(profile+'-'+variant+'.slang');path.write_text(source)
                common=[str(compiler),str(path),'-entry','main','-profile','spirv_1_5','-fp-mode','precise','-O3']
                binary=path.with_suffix('.spv');reflection=path.with_suffix('.reflection.json')
                subprocess.run([*common,'-target','spirv','-o',str(binary),'-reflection-json',str(reflection)],check=True,capture_output=True,text=True)
                assembly=path.with_suffix('.spvasm')
                subprocess.run([*common,'-target','spirv-asm','-o',str(assembly)],check=True,capture_output=True,text=True)
                normalized[variant]=normalized_instructions(assembly.read_text())
                normal=path.with_suffix('.normalized.txt');normal.write_text(normalized[variant])
                resources[variant]=json.loads(reflection.read_text())['parameters']
                artifacts[variant]=dict(source_sha256=digest(path),spirv_sha256=digest(binary),
                    reflection_sha256=digest(reflection),normalized_sha256=digest(normal),command=common)
            equal=normalized['baseline']==normalized['candidate']
            same_layout=resources['baseline']==resources['candidate']
            record['comparisons'].append(dict(compiler=version,compiler_sha256=digest(compiler),profile=profile,
                instruction_stream_equal=equal,resource_layout_equal=same_layout,artifacts=artifacts))
            print(version,profile,'instructions',equal,'layout',same_layout,flush=True)
            if not equal or not same_layout:record['status']='failed'
    (args.output/'instruction-equivalence.json').write_text(json.dumps(record,indent=2)+'\n')
    if record['status']!='passed':raise SystemExit(1)


if __name__=='__main__':main()
