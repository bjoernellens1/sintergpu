#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Verify actual installed-module imports against unchanged Caspar exports.

Only debug names/source and SSA identifier spelling are ignored. Executable
order, types, constants, decorations, resource layouts and actual LocalSize stay
checked. Comparator derives from candidate2def63b compile_inlined_point.py.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'src'))


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



def verify_compiler(compiler,output):
    from benchmarks.point_plane_support import point_case
    from sintergpu import module_path
    compiler=Path(compiler).resolve();output=Path(output);output.mkdir(parents=True,exist_ok=True)
    core=module_path('optimization.point_plane');include=core.parent.parent
    version=subprocess.run([str(compiler),'-version'],capture_output=True,text=True,check=True)
    record=dict(compiler_version=(version.stdout+version.stderr).strip(),compiler_sha256=hashlib.sha256(compiler.read_bytes()).hexdigest(),
        module_sha256=hashlib.sha256(core.read_bytes()).hexdigest(),comparisons=[],gpu_execution='not performed')
    for profile in ('aosoa','scalar'):
        for size in (1024,256):
            case=point_case(None,1025,profile,local_size=size);code={};layouts={};artifacts={}
            for variant,source in case['sources'].items():
                path=output/f'{profile}-{size}-{variant}.slang';path.write_text(source)
                spv=path.with_suffix('.spv');reflection=path.with_suffix('.reflection.json');assembly=path.with_suffix('.spv-asm')
                common=[str(compiler),str(path),'-I',str(include),'-entry','main','-profile','spirv_1_5','-fp-mode','precise','-O3']
                subprocess.run([*common,'-target','spirv','-o',str(spv),'-reflection-json',str(reflection)],check=True,capture_output=True,text=True)
                subprocess.run([*common,'-target','spirv-asm','-o',str(assembly)],check=True,capture_output=True,text=True)
                actual=1024 if variant=='baseline' else size
                assert re.findall(r'OpExecutionMode\s+%\w+\s+LocalSize\s+(\d+)\s+(\d+)\s+(\d+)',assembly.read_text())==[(str(actual),'1','1')]
                code[variant]=normalized_instructions(assembly.read_text());layouts[variant]=json.loads(reflection.read_text())['parameters']
                artifacts[variant]={name:hashlib.sha256(p.read_bytes()).hexdigest() for name,p in [('source',path),('spirv',spv),('assembly',assembly),('reflection',reflection)]}
            same_layout=layouts['baseline']==layouts['candidate']
            equal=code['baseline']==code['candidate']
            launch_equal=code['baseline']==code['candidate'].replace(' LocalSize 256 1 1',' LocalSize 1024 1 1')
            assert same_layout and launch_equal,(profile,size,'import changed executable or resource ABI')
            record['comparisons'].append(dict(profile=profile,baseline_local_size=1024,candidate_local_size=size,
                instruction_stream_equal=equal,equal_except_declared_local_size=launch_equal,resource_layout_equal=same_layout,artifacts=artifacts))
    subprocess.run([str(compiler),str(ROOT/'examples/point_plane_import.slang'),'-I',str(include),'-target','spirv','-profile','spirv_1_5',
        '-entry','main','-fp-mode','precise','-O3','-o',str(output/'example.spv')],check=True,capture_output=True,text=True)
    record['example_compiled']=True
    (output/'results.json').write_text(json.dumps(record,indent=2)+'\n')
    return record


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--slangc',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();r=verify_compiler(args.slangc,args.output)
    print(json.dumps(dict(status='passed',compiler=r['compiler_version'],comparisons=len(r['comparisons']))))


if __name__=='__main__':main()
