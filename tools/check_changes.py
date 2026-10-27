#!/usr/bin/env python3
"""Require classified changelog fragments and version changes on pull requests."""
import argparse
from pathlib import Path
import re
import subprocess


def validate_transition(old, new, compatibility, initial=False):
    def parse(value):
        if not re.fullmatch(r'(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)',value):
            raise ValueError('strict SemVer required')
        return tuple(map(int,value.split('.')))
    before,after=parse(old),parse(new)
    if initial:
        if compatibility != 'initial' or after != (0,1,0):
            raise ValueError('initial implementation must begin at 0.1.0')
        return
    if compatibility not in ('compatible','breaking'):
        raise ValueError('Compatibility must be compatible or breaking')
    if after <= before:
        raise ValueError('VERSION must increase for each change')
    if compatibility == 'breaking':
        if before[0] == 0 and after[:2] <= before[:2]:
            raise ValueError('breaking pre-1.0 changes require minor increment')
        if before[0] > 0 and after[0] <= before[0]:
            raise ValueError('breaking changes require major increment')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--base',required=True); args=p.parse_args()
    root=Path(__file__).resolve().parents[1]
    def git(*argv):
        return subprocess.check_output(['git',*argv],cwd=root,text=True)
    changed=git('diff','--name-only',args.base,'HEAD').splitlines()
    fragments=[x for x in changed if x.startswith('changes/') and x.endswith('.md') and (root/x).is_file()]
    if not fragments:
        raise SystemExit('a new or updated changelog fragment is mandatory')
    classifications=[]
    for name in fragments:
        match=re.search(r'^Compatibility: (initial|compatible|breaking)$',(root/name).read_text(),re.M)
        if not match:
            raise SystemExit(name+': missing Compatibility classification')
        classifications.append(match[1])
    initial=not git('ls-tree','--name-only',args.base,'pyproject.toml').strip()
    compatibility='breaking' if 'breaking' in classifications else classifications[0]
    validate_transition(git('show',args.base+':VERSION').strip(),(root/'VERSION').read_text().strip(),compatibility,initial)
    print('changelog classification and version transition passed')


if __name__=='__main__':
    main()
