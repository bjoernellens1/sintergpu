# SPDX-License-Identifier: Apache-2.0
"""Provisional control child. Import hooks do not sandbox hostile code."""
import hashlib
import importlib.abc
import json
from pathlib import Path
import subprocess
import sys

class Rejected(Exception):
    def __init__(self,reason): self.reason=reason

class Guard(importlib.abc.MetaPathFinder):
    frozen=False
    def find_spec(self,fullname,path=None,target=None):
        if fullname.startswith(('timing_v2_snapshot_','sintergpu','benchmarks')):
            if fullname not in {'timing_v2_snapshot_allow','timing_v2_snapshot_late'}: raise Rejected('unexpected_import')
            if self.frozen: raise Rejected('late_import')
        return None


def main(root,scenario):
    if not sys.flags.isolated or not sys.flags.no_site or not sys.dont_write_bytecode: raise Rejected('isolation_flags')
    if 'site' in sys.modules or any('site-packages' in p or 'dist-packages' in p for p in sys.path): raise Rejected('site_enabled')
    status=Path('/proc/self/status').read_text()
    # Negative scenarios modify only the diagnostic text, never real privileges.
    for scenario_name,field in (('cap_prm','CapPrm'),('cap_inh','CapInh'),('cap_amb','CapAmb')):
        if scenario==scenario_name: status=status.replace(field+':\t0000000000000000',field+':\t0000000000000001')
    if scenario=='cap_missing': status='\n'.join(line for line in status.splitlines() if not line.startswith('CapAmb:'))
    required=('CapEff','CapPrm','CapInh','CapAmb','CapBnd')
    fields={}
    for line in status.splitlines():
        key,separator,value=line.partition(':')
        if separator and key in (*required,'NoNewPrivs'):
            if key in fields: raise Rejected('privilege_drop')
            fields[key]=value.strip()
    if any(fields.get(key)!='0000000000000000' for key in required) or fields.get('NoNewPrivs')!='1': raise Rejected('privilege_drop')
    if scenario=='preimport': sys.modules['timing_v2_snapshot_late']=object()
    if scenario=='shadow': sys.modules['sintergpu']=object()
    if any(name.startswith(('timing_v2_snapshot_','sintergpu','benchmarks')) for name in sys.modules): raise Rejected('preimport_or_shadow')
    if list(root.rglob('*.pyc')) or list(root.rglob('__pycache__')): raise Rejected('bytecode')
    guard=Guard(); sys.meta_path.insert(0,guard); sys.path.insert(0,str(root/'tests'))
    import timing_v2_snapshot_allow as allowed
    path=Path(allowed.__file__).resolve()
    if path!=root/'tests/timing_v2_snapshot_allow.py' or allowed.execute()!='allowlisted-module-executed': raise Rejected('allowed_module_identity')
    module_hash=hashlib.sha256(path.read_bytes()).hexdigest()
    guard.frozen=True
    if scenario=='late': __import__('timing_v2_snapshot_late')
    if scenario=='unexpected': __import__('sintergpu.unexpected')
    try: (root/'unauthorized').write_text('must fail')
    except OSError: pass
    else: raise Rejected('snapshot_writable')
    remount=subprocess.run(['/usr/bin/mount','-o','remount,rw',str(root)],capture_output=True)
    if remount.returncode==0: raise Rejected('remount_succeeded')
    nested=subprocess.run(['/usr/bin/unshare','-Urnm','/usr/bin/mount','-t','tmpfs','tmpfs',str(root)],capture_output=True)
    if nested.returncode==0: raise Rejected('nested_overmount_succeeded')
    return {'isolated':True,'write_blocked':True,'remount_blocked':True,'nested_overmount_blocked':True,
            'capabilities_dropped':True,'allowed_module_executed':True,'allowed_module_sha256':module_hash}

try:
    print(json.dumps(main(Path(sys.argv[1]),sys.argv[2])))
except Rejected as exc:
    print(json.dumps({'phase':'child','reason':exc.reason}),file=sys.stderr); sys.exit(1)
except Exception:
    print(json.dumps({'phase':'child','reason':'unexpected_exception'}),file=sys.stderr); sys.exit(1)
