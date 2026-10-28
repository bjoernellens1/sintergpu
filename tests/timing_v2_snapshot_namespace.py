# SPDX-License-Identifier: Apache-2.0
"""Internal namespace setup for CPU-only provisional snapshot experiment."""
import atexit
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile


def main():
    source=Path(sys.argv[1]); spec=json.loads(sys.stdin.read()); scenario=spec['scenario']
    mountpoint=Path(tempfile.mkdtemp(prefix='sintergpu-v2-snapshot-'))
    def cleanup():
        for _ in range(2): subprocess.run(['/usr/bin/umount',str(mountpoint)],capture_output=True)
        try: mountpoint.rmdir()
        except OSError: pass
    atexit.register(cleanup)
    subprocess.run(['/usr/bin/mount','-t','tmpfs','-o','size=64m,mode=0700','tmpfs',str(mountpoint)],check=True)
    for name,want in spec['files'].items():
        if name.endswith('.pyc') or '__pycache__' in Path(name).parts: raise ValueError('bytecode forbidden')
        src=source
        for part in Path(name).parts:
            src=src/part
            if src.is_symlink(): raise ValueError('symlinked source ancestor')
        if not src.is_file() or not src.resolve().is_relative_to(source.resolve()): raise ValueError('invalid source')
        data=src.read_bytes()
        if hashlib.sha256(data).hexdigest()!=want: raise ValueError('source changed during copy')
        dest=mountpoint/name; dest.parent.mkdir(parents=True,exist_ok=True); dest.write_bytes(data)
    if scenario=='pyc': (mountpoint/'tests/injected.pyc').write_bytes(b'untrusted')
    subprocess.run(['/usr/bin/mount','--bind',str(mountpoint),str(mountpoint)],check=True)
    subprocess.run(['/usr/bin/mount','-o','remount,bind,ro',str(mountpoint)],check=True)
    child=mountpoint/'tests/timing_v2_snapshot_child.py'
    command=['/usr/bin/setpriv','--bounding-set=-all','--inh-caps=-all','--ambient-caps=-all','--no-new-privs',sys.executable,'-I','-S','-B',str(child),str(mountpoint),scenario]
    result=subprocess.run(command,capture_output=True,text=True)
    if result.returncode:
        try: error=json.loads(result.stderr.strip())
        except (ValueError,TypeError): error={'phase':'child','reason':'unstructured_failure'}
        if set(error)!={'phase','reason'} or error['phase']!='child': error={'phase':'child','reason':'unstructured_failure'}
        print(json.dumps(error),file=sys.stderr); return 1
    after={name:hashlib.sha256((mountpoint/name).read_bytes()).hexdigest() for name in spec['files']}
    if after!=spec['files']: raise RuntimeError('snapshot bytes changed')
    print(json.dumps({'child':json.loads(result.stdout),'before':spec['files'],'after':after})); return 0

try: sys.exit(main())
except Exception:
    print(json.dumps({'phase':'setup','reason':'setup_failure'}),file=sys.stderr); sys.exit(1)
