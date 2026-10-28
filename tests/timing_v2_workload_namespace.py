# SPDX-License-Identifier: Apache-2.0
"""Private namespace setup for provisional NumPy-only CPU smoke."""
import atexit
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile


def copy(root,destination,files):
    for name,digest in files.items():
        rel=Path(name)
        if rel.is_absolute() or '..' in rel.parts or rel.suffix=='.pyc' or '__pycache__' in rel.parts: raise ValueError('bad path')
        src=root
        for part in rel.parts:
            src=src/part
            if src.is_symlink(): raise ValueError('symlink')
        if not src.is_file() or not src.resolve().is_relative_to(root.resolve()): raise ValueError('source')
        raw=src.read_bytes()
        if hashlib.sha256(raw).hexdigest()!=digest: raise ValueError('source drift')
        target=destination/rel;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(raw)


def main():
    source=Path(sys.argv[1]);spec=json.loads(sys.stdin.read());site=Path(spec['numpy_site'])
    mountpoint=Path(tempfile.mkdtemp(prefix='sintergpu-v2-workload-'))
    def cleanup():
        for _ in range(2): subprocess.run(['/usr/bin/umount',str(mountpoint)],capture_output=True)
        try: mountpoint.rmdir()
        except OSError: pass
    atexit.register(cleanup)
    subprocess.run(['/usr/bin/mount','-t','tmpfs','-o','size=192m,mode=0700','tmpfs',str(mountpoint)],check=True)
    copy(source,mountpoint,spec['sources']);copy(site,mountpoint/'thirdparty',spec['external'])
    if spec['scenario']=='pyc': (mountpoint/'tests/injected.pyc').write_bytes(b'untrusted')
    if spec['scenario']=='numpy_source_tamper':
        target=mountpoint/'thirdparty/numpy/__init__.py'; target.write_bytes(target.read_bytes()+b'\n# altered\n')
    if spec['scenario']=='numpy_native_tamper':
        candidates=list((mountpoint/'thirdparty/numpy/_core').glob('_multiarray_umath*.so'))
        if len(candidates)!=1: raise ValueError('missing NumPy native module')
        target=candidates[0]; target.write_bytes(target.read_bytes()+b'altered')
    if spec['scenario']=='source_drift':
        target=mountpoint/'src/sintergpu/vkl.py'; target.write_bytes(target.read_bytes()+b'\n# altered\n')
    marker=Path('/tmp')/(mountpoint.name+'-preload-marker')
    if spec['scenario']=='source_side_effect':
        target=mountpoint/'src/sintergpu/vkl.py'
        code="\nfrom pathlib import Path as _MarkerPath\n_MarkerPath("+repr(str(marker))+").write_text('executed')\n"
        target.write_bytes(target.read_bytes()+code.encode())
    (mountpoint/'policy.json').write_text(json.dumps({'allowed':spec['allowed'],'sources':spec['sources'],'external':spec['external']}))
    subprocess.run(['/usr/bin/mount','--bind',str(mountpoint),str(mountpoint)],check=True)
    subprocess.run(['/usr/bin/mount','-o','remount,bind,ro',str(mountpoint)],check=True)
    child=mountpoint/'tests/timing_v2_workload_child.py'
    command=['/usr/bin/setpriv','--bounding-set=-all','--inh-caps=-all','--ambient-caps=-all','--no-new-privs',sys.executable,'-I','-S','-B',str(child),str(mountpoint),spec['scenario']]
    result=subprocess.run(command,capture_output=True,text=True)
    if result.returncode:
        if marker.exists():
            marker.unlink()
            print(json.dumps({'phase':'child','reason':'side_effect_observed'}),file=sys.stderr);return 1
        try: error=json.loads(result.stderr.strip())
        except (ValueError,TypeError): error={'phase':'child','reason':'unstructured_failure'}
        print(json.dumps(error),file=sys.stderr);return 1
    if marker.exists():
        marker.unlink()
        raise ValueError('unexpected side effect')
    source_after={p:hashlib.sha256((mountpoint/p).read_bytes()).hexdigest() for p in spec['sources']}
    external_after={p:hashlib.sha256((mountpoint/'thirdparty'/p).read_bytes()).hexdigest() for p in spec['external']}
    if source_after!=spec['sources'] or external_after!=spec['external']: raise ValueError('snapshot drift')
    print(json.dumps({'child':json.loads(result.stdout),'source_before':spec['sources'],'source_after':source_after,
                      'numpy_before':spec['external'],'numpy_after':external_after}));return 0

try:sys.exit(main())
except Exception:
    print(json.dumps({'phase':'setup','reason':'setup_failure'}),file=sys.stderr);sys.exit(1)
