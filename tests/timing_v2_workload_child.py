# SPDX-License-Identifier: Apache-2.0
"""Fresh NumPy-only source snapshot smoke. Native runtime remains unverified."""
import hashlib
import importlib.abc
import importlib.machinery
import json
import math
import struct
from pathlib import Path
import subprocess
import sys

class Rejected(Exception):
    def __init__(self,reason): self.reason=reason



def _f32(value):
    """Round once to IEEE binary32 using Python struct, without NumPy."""
    try: return struct.unpack('<f',struct.pack('<f',value))[0]
    except OverflowError: return math.copysign(math.inf,value)


def _same_float_bits(a,b):
    return struct.pack('<f',float(a))==struct.pack('<f',float(b))


def check_depth_convert_scalar(case,actual):
    """Independent scalar oracle for all outputs and their untouched tails."""
    params=case['params']; count=params['count']; rep=params['representation']
    if rep not in (0,1,2): raise Rejected('oracle_representation')
    arrays=case['arrays']; outputs=('depthOut','inverseOut','validOut','confidenceOut')
    if set(actual)!=set(outputs) or set(case['outputs'])!=set(outputs): raise Rejected('oracle_output_set')
    scale=_f32(params['scale']); focal=_f32(params['focalBaseline'])
    for name in outputs:
        got=actual[name]; seed=arrays[name]
        if got.dtype!=seed.dtype or got.shape!=seed.shape or case['live_counts'][name]!=count: raise Rejected('oracle_output_layout')
        if got[count:].tobytes()!=seed[count:].tobytes(): raise Rejected('workload_guard')
    for i in range(count):
        value=float(arrays['values'][i]); valid=bool(arrays['valid'][i]); confidence=float(arrays['confidence'][i])
        usable=valid and math.isfinite(value) and value>0
        depth=0.0
        if usable:
            if rep==0: depth=_f32(value*scale)
            elif rep==1:
                product=_f32(value*scale)
                depth=_f32(1.0/product) if product!=0 else math.inf
            else: depth=_f32(_f32(1.0/value)*focal)
            usable=math.isfinite(depth) and depth>0
        expected_depth=depth if usable else 0.0
        expected_inverse=_f32(1.0/depth) if usable else 0.0
        expected_confidence=confidence if usable else 0.0
        if (int(actual['validOut'][i])!=int(usable) or
            not _same_float_bits(actual['depthOut'][i],expected_depth) or
            not _same_float_bits(actual['inverseOut'][i],expected_inverse) or
            not _same_float_bits(actual['confidenceOut'][i],expected_confidence)):
            raise Rejected('oracle_mismatch')

def main(root,scenario):
    if not sys.flags.isolated or not sys.flags.no_site or not sys.dont_write_bytecode or 'site' in sys.modules: raise Rejected('python_isolation')
    fields={line.partition(':')[0]:line.partition(':')[2].strip() for line in Path('/proc/self/status').read_text().splitlines() if ':' in line}
    if any(fields.get(k)!='0000000000000000' for k in ('CapEff','CapPrm','CapInh','CapAmb','CapBnd')) or fields.get('NoNewPrivs')!='1': raise Rejected('privilege_drop')
    if scenario=='preimport': sys.modules['benchmarks.vkl_timings']=object()
    if scenario=='shadow': sys.modules['sintergpu']=object()
    if any(name.startswith(('sintergpu','benchmarks','numpy')) for name in sys.modules): raise Rejected('preimport_or_shadow')
    if list(root.rglob('*.pyc')) or list(root.rglob('__pycache__')): raise Rejected('bytecode')
    policy=json.loads((root/'policy.json').read_text()); allowed=policy['allowed']; sources=policy['sources']; external=policy['external']
    sys.path[:0]=[str(root/'src'),str(root),str(root/'thirdparty')]
    namespaces={'benchmarks','sintergpu.references','sintergpu.adapters'}
    class Guard(importlib.abc.MetaPathFinder):
        frozen=False
        def find_spec(self,fullname,path=None,target=None):
            if fullname=='torch' or fullname.startswith('torch.'): raise ModuleNotFoundError("Torch intentionally unavailable",name='torch')
            if fullname.startswith(('sintergpu','benchmarks')):
                if fullname not in allowed and fullname not in namespaces: raise Rejected('unexpected_import')
                if self.frozen: raise Rejected('late_import')
                spec=importlib.machinery.PathFinder.find_spec(fullname,path)
                if spec is None: return None
                if fullname in namespaces:
                    if spec.origin is not None or any(not Path(x).resolve().is_relative_to(root.resolve()) for x in spec.submodule_search_locations): raise Rejected('module_shadow')
                else:
                    want=(root/allowed[fullname]).resolve()
                    if not spec.origin or Path(spec.origin).resolve()!=want or want.suffix!='.py': raise Rejected('module_shadow')
                    expected=sources.get(allowed[fullname])
                    if expected is None or hashlib.sha256(want.read_bytes()).hexdigest()!=expected: raise Rejected('module_source_preload')
                return spec
            if fullname=='numpy' or fullname.startswith('numpy.'):
                spec=importlib.machinery.PathFinder.find_spec(fullname,path)
                if spec is None: return None
                if not spec.origin: raise Rejected('numpy_sourceless')
                origin=Path(spec.origin).resolve();base=(root/'thirdparty').resolve()
                if not origin.is_relative_to(base) or origin.suffix not in ('.py','.so'): raise Rejected('numpy_shadow')
                rel=str(origin.relative_to(base))
                if rel not in external or hashlib.sha256(origin.read_bytes()).hexdigest()!=external[rel]: raise Rejected('numpy_identity')
                return spec
            return None
    guard=Guard();sys.meta_path.insert(0,guard)
    import benchmarks.vkl_timings as recipe
    cases=[case for case in recipe.cases() if case['operation']=='vkl.depth_convert.depthConvert']
    if not cases: raise Rejected('workload_missing')
    expected_representations={0,1,2}
    if len(cases)!=6 or {case['params']['representation'] for case in cases}!=expected_representations: raise Rejected('workload_case_set')
    actual=None
    for case_index,case in enumerate(cases):
        actual=recipe.reference_call(case)
        if scenario=='mutate_nonfirst' and case_index==1:
            actual['confidenceOut'][min(5,case['params']['count']-1)]=123456.0
        check_depth_convert_scalar(case,actual)
    loaded={};native=[]
    for name,module in tuple(sys.modules.items()):
        filename=getattr(module,'__file__',None)
        if name=='__main__':
            path=Path(filename).resolve() if filename else None
            if path!=(root/'tests/timing_v2_workload_child.py').resolve() or hashlib.sha256(path.read_bytes()).hexdigest()!=sources['tests/timing_v2_workload_child.py']: raise Rejected('bootstrap_identity')
            continue
        if not name.startswith(('sintergpu','benchmarks','numpy')):
            if filename and Path(filename).resolve().is_relative_to(root.resolve()): raise Rejected('unknown_local_alias:'+name)
            continue
        if name in namespaces: continue
        if not filename: raise Rejected('sourceless_module')
        path=Path(filename).resolve()
        if name.startswith('numpy'):
            if not path.is_relative_to((root/'thirdparty').resolve()): raise Rejected('numpy_shadow')
            rel=str(path.relative_to(root/'thirdparty')); expected=external.get(rel)
            if path.suffix=='.so': native.append(rel)
        else:
            if name not in allowed or not path.is_relative_to(root.resolve()): raise Rejected('module_shadow')
            rel=str(path.relative_to(root));expected=sources.get(rel)
            if rel!=allowed[name] or path.suffix!='.py': raise Rejected('module_shadow')
        digest=hashlib.sha256(path.read_bytes()).hexdigest()
        if digest!=expected: raise Rejected('module_identity')
        loaded[name]={'path':rel,'sha256':digest}
    guard.frozen=True
    if scenario=='late': __import__('sintergpu.timing_closure_v2')
    if scenario=='unexpected': __import__('sintergpu.unexpected')
    try:(root/'unauthorized').write_text('bad')
    except OSError:pass
    else:raise Rejected('snapshot_writable')
    if subprocess.run(['/usr/bin/mount','-o','remount,rw',str(root)],capture_output=True).returncode==0: raise Rejected('remount_succeeded')
    native_maps=[]
    for line in Path('/proc/self/maps').read_text().splitlines():
        pieces=line.split(None,5)
        if len(pieces)<6 or not pieces[5].startswith('/') or '.so' not in pieces[5]: continue
        name=pieces[5]
        if name not in native_maps: native_maps.append(name)
    native_receipts=[]
    for name in sorted(native_maps):
        path=Path(name); digest=None
        if path.is_file():
            hasher=hashlib.sha256()
            with path.open('rb') as f:
                for block in iter(lambda:f.read(1<<20),b''): hasher.update(block)
            digest=hasher.hexdigest()
        native_receipts.append({'path':name,'sha256_observed_unverified':digest})
    return {'scope':'Independent scalar float32 oracle checks every live output and tail in six CPU cases; no GPU or Torch donor parity.',
            'native_mappings_unverified':native_receipts,'python_version_unverified':sys.version,'recipe':'vkl.depth_convert.depthConvert','case_count':len(cases),'outputs':sorted(actual),'loaded':loaded,
            'numpy_native_modules_unverified':sorted(native),'torch_loaded':'torch' in sys.modules,'write_blocked':True,'remount_blocked':True}

try:print(json.dumps(main(Path(sys.argv[1]),sys.argv[2])))
except Rejected as exc:print(json.dumps({'phase':'child','reason':exc.reason}),file=sys.stderr);sys.exit(1)
except Exception:print(json.dumps({'phase':'child','reason':'unexpected_exception'}),file=sys.stderr);sys.exit(1)
