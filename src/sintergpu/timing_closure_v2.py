# SPDX-License-Identifier: Apache-2.0
"""Provisional per-workload source identities; not a timing qualification.

The registered family and runtime/module equivalence need independent review.
Manifests are provisional file inventories only. They MUST NOT be used to
assert current measurement status or to authorize collection or promotion.
Loaded-code identity, immutable process snapshots, compiler/runtime mapping
and full dependency completeness have not been established. No historical v1
measurements are promoted.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import sys

# Family policy is library-owned, not supplied by a measurement caller. This
# provisional policy is for the existing vkl_timings depthConvert path only.
# VKL reference __init__ conditionally imports Torch modules. This provisional
# profile requires the Torch-enabled imported module set; unsupported imports
# and unobserved runtime configurations fail closed pending policy review.
_VKL_NAMES='''__init__ camera_registration compact contract dense_pose_depth dense_track_reduce depth_align depth_convert depth_cost_volume depth_densify depth_evidence depth_fuse depth_pyramid direct_pose_depth evidence geometry keyframe_filter keypoint_evidence keypoint_selectors landmark_update lk numpy photometric_isp project_depth_evidence pyramid reproject rescale residual rigid_alignment sparse_geometry sparse_pose_reduce spatial temporal temporal_best temporal_compare temporal_depth_fuse temporal_reproject tracking_prepare tracking_solve triangulate'''.split()
_BASE={
 'shader':'src/sintergpu/kernels/vkl/depth_convert.slang',
 'runner':'benchmarks/vkl_timings.py',
 'fixture':'benchmarks/locus_timings.py',
 'reference.geometry':'src/sintergpu/references/geometry_depth_numeric.py',
 'reference.resampling':'src/sintergpu/references/depth_resampling_numeric.py',
 'reference.temporal':'src/sintergpu/references/temporal_best_numeric.py',
 'adapter':'src/sintergpu/adapters/slangpy.py',
 'setup':'src/sintergpu/vkl.py',
 'catalog':'src/sintergpu/catalog.py',
 'timing':'src/sintergpu/timings.py',
 'closure_validator':'src/sintergpu/timing_closure_v2.py',
 'package':'src/sintergpu/__init__.py',
 'version':'src/sintergpu/data/VERSION',
 'catalog_data':'src/sintergpu/data/catalog.json',
}
for _name in _VKL_NAMES:
    _BASE['reference.vkl.'+_name]='src/sintergpu/references/vkl/'+_name+'.py'
_MODULES={
 'benchmarks.vkl_timings':_BASE['runner'], 'benchmarks.locus_timings':_BASE['fixture'],
 'sintergpu':_BASE['package'], 'sintergpu.vkl':_BASE['setup'],
 'sintergpu.catalog':_BASE['catalog'], 'sintergpu.timings':_BASE['timing'],
 'sintergpu.timing_closure_v2':_BASE['closure_validator'],
 'sintergpu.adapters.slangpy':_BASE['adapter'],
 'sintergpu.references.geometry_depth_numeric':_BASE['reference.geometry'],
 'sintergpu.references.depth_resampling_numeric':_BASE['reference.resampling'],
 'sintergpu.references.temporal_best_numeric':_BASE['reference.temporal'],
 'sintergpu.references.vkl':_BASE['reference.vkl.__init__'],
 'sintergpu.references.vkl.numpy':_BASE['reference.vkl.numpy'],
}
_MODULES.update({('sintergpu.references.vkl.'+name):_BASE['reference.vkl.'+name] for name in _VKL_NAMES if name not in ('__init__','numpy')})
POLICIES={'vkl.depth_convert.depthConvert':{'files':_BASE,'required_modules':_MODULES,'optional_modules':{},
 'compiler':{'backend':'vulkan','floating_point_mode':'precise','entrypoint':'depthConvert'},
 'shader_directives':[], 'shader_sha256':'2ced5aa077ac8484e0bbd2426f468caf69ce707b25f740b206541303fb2a18f5', 'policy_version':1}}


def _canonical(obj):
    return json.dumps(obj,sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False).encode('utf8')


def _policy(family):
    if not isinstance(family,str) or family not in POLICIES: raise ValueError('unregistered family')
    return POLICIES[family]


def _path(root,name):
    if not isinstance(name,str) or not name or '\\' in name: raise ValueError('invalid source path')
    relative=PurePosixPath(name)
    if relative.is_absolute() or any(p in ('.','..') for p in relative.parts) or str(relative)!=name: raise ValueError('path must be canonical and relative')
    root=Path(root).resolve(strict=True); current=root
    for part in relative.parts:
        current=current/part
        if current.is_symlink(): raise ValueError('symlink in source path: '+name)
    if not current.is_file() or not current.resolve().is_relative_to(root): raise ValueError('missing or outside root: '+name)
    return current


def _shader_directives(data):
    # No preprocessor include or Slang import is presently declared by this
    # family. Unknown/new transitive imports fail closed pending policy review.
    text=data.decode('utf8')
    return re.findall(r'^\s*(?:#\s*include|import)\s+[^\n;]+',text,re.MULTILINE)


def _check_modules(root,policy):
    modules=sys.modules
    for name,path in policy['required_modules'].items():
        module=modules.get(name)
        if module is None: raise ValueError('required loaded module missing: '+name)
        actual=getattr(module,'__file__',None)
        if not actual or Path(actual).resolve()!=_path(root,path).resolve(): raise ValueError('loaded module path mismatch: '+name)
    for name,path in policy['optional_modules'].items():
        if name in modules:
            actual=getattr(modules[name],'__file__',None)
            if not actual or Path(actual).resolve()!=_path(root,path).resolve(): raise ValueError('loaded module path mismatch: '+name)
    # Detect unknown loaded local submodules rather than silently assuming their
    # source is included. Outside-root shadow modules with this namespace fail.
    known=set(policy['required_modules'])|set(policy['optional_modules'])
    for name in modules:
        if name in known: continue
        if name in ('benchmarks','sintergpu.adapters','sintergpu.references'): continue
        if name.startswith(('sintergpu.', 'benchmarks.')): raise ValueError('unknown loaded local module: '+name)
        filename=getattr(modules[name],'__file__',None)
        if filename and Path(filename).resolve().is_relative_to(Path(root).resolve()):
            raise ValueError('unknown loaded local module: '+name)


def build_manifest(root,family,*,compiler):
    policy=_policy(family)
    if compiler!=policy['compiler']: raise ValueError('compiler configuration differs from registered policy')
    _check_modules(root,policy)
    records=[]
    for role,path in sorted(policy['files'].items()):
        data=_path(root,path).read_bytes()
        if role=='shader':
            if _shader_directives(data)!=policy['shader_directives']: raise ValueError('unregistered transitive shader dependency')
            if hashlib.sha256(data).hexdigest()!=policy['shader_sha256']: raise ValueError('shader differs from pinned policy bytes')
        records.append({'role':role,'path':path,'sha256':hashlib.sha256(data).hexdigest()})
    payload={'schema_version':2,'status':'provisional_not_for_collection','family':family,'policy_version':policy['policy_version'],'compiler_claim':dict(compiler),'files':records}
    return {**payload,'sha256':hashlib.sha256(_canonical(payload)).hexdigest()}


def validate_manifest(root,manifest):
    """Check provisional file inventory consistency, not executing-code identity."""
    if not isinstance(manifest,dict) or set(manifest)!={'schema_version','status','family','policy_version','compiler_claim','files','sha256'}: raise ValueError('invalid manifest shape')
    if manifest['status']!='provisional_not_for_collection': raise ValueError('provisional status required')
    if type(manifest['schema_version']) is not int or manifest['schema_version']!=2: raise ValueError('unknown schema')
    policy=_policy(manifest['family']); records=manifest['files']
    if type(manifest['policy_version']) is not int or manifest['policy_version']!=policy['policy_version']: raise ValueError('unknown policy version')
    if not isinstance(records,list): raise ValueError('invalid records')
    roles=[];paths=[]
    for r in records:
        if not isinstance(r,dict) or set(r)!={'role','path','sha256'}: raise ValueError('invalid record')
        if not isinstance(r['role'],str) or not isinstance(r['path'],str): raise ValueError('invalid role/path type')
        roles.append(r['role']); paths.append(r['path'])
        if r['role'] not in policy['files'] or policy['files'][r['role']]!=r['path']: raise ValueError('unknown or wrong role/path')
        if not isinstance(r['sha256'],str) or not re.fullmatch('[0-9a-f]{64}',r['sha256']): raise ValueError('invalid hash')
    if len(roles)!=len(set(roles)) or len(paths)!=len(set(paths)): raise ValueError('duplicate role/path')
    if set(roles)!=set(policy['files']): raise ValueError('missing dependency roles')
    expected=build_manifest(root,manifest['family'],compiler=manifest['compiler_claim'])
    if manifest!=expected: raise ValueError('manifest digest, ordering or current bytes differ')
    return manifest['sha256']


def require_collection_ready(manifest):
    """No v2 manifest currently proves loaded-code/compiler identity.

    A future reviewed implementation must launch in a freshly created process
    against a verified immutable snapshot, track local imports before/after,
    identify actual compiler/runtime mappings and establish source/bytecode
    equivalence before enabling collection or current-status promotion.
    """
    raise ValueError('provisional manifest cannot authorize collection or promotion')
