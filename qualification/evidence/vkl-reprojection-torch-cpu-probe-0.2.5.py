import importlib.util,json,types,sys,platform
from pathlib import Path
import numpy as np
import torch
from benchmarks import reprojection_support
from sintergpu.references.temporal_reprojection_numeric import temporal_reproject_rgb
spec=importlib.util.spec_from_file_location('reprojection_gpu_test',Path('tests/test_vkl_reprojection_gpu.py'));mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
class Captured(Exception):pass
captured={}
def capture(device,graph,inputs):
 captured.update(inputs=inputs);raise Captured
reprojection_support.PreparedGraph=capture
fake=types.SimpleNamespace(info=types.SimpleNamespace(limits=types.SimpleNamespace(max_compute_dispatch_thread_groups=[65535])))
try:mod.test_t0_full_outputs_and_source_major_packing(fake,8,3,17,19,False,False,False)
except Captured:pass
inputs=captured['inputs'];target_rgb=np.zeros((8,3,17,19),np.float32)
expected=temporal_reproject_rgb(target_rgb=target_rgb,**inputs)
donor=mod.frozen_module('temporal_reproject').temporal_reproject_rgb(target_rgb=torch.from_numpy(target_rgb),**{k:torch.from_numpy(v) for k,v in inputs.items()})
results={}
for key,a in expected.items():
 b=donor[key].numpy();bad=~np.isclose(a,b,rtol=8e-6,atol=8e-7,equal_nan=True);idx=np.argwhere(bad)
 results[key]={'mismatches':int(bad.sum()),'max_abs':float(np.nanmax(np.abs(a.astype(float)-b.astype(float)))),'first_indices':idx[:12].tolist(),'values':[[float(a[tuple(i)]),float(b[tuple(i)])] for i in idx[:12]]}
print(json.dumps(dict(python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,cpu=platform.processor(),results=results),indent=2))
from sintergpu.references.reprojection_numeric import reproject_rgbd,sample_pixels
b,n,h,w=8,3,17,19
def repeat(a):return np.repeat(a[:,None],n,axis=1).reshape((b*n,)+a.shape[1:])
args=(repeat(inputs['target_depth']),repeat(inputs['target_intrinsics']),repeat(inputs['target_pose_c2w']),inputs['source_intrinsics'].reshape(b*n,3,3),inputs['source_pose_c2w'].reshape(b*n,4,4))
np_d4=reproject_rgbd(*args,target_hw=(h,w));torch_d4=mod.frozen_module('reproject').reproject_rgbd(*[torch.from_numpy(a) for a in args],target_hw=(h,w))
def summary(a,b):
 bad=~np.isclose(a,b,rtol=8e-6,atol=8e-7,equal_nan=True)
 return dict(mismatches=int(bad.sum()),max_abs=float(np.nanmax(np.abs(a.astype(float)-b.astype(float)))))
layer={k:summary(np_d4[k],v.numpy()) for k,v in torch_d4.items()}
from_donor_xy=sample_pixels(inputs['source_rgb'].reshape(b*n,3,h,w),torch_d4['projected_xy'].numpy(),mode='bilinear').reshape(b,n,3,h,w)
layer['numpy_sampler_on_frozen_xy_vs_frozen_rgb']=summary(from_donor_xy,donor['aligned_rgb'].numpy())
layer['numpy_sampler_on_frozen_xy_vs_original_numpy_rgb']=summary(from_donor_xy,expected['aligned_rgb'])
print(json.dumps(dict(layer_boundaries=layer,torch_config=torch.__config__.show()),indent=2))
