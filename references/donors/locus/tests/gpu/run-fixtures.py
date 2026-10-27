#!/usr/bin/env python3
"""Execute identical scalar vectors on an explicitly chosen GPU development runner."""
import argparse,json,math,struct,subprocess,sys
from pathlib import Path
from cases import cases
from reference import execute
root=Path(__file__).resolve().parents[2]
def packed(rows):return struct.pack('<'+'f'*len(rows)*4,*[v for row in rows for v in row])
def rounded(rows):return [list(struct.unpack('<4f',struct.pack('<4f',*row)))for row in rows]
def main():
 p=argparse.ArgumentParser();p.add_argument('--backend',choices=['vulkan','metal','reference'],default='reference');p.add_argument('--runner');p.add_argument('--shaders',type=Path,default=root/'build/shaders');p.add_argument('--out',type=Path,default=root/'build/gpu-fixtures');p.add_argument('--allow-unavailable',action='store_true');a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True);records=[]
 if a.backend!='reference'and not a.runner:p.error('--runner required for execution')
 for case in cases():
  params=list(struct.unpack('<32f',struct.pack('<32f',*case['params'])));inputs=[rounded(b)for b in case['inputs']];initial=rounded(case['output']);expected,state=execute(case['name'],params,inputs,initial,case['state'])
  buffers=[packed(b)for b in inputs]+[packed(initial),struct.pack('<'+'I'*len(case['state']),*case['state']),struct.pack('<32f',*params)]
  fixture=a.out/(case['label']+'.bin');output=a.out/(case['label']+'.out');fixture.write_bytes(struct.pack('<9I',0x4c475031,case['width'],case['height'],*[len(b)for b in buffers])+b''.join(buffers))
  if a.backend=='reference':records.append({'case':case['label'],'status':'reference-passed'});continue
  artifact=a.shaders/(case['name']+('.spv'if a.backend=='vulkan'else'.metallib'))
  result=subprocess.run([a.runner,str(artifact),str(fixture),str(output),case['name']],capture_output=True,text=True)
  if result.returncode==77 and a.allow_unavailable:
   evidence={'backend':a.backend,'status':'pending','reason':result.stderr.strip(),'cases':records,'physical_capture_interop':'pending: texture/buffer bridge and camera consumer orchestration not integrated'};(a.out/'results.json').write_text(json.dumps(evidence,indent=2)+'\n');print(json.dumps(evidence));return 0
  if result.returncode:raise RuntimeError(case['label']+': '+result.stderr)
  payload=output.read_bytes();expected_length=len(buffers[3])+len(buffers[4])
  if len(payload)!=expected_length:raise AssertionError('output length mismatch')
  actual=struct.unpack('<'+'f'*(len(buffers[3])//4),payload[:len(buffers[3])]);golden=[v for row in expected for v in row]
  for i,(v,e)in enumerate(zip(actual,golden)):
   if not(math.isnan(v)and math.isnan(e))and not math.isclose(v,e,abs_tol=2e-4,rel_tol=2e-4):raise AssertionError(f'{case["label"]}: float[{i}] actual={v} expected={e}')
  actual_state=list(struct.unpack('<'+'I'*(len(buffers[4])//4),payload[len(buffers[3]):]))
  if actual_state!=state:raise AssertionError(f'{case["label"]}: state actual={actual_state} expected={state}')
  records.append({'case':case['label'],'status':'passed','metrics':json.loads(result.stdout)})
 evidence={'backend':a.backend,'status':'passed','cases':records,'tolerance':{'absolute':2e-4,'relative':2e-4},'physical_capture_interop':'pending: texture/buffer bridge and camera consumer orchestration not integrated'};(a.out/'results.json').write_text(json.dumps(evidence,indent=2)+'\n');print(json.dumps(evidence));return 0
if __name__=='__main__':
 try:sys.exit(main())
 except(Exception)as e:print(str(e),file=sys.stderr);sys.exit(1)
