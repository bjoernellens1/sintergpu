# SPDX-License-Identifier: Apache-2.0
"""Explicit incumbent-vs-itself diagnostic, never candidate qualification."""
import argparse
from datetime import datetime,timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform

import slangpy as spy

from family_support import point_case
from family_timings import measure,METHOD


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--shared-resources',action='store_true')
    parser.add_argument('--cpu',type=int)
    parser.add_argument('--count',type=int,default=1025)
    parser.add_argument('--blocks',type=int,default=1000)
    args=parser.parse_args()
    if os.getenv('SINTERGPU_GPU_TESTS')!='1':raise RuntimeError('GPU opt-in required')
    if args.cpu is not None:
        if args.cpu not in os.sched_getaffinity(0):raise ValueError('CPU not permitted')
        os.sched_setaffinity(0,{args.cpu})
    lock=open('/tmp/sintergpu-gpu.lock','a+')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    owner=dict(pid=os.getpid(),host=platform.node(),started_utc=datetime.now(timezone.utc).isoformat(),
        experiment='A/A incumbent only; NOT candidate qualification',output=str(args.output.resolve()))
    owner_path=Path('/tmp/sintergpu-gpu-owner.json');owner_path.write_text(json.dumps(owner,indent=2)+'\n')
    device=None
    try:
        adapter=next(a for a in spy.Device.enumerate_adapters(spy.DeviceType.vulkan) if a.vendor_id==0x1002 and a.device_id==0x1586)
        device=spy.Device(type=spy.DeviceType.vulkan,adapter_luid=adapter.luid,
            compiler_options={'floating_point_mode':spy.SlangFloatingPointMode.precise})
        case=point_case(device,args.count,'aosoa')
        # Deliberately run the same compiled incumbent twice. This diagnostic
        # tests the noise/allocation floor and is labelled as such in the record.
        case['kernels']['candidate']=case['kernels']['baseline']
        case['sources']['candidate']=case['sources']['baseline']
        if args.shared_resources:
            case['buffers']['candidate']=case['buffers']['baseline']
            case['bindings']['candidate']=case['bindings']['baseline']
        row=measure(case,args.blocks,10)
        result=dict(experiment=owner['experiment'],method=METHOD,blocks=args.blocks,
            shared_resources=args.shared_resources,cpu_affinity=sorted(os.sched_getaffinity(0)),
            owner=owner,device=device.info.adapter_name,measurement=row,
            interpretation='A/A cannot qualify a candidate; false rejections expose measurement uncertainty')
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
        print(json.dumps({name:dict(decision=value['decision'],median_upper=value['median_ratio_upper95'],p95_upper=value['p95_ratio_upper95']) for name,value in row['acceptance'].items()}))
    finally:
        if device is not None:device.close()
        owner['finished_utc']=datetime.now(timezone.utc).isoformat();owner_path.write_text(json.dumps(owner,indent=2)+'\n');lock.close()


if __name__=='__main__':main()
