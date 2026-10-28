# SPDX-License-Identifier: Apache-2.0
"""Validate complete raw timing runs; no performance qualification."""
import hashlib
import math
import re
import struct

SAMPLE_BYTES=74268
RECORD_BYTES=SAMPLE_BYTES+32
PARAMS=struct.pack("<8I",1025,512,1032,1032,1032,1032,0,0)
HEX=re.compile(r'[0-9a-f]{64}\Z')
HEADER=re.compile(r'# timestamp_valid_bits=(\d+) period_ns=([0-9.eE+-]+) queue_family=(\d+)\Z')

def parse(journal,raw,validator,*,samples=50):
    if type(samples) is not int or samples<50: raise ValueError('sample count')
    lines=journal.decode('ascii').splitlines()
    if len(lines)!=2+samples*3 or lines[0]!='# unqualified engineering diagnostics; ticks and host nanoseconds': raise ValueError('incomplete journal')
    m=HEADER.fullmatch(lines[1]);
    if not m: raise ValueError('timestamp metadata')
    bits=int(m[1]);period=float(m[2]);queue=int(m[3])
    if not 1<=bits<=64 or not math.isfinite(period) or period<=0: raise ValueError('timestamp capability')
    if len(raw)!=samples*3*2*RECORD_BYTES: raise ValueError('raw readback size')
    result=[];baseline=None
    for i,line in enumerate(lines[2:]):
        fields=line.split()
        if len(fields)!=10 or not HEX.fullmatch(fields[9]): raise ValueError('sample format')
        try: run,sample,slot,t0,t1,t2,t3,bound,transfer=map(int,fields[:9])
        except ValueError as exc: raise ValueError('sample numeric fields') from exc
        if (run,sample,slot)!=(0,i//3,i%3): raise ValueError('ABA order')
        ticks=(t0,t1,t2,t3);mask=(1<<bits)-1
        if bound<=0 or transfer<=0 or bound>=math.ldexp(period,bits): raise ValueError('sample bound')
        if any(t<0 or t>mask for t in ticks): raise ValueError('timestamp range')
        for a,b in ((t0,t3),(t1,t2)):
            delta=(b-a)&mask
            if not delta or delta*period>bound: raise ValueError('ambiguous timestamp')
        pos=i*2*RECORD_BYTES;resident=raw[pos:pos+SAMPLE_BYTES];transfer_raw=raw[pos+RECORD_BYTES:pos+RECORD_BYTES+SAMPLE_BYTES]
        if raw[pos+SAMPLE_BYTES:pos+RECORD_BYTES]!=PARAMS or raw[pos+RECORD_BYTES+SAMPLE_BYTES:pos+2*RECORD_BYTES]!=PARAMS: raise ValueError("parameter readback mismatch")
        if hashlib.sha256(resident).hexdigest()!=fields[9] or resident!=transfer_raw: raise ValueError('readback mismatch')
        validator(resident)
        result.append({'run':run,'sample':sample,'slot':slot,'ticks':ticks,'host_bound_ns':bound,'transfer_ns':transfer,'sha256':fields[9]})
        if slot==0:baseline=resident
        elif resident!=baseline:raise ValueError('ABA output mismatch')
    return {'timestamp_valid_bits':bits,'period_ns':period,'queue_family':queue,'samples':result}
