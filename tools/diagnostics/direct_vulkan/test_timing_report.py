# SPDX-License-Identifier: Apache-2.0
import hashlib
import pytest
import struct
from timing_report import parse,SAMPLE_BYTES

def test_complete_and_fail_closed():
    data=b'x'*SAMPLE_BYTES;digest=hashlib.sha256(data).hexdigest();header=['# unqualified engineering diagnostics; ticks and host nanoseconds','# timestamp_valid_bits=32 period_ns=1 queue_family=0']
    rows=[f'0 {i} {s} 1 2 3 4 100 50 {digest}' for i in range(50) for s in range(3)]
    journal=('\n'.join(header+rows)+'\n').encode();raw=(data+struct.pack('<8I',1025,512,1032,1032,1032,1032,0,0))*(50*3*2);seen=[]
    assert len(parse(journal,raw,lambda x:seen.append(len(x)))['samples'])==150 and len(seen)==150
    with pytest.raises(ValueError,match='incomplete'):parse(b'\n'.join(journal.splitlines()[:-1]),raw,lambda x:None)
    with pytest.raises(ValueError,match='raw readback size'):parse(journal,raw[:-1],lambda x:None)
    altered_param=bytearray(raw);altered_param[SAMPLE_BYTES]=0
    with pytest.raises(ValueError,match='parameter readback mismatch'):parse(journal,bytes(altered_param),lambda x:None)
    altered=bytearray(raw);altered[100]=0
    with pytest.raises(ValueError,match='readback mismatch'):parse(journal,bytes(altered),lambda x:None)
    swapped=rows.copy();swapped[1],swapped[2]=swapped[2],swapped[1]
    with pytest.raises(ValueError,match='ABA order'):parse(('\n'.join(header+swapped)+'\n').encode(),raw,lambda x:None)
    wrapped=rows.copy();wrapped[0]=wrapped[0].replace(' 100 50 ',' 4294967296 50 ')
    with pytest.raises(ValueError,match='sample bound'):parse(('\n'.join(header+wrapped)+'\n').encode(),raw,lambda x:None)

    variant=b'y'*SAMPLE_BYTES;variant_hash=hashlib.sha256(variant).hexdigest();changed=bytearray(raw)
    offset=2*(SAMPLE_BYTES+32);changed[offset:offset+SAMPLE_BYTES]=variant;changed[offset+SAMPLE_BYTES+32:offset+2*SAMPLE_BYTES+32]=variant
    changed_rows=rows.copy();changed_rows[1]=changed_rows[1].replace(digest,variant_hash)
    with pytest.raises(ValueError,match='ABA output mismatch'):parse(('\n'.join(header+changed_rows)+'\n').encode(),bytes(changed),lambda x:None)
