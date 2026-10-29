# SPDX-License-Identifier: Apache-2.0
import pytest
from timing_transfer_components_report import validate_components


def records():
    samples=[]; parts=[]
    for i in range(150):
        prefix=f'0 {i//3} {i%3}'
        samples.append(f'{prefix} 1 2 3 4 9 10 ' + 'a'*64)
        parts.append(f'{prefix} 10 1 2 3 4')
    return ('\n'.join(samples)+'\n').encode(), ('\n'.join(parts)+'\n').encode()


def test_complete_and_mismatches():
    samples,parts=records(); result=validate_components(samples,parts)
    assert len(result['records'])==150
    for broken in (parts.rsplit(b'\n',2)[0]+b'\n', parts.replace(b'0 0 1 10',b'0 0 2 10',1), parts.replace(b'10 1 2 3 4',b'10 1 2 3 5',1)):
        with pytest.raises(ValueError): validate_components(samples,broken)
