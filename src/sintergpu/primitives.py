"""Explicit borrowed-encoder recipes; no allocation, queue, readback or runtime imports."""
from dataclasses import dataclass


@dataclass(frozen=True)
class ScanPass:
    entry: str
    count: int
    bindings: dict


def scan_layout(count, *, exclusive=False):
    """Return buffer element counts and ordered passes for a segmented scan.

    Buffers are scalar32 except segment ID buffers (uint32). All bindings must be
    separate allocations/ranges unless labels are identical. The caller allocates
    once, supplies initialized `values` and nondecreasing `segment_ids`, and uses
    kernels compiled with a uniform scalar type. Scratch never requires host zeroing:
    scanBlocks writes all subsequently read elements, including partial block ends.
    """
    if not isinstance(count, int) or not 0 <= count <= 0xffffffff:
        raise ValueError('count must fit uint32')
    sizes = {'values': count, 'segment_ids': count, 'output': count}
    passes, levels = [], []
    n, level = count, 0
    while n:
        values = 'values' if level == 0 else f'level{level}_values'
        ids = 'segment_ids' if level == 0 else f'level{level}_ids'
        output = 'output' if level == 0 else f'level{level}_output'
        blocks = (n+255)//256
        totals, block_ids = f'level{level+1}_values', f'level{level+1}_ids'
        sizes.update({values:n, ids:n, output:n, totals:blocks, block_ids:blocks})
        passes.append(ScanPass('scanBlocks', n, dict(values=values,segmentIds=ids,output=output,blockValues=totals,blockIds=block_ids)))
        levels.append((n, ids, output, block_ids, blocks, level))
        if blocks == 1:
            break
        n, level = blocks, level+1
    for n, ids, output, block_ids, blocks, level in reversed(levels):
        if blocks > 1:
            passes.append(ScanPass('scanCarry',n,dict(segmentIds=ids,output=output,blockIds=block_ids,scannedBlocks=f'level{level+1}_output')))
    if exclusive:
        sizes['exclusive_output'] = count
        if count:
            passes.append(ScanPass('scanExclusive',count,dict(segmentIds='segment_ids',inclusiveValues='output',output='exclusive_output')))
    return sizes, passes


def record_scan(kernels, buffers, count, *, command_encoder, max_workgroups_x, product=False, exclusive=False):
    """Record into a caller-owned encoder. Caller establishes pass write/read hazards.

    The runtime adapter must guarantee storage visibility between recorded dispatches
    (SlangPy's command encoder does this). No dispatch is issued for zero elements.
    """
    _, passes = scan_layout(count, exclusive=exclusive)
    if not isinstance(max_workgroups_x, int) or max_workgroups_x < 1:
        raise ValueError('supply the actual adapter maximum X dispatch workgroups')
    if any((step.count+255)//256 > max_workgroups_x for step in passes):
        raise ValueError('scan exceeds adapter X dispatch limit; partition before recording')
    for step in passes:
        kernels[step.entry].dispatch(thread_count=[((step.count+255)//256)*256,1,1],
            vars=dict(count=step.count, product=int(product), **{k:buffers[v] for k,v in step.bindings.items()}),
            command_encoder=command_encoder)
