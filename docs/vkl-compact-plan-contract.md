# VKL batched compact: proposed scalar32 byte ABI

This is a **design-only host plan**, not a dispatchable kernel, qualified GPU port,
release interface, or downstream migration. The frozen donor and independent
CPU reference retain provenance and separate licensing as documented by
`src/sintergpu/compact_plan.py`.

The caller supplies B, N, capacity C, value shape `[B,N,...]`, item byte width,
nonnegative byte strides and base offsets for values, byte-valued 0/1 masks,
and optional int32/int64 actions. Shape and stride pairs must be uploaded as a caller-owned contiguous `uint2`
descriptor buffer: one pair `(extent, byte_stride)` per input dimension,
including batch and source-item dimensions, with an 8-byte stride. The
`shape_strides_u32` plan value is the interleaved scalar32 representation. Source ranges are allocated as complete
uint32 words; the final word contains any caller padding needed for an aligned
word read. Read-only inputs may overlap, including zero strides. Negative strides,
object/reference dtypes, non-byte-addressable storage and input/output aliasing
are not covered. The byte ABI treats the low-address byte as the low 8 bits of
a word and copies opaque little-endian value bits; dtype interpretation stays
with the caller. All dimensions, addresses, lengths, metadata offsets and device
range/workgroup limits are checked on the host and must fit the declared uint32
ABI. Each candidate stage lists the exact reflected binding names, shared 15-field
`CompactParams` values, and source path. All reflected buffers require a valid
nonzero physical allocation: an absent action input uses a caller-owned four-byte
`action_words` read-only dummy, and any other zero-logical-length bound resource
uses its own four-byte allocation. Writable dummy ranges remain disjoint.
A zero-length values input needs a `value_words` dummy if the gather stage is
present. The plan records logical and minimum allocated byte sizes separately.
Base offsets and strides are in bytes within each whole-buffer binding; all
read-word boundaries must fit the supplied padded range.

The caller validates source mask values; unsupported masks are not silently
reinterpreted.

The intended sequence, separated by an explicit global storage barrier before
each nonempty stage, is:

1. `compactInit`: one lane per slot or batch initializes slot-to-source maps to
   `0xffffffff`, index/valid/action outputs to zero, and each int64 batch count
   to zero. This also handles N=0 and C=0 without accessing input.
2. `compactFlags`: one lane per source item reads its logical mask byte using
   the B,N mask strides; writes uint32 flags as the scan `values` buffer, and
   nondecreasing batch IDs as `segment_ids`. Both buffers have B*N elements.
3. Existing `scan_uint.slang` passes from `scan_layout(B*N)` calculate inclusive
   per-batch ranks using caller-owned scratch. Each pass binds its own `count`
   and `product=0`; the source is the uint32 scan specialization. The plan's
   bindings are the exact scan-layout labels; explicitly bind `values`, `segment_ids` and `output` to the
   flags, batch IDs and final rank arrays. Their own license and notice remain.
4. `compactScatterMetadata`: for each selected source item with rank <= C,
   writes the unique slot-to-source map, low/high index words, valid uint32,
   and optional action word(s). Each destination slot has one writer. Actions
   use B,N byte strides and little-endian byte reads; absent actions remain zero.
   Source index and truncated count high words are zero under the uint32 ABI.
5. `compactFinalizeCounts`: one lane per batch reads the last rank if N>0,
   writes `min(rank,C)` as low/high int64 words. Empty and capacity-zero batches
   retain initialized zeros. This is a *retained count*, not the untruncated total.
6. `compactGatherWords`: one lane per output uint32 word computes its four byte
   positions. For each in-range byte it derives batch, destination slot, trailing
   element and byte-within-item, consults slot-to-source map, resolves the full
   strided source byte address, then loads the containing source uint32 and
   extracts the appropriate byte. Invalid slots, output bytes beyond logical
   length, and padding become zero. Owning the whole word prevents races when
   bytes from two elements, slots or batches share a word; the final padding
   bytes are explicitly zero.

All output and scan scratch buffers, compiled kernels, execution context,
barriers, dispatch, submission and synchronization remain caller-owned. Output
values are tightly packed; indices/counts are logically int64 represented by two
little-endian uint32 words, valid is logically bool represented by uint32, and
codes preserve int32/int64 bits. Device storage buffer limits apply to physical
padded ranges, not just logical byte lengths. Read-only input ranges may overlap each other, including zero-stride views.
Every writable output or scratch range must be disjoint from all other input,
output and scratch ranges; repeated scan labels identify the same intentional
resource across passes. The current fixture binds whole Buffer objects: it rejects nonzero binding
offsets and requires declared ranges to cover the whole Buffer. It rejects a
shared Buffer identity whenever either role is writable. A future adapter may
support subranges only when actual verified views are bound. The runner must
verify backing allocation identities/ranges independently; the host plan cannot
prove them. The fixture checks `shape_strides.struct_size == 8`, but cannot
verify device contents without a readback: the runner must upload the exact
`shape_strides_u32` values and verify their contents outside the recording path
before dispatch.

Forward compaction alone is not differentiable integration: a backward operation
must scatter output gradients to input logical positions and **accumulate** for
aliased/zero-stride views, with its own contract, determinism and gradient tests.
Until that exists, no autograd support is claimed. Both compiler ABIs, CPU/donor
differentials including packed and strided views, synchronization and guard
checks, both target GPUs, and controlled pipeline/transfer timing remain gates.
