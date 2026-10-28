# Experimental VKL batched compaction

This is an opt-in experimental source profile, not a drop-in Torch operator,
a stable adapter, or a qualified performance replacement. It compacts `[B,N,...]`
values in stable source order into fixed `[B,C,...]` capacity. Selected items
beyond capacity are discarded, and unused slots are zero-filled. The result
includes original per-batch source indices, validity, optional int32/int64
codes, and **retained** counts `min(selected,C)`. Empty batches and capacity zero
are supported within the validated scalar32 bounds.

## Origin and sources

The numerical donor is `src/vkl/kernels/reference/compact.py` from
[bjoernellens1/vkl at 3384023](https://github.com/bjoernellens1/vkl/tree/3384023a311651ea4260cf12ddab1567972f89f3),
retained as [the frozen reference](../references/donors/vkl/src/vkl/kernels/reference/compact.py)
with SHA-256 `f4a323c1940f32a61a50dbe61d4669658804d7e82df040fd7d7940ac4bb62aff`.
[Provenance](../provenance/vkl.json) records owner authorization, authorship and
Apache-2.0 terms; the [inventory](../provenance/vkl-candidates.json) records
source details. The experimental forward shader is being packaged at
`src/sintergpu/kernels/vkl/compact_batched.slang`; its scan passes depend on
`src/sintergpu/kernels/primitives/scan_uint.slang`, which retains its separate
MIT notice. The experimental logical backward CPU-observed profile is being
packaged at `src/sintergpu/kernels/vkl/compact_backward_torch_cpu_profile.slang`.
These relocated sources require their own GPU qualification.

The host planner, [`plan_compact`](../src/sintergpu/compact_plan.py), retains a
design-only plan and candidate source paths. The experimental
[`PreparedCompact`](../src/sintergpu/compact_record.py) maps recognized stages to
packaged catalog operations and records them on a caller-owned encoder. It does
not allocate buffers, upload data, submit commands or wait. Its availability does
not qualify the relocated shaders for GPU execution.

## Layout and ownership

Values use a little-endian opaque-byte ABI with nonnegative byte strides;
mask bytes must be 0 or 1. Underlying byte-addressed storage must be padded to
whole uint32 words. The descriptor contains `(extent, byte_stride)` uint32
pairs at an eight-byte struct stride. Indices and counts are represented by
little-endian low/high uint32 words, validity by uint32, and actions by their
int32/int64 bit patterns. Counts are retained counts, unlike the separate
primitive's total count. Full buffer bindings, size and uint32 address bounds,
256-thread workgroup limits and dispatch limits must be validated against the
actual device. Zero-length bindings need valid independent four-byte dummies.

The caller owns inputs, output and scratch, device, command encoder, submission,
synchronization and resource lifetime. The recorder rejects any writable resource sharing an allocation with another
resource; read-only input ranges may overlap. Buffer handle inequality by
itself does not establish independent physical allocations. A storage barrier is
required before every nonempty stage and when subsequently consuming outputs.
The order is initialization, mask flags, segmented scan passes, metadata scatter,
count finalization and word-owned value gather. Each scan pass uses its own
planned bindings and parameters. Consult the full [plan contract](vkl-compact-plan-contract.md)
for limits, scratch and empty-case details.

### Python recording sketch

This forward-only experimental interface requires an explicit opt-in. The caller
must prepare whole buffers and verified allocation ranges, upload and independently
verify the descriptor words, and supply the actual device limits. The boolean
attestations record those caller checks; they cannot establish physical allocation
identity, GPU descriptor contents, queue/device compatibility or buffer lifetime.
Buffers, inputs and descriptor must remain stable until execution completes.

```python
from sintergpu.compact_record import PreparedCompact

# plan = plan_compact(..., **limits); buffers, ranges and prebound kernels are
# prepared by the caller. Kernels are keyed by catalog operation ID.
# Independently verify uploaded descriptor words, backing allocation identities,
# disjoint writable resources, and each compiled kernel's source/device/ABI.
graph = PreparedCompact(
    plan, buffers, ranges, limits=limits,
    descriptor_words=plan["shape_strides_u32"],
    allocations_verified=True, descriptor_verified=True,
    allow_experimental=True, kernels=kernels,
    prebound_identity_attested=True,
)
graph.record(caller_encoder)  # caller-owned encoder; barriers and dispatches
# Caller records a consumer barrier, submits, and manages completion/lifetimes.
```

Alternatively pass an experimental `SlangPyAdapter` as `adapter` instead of
`kernels`, with `runtime_identity` containing the caller's expected SHA-256 hashes
under `compiler_sha256`, `runtime_sha256` and `extension_sha256`. Preparation
checks the installed files, catalog source and source closure, requires empty
adapter cache entries for requested operations and prepares the kernels. Call
`graph.verify_source_identity()` explicitly when a filesystem recheck is wanted;
it reads files and is only available on the adapter path. `record()` checks the
prepared adapter cache identity and performs no file I/O. Source or runtime file
changes after preparation are not automatically detected; explicit rechecks do
not eliminate time-of-check/time-of-use races. File hashes do not attest dynamic
linker mappings or compilation options. On the prebound path, compiled identity
is entirely caller-attested, and `verify_source_identity()` is unavailable.

### C++ integration sketch

C++ consumers can compile the Slang entry points and scan source, then implement
the equivalent checked resource/dispatch plan in their own runtime. There is no
provided C++ compact planner or adapter yet. Pseudocode:

```cpp
// Caller validates sizes, byte strides, allocation ranges, shader identity,
// device limits and uploads the uint2 shape/stride descriptor.
for (const Stage& stage : validated_stages) {
    encoder.storage_barrier();
    bind_precompiled_kernel_and_resources(encoder, stage);
    dispatch(encoder, stage.groups);
}
encoder.storage_barrier(); // before consuming the compacted outputs
// The caller submits and manages completion and resource lifetime.
```

`validated_stages` here is application-provided pseudocode, not an existing C++
API. Port the exact reflected bindings and recursion from the Python plan; do not
infer scan dispatches from only the original input length.

## Backward and evidence boundaries

A separate experimental backward candidate scatters dense logical gradients
using the forward slot map. Its CPU-observed Torch bit profile changes negative
zero to positive zero and quiets signaling NaNs per floating component while
retaining sign and payload. Supported components are float16/32/64 and complex64/128.
It requires a valid, unique, in-range per-batch forward slot map, packed upstream
gradients, disjoint resources and explicit stage barriers. It does not perform
view/base accumulation, wire Torch autograd, or establish Torch GPU semantics.

The original-path forward candidate at commit `c389f32edc45d1660f2b38c506baa5e6ddf6bb4c`
passed six fixture cases on both [Radeon 8060S](../qualification/evidence/vkl-compact-correctness-8060s-c389f32/report.json)
and [RTX 4070 Ti](../qualification/evidence/vkl-compact-correctness-4070ti-c389f32/compact-correctness.json).
The retained RTX 4070 Ti [backward CPU-profile probe](../qualification/evidence/vkl-compact-backward-profile-4070ti-3c5b997/compact-backward-profile.json)
is separate. These probes use fixture readbacks and guards; physical backing
allocation disjointness was not verified. They are correctness probes, not
measurements or qualification of the relocated sources. CPU differential tests
and compiler checks add evidence for the tested contracts, not full-domain proof.
Radeon backward-profile execution, relocated-source GPU checks, controlled
kernel/pipeline/transfer performance, actual framework interop, view accumulation
and downstream adoption remain pending. Do not switch a consumer default until
its parity and no-regression gates pass.
