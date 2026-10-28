# Draft allocation query patches: incomplete, unavailable

These partial source patches are review artifacts only. The RHI patch proposes a
separate versioned COM query interface without changing `IBuffer`'s vtable. It is
based on the EliGSiR creator-receipt-patched RHI header (SHA-256
`516a4390911cae8b3ca93e9191ef145c39f232aaef80fa9f4c8c1ed2111f9111`), itself
based on Slang RHI commit `1a9768741246b1f356116d5dfa305933034ce049`. The
SlangPy patch applies to the audited `resource.cpp` SHA-256
`a1d0082a7440f17d4a51c231fcc14ffb66088e8fdb869058d4a21aec735335d4`
from the SlangPy 0.43.1 source set. Modified files carry a SinterGPU notice;
both upstream projects declare `SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception`.
Retain their complete license texts and notices and the existing EliGSiR patch
notices when distributing any patched source or binary. The local-only patch
application tests require the audited source checkout; skips mean no source
application validation and do not constitute CI proof.

Neither patch implements the interface. The Python property always returns
`None`, including for ordinary buffers. In particular it does not query or
cast a private backend object. There is no compiled runtime, no mapping, no
selected-device or provider receipt here and no GPU/CPU performance result.

Before an operational implementation: resolve the pinned COM aggregation and
reference-count implementation, arrange a live owner generation and copied
allocation size only after a successful bind, reject every unsupported path,
connect the retained interface to a SlangPy value copy with close/release
serialization, and bind it to the separately created same-device UUID and
instance dispatch observations. Review externally exposed handles and document
that the controlled fixture prohibits foreign native mutations; this is not a
universal interception guarantee. The interface UUID, ABI layout, exception and
thread rules remain provisional. Compile, independent source review, device and
runtime tests are mandatory before use.

## Separate experimental implementation patches

`rhi-allocation-observation-experimental.patch` and
`slangpy-allocation-observation-experimental.patch` are a different, unqualified
candidate. The former adds an optional COM query on the Vulkan buffer object,
records the allocation requirement and makes the observation available only
after an ordinary non-exportable buffer is bound. Native-wrapped and shared
buffers are unavailable; calling the backend shared-handle method taints future
queries. The latter queries this interface and returns a copied Python value;
it returns `None` after SGL close or when the query fails.

The COM interface retains the underlying RHI object, but it can outlive logical
SGL close. A Python dictionary is not a lease. A controlled caller must keep
the buffer and device alive, serialize close/release and native operations,
and requery immediately before and after use. Native Vulkan operations outside
this backend can escape observation; this does not prove physical backing
independence or forbid foreign changes. Associate observations with the existing
same-device UUID creation receipt separately; provider identity, runtime/binary
identity and cross-receipt association are not established by these patches.
Neither implementation provides GPU qualification or performance equivalence.

The isolated Debug build and link succeeded with `SGL_WARNINGS_AS_ERRORS=OFF`;
Python 3.13/glibc feature-test macro redefinition warnings remain. The linked
CPU C++ probe constructs the actual patched buffer implementation without a
Vulkan device and uses synthetic handles. It checks untracked rejection, field
copying, explicit taint, and query-interface reference retention/destruction;
it does not test an actual Vulkan allocation or exercise actual export calls.
A separately built SlangPy CPU device returns `None` for the query both before
and after SGL close. Explicitly set `LD_LIBRARY_PATH` to the isolated build:
without it an unrelated installed `libsgl.so` was selected. Tests did not
initialize a GPU. Device ownership, shared/export, wrapped-resource, concurrency,
and actual Vulkan execution remain unverified. The CPU model tests exercise a
simulation only. These patches need further independent review and must not be
used as an allocation attestation.

The experiment's build used CMake 3.24+, Ninja, `/usr/bin/c++` (GCC 16.2.1)
and `/home/bjoern/miniforge3/bin/python` (3.13.12), plus the existing isolated
EliGSiR pinned dependency, Vulkan-header and Slang 2026.12 source facade. Configure
as Debug with `-DSGL_WARNINGS_AS_ERRORS=OFF`, and target `slangpy_ext` and a
`rhi_receipt_cpu_probe` executable linked against the same `slang-rhi` target.
The local probe source is `rhi-allocation-observation-cpu-probe.cpp`; the SGL
smoke is `slangpy-allocation-cpu-smoke.py`, run with the isolated source package
on `PYTHONPATH` and its build directory on `LD_LIBRARY_PATH`. The normal
creator-receipt patches must first be applied to the exact pinned RHI/SlangPy
sources; then check and apply these patches before configuring. Source patches
and test results do not authenticate an independently built runtime.

For source identity the experimental RHI patch expects, in addition to the
creator-receipt-patched header hash listed above, `src/vulkan/vk-buffer.h`
SHA-256 `78b0414ace467f49b19666aa545162fe90eefcb9355af48041dd699b71f49f52`
and `src/vulkan/vk-buffer.cpp` SHA-256
`38823e393d4208bbf1bd951dcfef6f02ac58ff3432b0ef747c5b505cfb456d31`.
The SlangPy binding input hash is listed above. Local applied-source hashes were
`861508fe63a9d496f299bd7d5f697e5a4bb3947fe7672182bfe016d6e36d18cb` (RHI
header), `18f7c6603d0e9fb4fc217a66e4e94738df71dd5718b8e29858d03236eef7e381`
(buffer header), `bbb2e85eb9efab04675a7bbd04f681740a5c8979bf7ac3e1b7af9bf212aa787e`
(buffer source), and `571d212d21adf5361eeb76a80e46eb1872af78eab092e028de5bf6c0be225081`
(SlangPy binding). Exact outputs depend on the documented input and patch bytes.

The local parent CMake harness added:

```cmake
add_executable(rhi_receipt_cpu_probe /absolute/path/to/rhi-allocation-observation-cpu-probe.cpp)
target_include_directories(rhi_receipt_cpu_probe PRIVATE "${sdk_root}/rhi-source/src" "${sdk_root}/dependency-source/vulkan-headers/include")
target_link_libraries(rhi_receipt_cpu_probe PRIVATE slang-rhi)
```
