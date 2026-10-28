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

## Review-required Vulkan cases

`rhi_allocation_vulkan_cases.py` is an opt-in case function, not a standalone
runner. It uses a borrowed canonical exclusive lease, compares the same live
backend's typed instance/physical-device/device creation handles, UUID and raw
Vulkan driver version, and exercises two ordinary buffers, a shared buffer,
and SGL-close rejection. Shared-buffer creation failures fail the run. The
cases do not invoke shared-handle export: an ordinary buffer's memory is not
exportable, so doing that would violate Vulkan valid usage. Wrapped/imported
buffers cannot be constructed through this SGL interface and remain pending.
Real shared-handle export, handle reuse, and GPU-side lifecycle are not tested.

No reviewed runner exists yet. Before calling the cases, a future runner must
hold the canonical lease before creating the device and until teardown; verify
exact source and binary closure, loaded `libsgl` and extension mappings, Vulkan
loader/ICD and driver; reject ambiguity; capture write-once success and failure
receipts; and prohibit foreign native Vulkan mutation. The case function's
borrowed lease check does not prove the lock was held at device creation or
verify any runtime binary, ICD, loader, provider, or source identity. Matching
live observed memory handles is not physical backing-independence proof.
`test_rhi_allocation_vulkan_cases.py` uses only CPU fake devices and confirms
fixture control flow; no GPU or real Vulkan path has been tested.

`rhi_allocation_vulkan_runner.py` is a fail-closed orchestration scaffold. Its
production preflight always raises before device creation because the existing
instrumented backend does not expose its actual instance dispatch provider and
there is no verified immutable runtime/loader/ICD closure. It therefore cannot
collect Vulkan results yet. CPU mock tests cover lease ordering, case-owned
teardown, failure receipts, write-once output and final-lease failures. A future
implementation must replace the preflight with an independently reviewed,
source-backed provider observation and exact loaded binary/ICD identity checks;
passing caller-supplied identity strings or inspecting a separately loaded Vulkan
library is insufficient. The scaffold must remain unqualified until that work
and real-device review are complete.

## Separate exploratory smoke (review required)

`rhi_allocation_exploratory_smoke.py` is a separate, unqualified scaffold.
Production execution is unconditionally blocked before runtime import or device
enumeration pending independent provenance review; only injected CPU mocks can
exercise the flow. A future reviewed implementation would require `--execute`, an explicit single `VK_DRIVER_FILES` setting, a single
visible adapter matching the requested vendor/device, the same backend's
creation receipt matching expected UUID and raw driver version before any
buffer allocation, and the canonical lock from before runtime import through
teardown. It hashes requested binary files and checks that the extension,
`libsgl` and compiler have matching mapped path and inode before and after
the cases. It records mapping and stat device numbers separately: overlay
filesystems can report different devices, so their equality is not established. These hashes are caller-supplied comparisons, not trusted
build provenance. The ICD JSON hash identifies a configuration file only;
loaded loader, ICD binary, dispatch provider, complete closure and physical
allocation independence are not established. Only an `exploratory_only` or
failure receipt can result. No timings or qualification result are produced.

For a future reviewed run, first build and identify the instrumented runtime on
the target host, restrict that host to one target ICD/adapter, use a fresh output
path and provide exact binary paths and lowercase SHA-256 values. Set the target
runtime's `PYTHONPATH` and `LD_LIBRARY_PATH`; provide `--uuid`, `--driver`,
`--vendor`, `--device`, `--extension`, `--extension-sha`, `--libsgl`,
`--libsgl-sha`, `--compiler`, `--compiler-sha`, and `--output`, after a future reviewed implementation removes the production block. Expected UUID/driver must
come from independently collected device evidence. The local build is not a
build or identity receipt for the 4070 workstation. The CPU fake tests do not
exercise Vulkan. No GPU execution has been performed for this smoke.

## Same-backend physical limits and dispatch-address candidate

The additional `rhi-device-limit-observation-experimental.patch` and
`slangpy-device-limit-observation-experimental.patch` layer on the earlier
allocation-observation experimental patches and the EliGSiR creator receipt.
They are an isolated source configuration, not a compatible replacement for a
released SlangPy/RHI ABI: the output creation receipt changes from schema 1 to
schema 2, and both RHI and SlangPy must be rebuilt together. The existing
receipt *request descriptor* remains schema 1. The schema-2 output captures the
limits from the `VkPhysicalDeviceProperties` already queried by the backend
for its selected physical device, and on Linux x86-64 records the backend's
actual `vkGetInstanceProcAddr` and `vkGetPhysicalDeviceProperties` function
addresses. The Python validation helper rejects missing, invalid or mismatched
receipts. A separate optional `dladdr`/`/proc/self/maps` observation never calls
the recorded pointer; it does not authenticate mapped pages, rule out
interposition or establish whether an address is an ICD rather than a loader
trampoline. Foreign native manipulation and physical backing identity remain
outside the observation. No runtime promotion or GPU execution is authorized by
this candidate.

Layered input SHA-256 values: RHI `include/slang-rhi.h`
`861508fe63a9d496f299bd7d5f697e5a4bb3947fe7672182bfe016d6e36d18cb` and
`src/vulkan/vk-device.cpp` `5ce3c4d5dfd53a4b06cc14ee93a3232edde0f6504836927f6ca9803f6c67d813`;
SlangPy `src/sgl/device/device.cpp`
`1fdd14494516445a6fcbb0212713b56e5ae09dda6956e64563e3a7b26a5066bb` and
`src/slangpy_ext/device/device.cpp`
`0367c195fd70e11e32ec5136dc97616efdeaf2c7110978ff6e9aeef829bb8fd3`.
Modified source carries SinterGPU notices and retains upstream
`SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception` licensing and
existing EliGSiR provenance. The diagnostic consumer is
`rhi_device_limits_observation.py`; its tests use CPU fakes and an ordinary
libc function address only, never initialize Vulkan. Local patch tests require
the audited source and skip if absent; they do not prove CI coverage.

The local isolated paired Debug build linked with GCC 16.2.1, Python 3.13.12
and `SGL_WARNINGS_AS_ERRORS=OFF`; it reported existing Python/glibc macro
redefinition warnings. `slangpy-device-limits-cpu-smoke.py` confirms the
schema-2 binding fields exist and a CPU device returns no Vulkan receipt before
or after close. The CPU fake checks do not establish Vulkan field values. The
local build was not executed on a GPU. Local patch SHA-256 values are
`396d99141696c8a0b4cfcd38e99736f5bb9d72f2564389cf7f11fa34c15cf298` (RHI)
and `748ed81ecb03d8c3d2746fcdb5ca91d5389f874950dbb2bda93e9d05e6824e49`
(SlangPy). The resulting local `libsgl.so` was
`0cd4c582faa8405188c520f856b569682c9e57717bbdcfb6e8672ce6576e0e1f` and
Python extension was
`943accb68cb5ba9ed1c2c903f00c5ddedd94ea5f293aba1e2e1e0f8b87b180a4`;
these local hashes do not authenticate another build or host.

To reproduce the local source application and build, start from the separately
acquired EliGSiR creator-receipt-patched `rhi-source`, `slangpy-source`,
`parent-source/CMakeLists.txt`, `dependency-source` and `slang-sdk` for the
pinned configuration described above. On the original host they were under
`/home/bjoern/git/eligsir-refactor-execution/quality-current-receipt-sdk-build-v1`.
That path and its local checkout are not release artifacts. Copy the sources
into a fresh scratch root and verify the **base** identities in the table below
before applying the allocation patches. Then verify the **limit input** hashes
immediately before applying the corresponding limit patches. Finally verify
all **limit output** hashes. The RHI and SlangPy source files listed in the
allocation section above also have to match their documented base and
post-allocation hashes. These checks bind only the named files, not the complete
dependency closure.

| Source file | Creator-patched base SHA-256 | Limit input SHA-256 | Limit output SHA-256 |
|---|---|---|---|
| RHI `include/slang-rhi.h` | `516a4390911cae8b3ca93e9191ef145c39f232aaef80fa9f4c8c1ed2111f9111` | `861508fe63a9d496f299bd7d5f697e5a4bb3947fe7672182bfe016d6e36d18cb` | `5fdfd45c7bed85d4ab98a9116cc50782a55858c011daee6756f1cf0ba2518903` |
| RHI `src/vulkan/vk-device.cpp` | `5ce3c4d5dfd53a4b06cc14ee93a3232edde0f6504836927f6ca9803f6c67d813` | `5ce3c4d5dfd53a4b06cc14ee93a3232edde0f6504836927f6ca9803f6c67d813` | `52a57fc7b08bba8f06402a987da3e535b714785a22018e8a42dc17172baf222b` |
| SlangPy `src/sgl/device/device.cpp` | `1fdd14494516445a6fcbb0212713b56e5ae09dda6956e64563e3a7b26a5066bb` | `1fdd14494516445a6fcbb0212713b56e5ae09dda6956e64563e3a7b26a5066bb` | `020948db4622a3e16135791376215e39629241ba7b9d48fab0aff8f78ce9a04d` |
| SlangPy `src/slangpy_ext/device/device.cpp` | `0367c195fd70e11e32ec5136dc97616efdeaf2c7110978ff6e9aeef829bb8fd3` | `0367c195fd70e11e32ec5136dc97616efdeaf2c7110978ff6e9aeef829bb8fd3` | `39917999f068d437a7b6714110458c3d95e7a9a532aab57e83d3bbd295d53721` |

Apply the RHI allocation patch in `rhi-source` and the SlangPy allocation patch
in `slangpy-source`, verify the limit-input hashes, then apply the respective
limit patches and verify the limit-output hashes. Point copied
`parent-source/CMakeLists.txt` at those sibling source directories and the
pinned sibling dependencies. Configure and build:

```sh
cmake -S "$scratch/parent-source" -B "$scratch/build" -G Ninja \
  -DPython_EXECUTABLE=/home/bjoern/miniforge3/bin/python \
  -DCMAKE_BUILD_TYPE=Debug -DSGL_WARNINGS_AS_ERRORS=OFF
cmake --build "$scratch/build" --target slangpy_ext -j 3
LD_LIBRARY_PATH="$scratch/build" PYTHONPATH="$scratch/slangpy-source" \
  /home/bjoern/miniforge3/bin/python \
  tools/diagnostics/patches/slangpy-device-limits-cpu-smoke.py
```

`$scratch` denotes the new isolated root, not an installed runtime. This
procedure uses host-specific dependencies and records a CPU-only build; the
checked-in CI runs the fake-receipt and libc address tests, not this build or
the local-source-dependent patch checks. Reconstructing an equivalent build on
another machine requires acquiring and verifying the same complete dependencies
and source identities separately.
