# Experimental recorded-dispatch association

These new RHI and SlangPy patches layer on the isolated shader-byte observation
candidate and all earlier creator/allocation/device-limit patches. They are not
released runtime code. Preserve the upstream `Apache-2.0 WITH LLVM-exception`
license and prior attribution notices. The exact layered input/output source
hashes are listed in `test_rhi_dispatch_association_patch.py`.

The Vulkan backend resolves a virtual compute pipeline to a concrete pipeline
before command recording. At successful concrete pipeline creation, this
candidate copies the recorded shader-module bytes only when its single compute
stage matches that shader module, its Vulkan entry point is `main`, no Vulkan
specialization-info payload is present, and the pipeline-binary cache was not
hit. A direct `vkCmdDispatch` record copies the concrete pipeline handle,
shader-module handle, module bytes and group counts to its command-buffer
observation. Indirect compute dispatch and command callbacks invalidate the
entire observation; failed command-buffer finalization and pool reset leave it
unavailable. These restrictions are intentionally conservative.

The optional COM query retains the command buffer, and the SlangPy binding
copies values with RAII ownership. The caller must serialize queries with
recording, submission, reset, pool reuse, reload and close. A copied Python
record is historical, not a lease. Generation identifies this instrumented
recording, not global driver identity. This is an observation of what was
*recorded*, not proof of queue submission, GPU execution, descriptor contents,
provider identity, physical allocation backing, immutable loaded runtime, or
source/compiler qualification. Foreign native Vulkan operations are outside the
controlled-process contract. The added host byte copies and instrumentation
are not performance equivalent; allocation failures in the diagnostic path may
alter failure behavior. The existing point-plane production gate remains closed.

Local isolated Debug RHI/SlangPy build/link passed with GCC 16.2.1, Python
3.13.12 and `SGL_WARNINGS_AS_ERRORS=OFF`. The C++ CPU probe uses fake Vulkan
teardown callbacks and synthetic concrete pipeline/program state: it covers two
pipeline handles and group tuples in order, unavailable snapshots, output schema,
query reference ownership and exception unwind. It does not itself exercise actual specialization-cache resolution or a driver.
A second CPU-only mock probe compiles the actual command recorder and calls the
actual `createComputePipeline2` with fake Vulkan callbacks. It checks direct
bind/dispatch, an indirect call and callback after a direct dispatch,
finalization failure, reset failure, reset/reuse, creation failure and
non-null Vulkan specialization-info rejection. It does not initialize a
Vulkan loader/device or exercise a real driver, actual specialization/cache
resolution or GPU execution.
A third CPU-only probe seeds two distinct specialization keys and concrete
pipelines into the actual in-memory shader cache, invokes the actual command
encoder cache lookup and recorder, and checks the resulting pipeline handles,
shader bytes, dispatch order and group sizes. It does not compile or create
specializations, exercise cache misses or population, or exercise the separate
persistent pipeline-binary cache. It does not initialize a Vulkan device or
execute on a GPU.
The SlangPy CPU-device smoke confirms the query is unavailable before and after
close. Local source-application tests skip without the audited source; they do
not establish CI or build provenance. No GPU execution has been performed.

Patch SHA-256: RHI
`7b8cfda9c889e750c7a6af3e6e78d8ee5b2a005924a5745b8e22044253410aa6`;
SlangPy `d28017a204a917aacf12bb24a40ab2ab9939d62ad5febcf0dda928bc70ecece7`.
Local build `libsgl.so` SHA-256
`f4041769e9bf96debb5f3c5a6a7fe25207ea8b2a7002633db9552309eac99861`;
extension SHA-256
`94af2d6c9bdadb0711500648c511385539be6971fd137d8faebd810c3f1cc111`.
These do not authenticate another host or build. Review this candidate and the
entire layer chain independently before any hardware use.


The mock-Vulkan probe is `rhi-dispatch-mock-vulkan-cpu-probe.cpp`; its source
SHA-256 is `50e2a367acf1c1bfd03f807a542b2dcb1ed9d44050d2d04b2adfe2a6b44895ef`.
Locally it was linked with the isolated diagnostic RHI and executed with both
Vulkan ICD environment variables pointing to nonexistent files; it exited 0.
`ldd` showed no linked Vulkan, NVIDIA or AMD library. Its local executable SHA-256
was `9563e4687c1c83416f24932baf971cb2c09558122d6d4960c2b762533b8789c3`;
that hash does not authenticate another build.

The seeded-cache probe is `rhi-specialized-cache-mock-cpu-probe.cpp`, SHA-256
`b51ca8e80a4800a53fa604410d4e58067e0b3034b4b0bc50a45b117e3165a5e5`.
It was built in the isolated Debug diagnostic lab by adding an executable target
linked to `slang-rhi`, with the patched RHI source and pinned Vulkan headers
on its include path, then running:

```sh
cmake --build /dev/shm/sintergpu-dispatch-observe-lab/build --target rhi_specialized_cache_mock_cpu_probe
VK_DRIVER_FILES=/nonexistent/sintergpu-no-vulkan-icd.json VK_ICD_FILENAMES=/nonexistent/sintergpu-no-vulkan-icd.json /dev/shm/sintergpu-dispatch-observe-lab/build/rhi_specialized_cache_mock_cpu_probe
```

The local build and independent no-ICD rerun exited successfully. The local
executable SHA-256 was
`23cef06fca22ee701692d0f019a94c41e01b819f501867d56eb9f4236def94aa`;
this identifies that executable only, not another build or runtime. The seeded
cache is synthetic and does not establish specialization generation, driver
behavior, shader execution, or performance qualification.
