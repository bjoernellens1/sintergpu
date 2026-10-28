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
query reference ownership and exception unwind. It does not exercise actual
specialization-cache resolution, real Vulkan command recording, failed
`vkEndCommandBuffer`, reset/reuse, callbacks, indirect dispatch or a driver.
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
