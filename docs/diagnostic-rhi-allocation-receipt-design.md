# Proposed isolated RHI allocation diagnostic (not implemented)

This is a design and source audit, not a patch, runtime receipt, allocation proof,
performance result, or qualification. No runtime was built or executed for it.

The reviewed reference is the EliGSiR Vulkan creation-receipt patch, SHA-256
`9262cb800403035f6ca49e19f61bf0d917ca821185c521da1b73fc495903289c`.
Its preparation script pins Slang RHI commit
`1a9768741246b1f356116d5dfa305933034ce049`, verifies source and license
hashes, and adds modified-file notices. Any future patch must retain that
provenance and the original Slang RHI and Apache notices and use the exact
corresponding SlangPy binding source; do not modify an installed runtime.
The inspected RHI `src/vulkan/vk-buffer.cpp` has SHA-256
`38823e393d4208bbf1bd951dcfef6f02ac58ff3432b0ef747c5b505cfb456d31`.

Ordinary RHI buffer creation allocates memory, binds at offset zero and retains
both handles in `VKBufferHandleRAII`; native-handle buffers take a separate
path, and shared buffers request exportable memory. The RAII destructor frees
both objects, while buffer deletion is deferred. A Vulkan buffer handle alone
therefore neither identifies a physical allocation range nor establishes its
lifetime. The RHI uses its own Vulkan loader and instance dispatch; the actual
backend physical device and function pointers, rather than device name or a
second independently enumerated instance, are the appropriate observations.

A viable separately built diagnostic needs all of the following before use:

* Bind patch, all upstream sources, SlangPy binding, compiler and produced
  binaries to verified identities; fail on source mismatch. Expose a versioned
  API with explicit owned/borrowed lifetime and concurrency rules.
* Observe the backend physical device UUID and instance dispatch addresses at
  creation. Independently capture mapped loader and provider identities; an
  address or path alone is not binary authentication.
* Record successful backend-created buffer and allocation objects, actual
  allocation size, binding offset and buffer extent, device identity, and
  monotonically unique owner generations. Reject unknown/failed, native,
  imported, exported, shared, sparse, aliased or rebound resources. Keep the
  underlying object alive for the receipt's whole use; invalidate on teardown.
* Account for every path that can expose or manipulate native device/buffer
  handles and for deferred deletion, cross-thread races and device loss.
  Foreign native callers can bypass backend observation, so a receipt cannot
  claim exclusive tracking unless the runtime prevents such access or a reviewed
  complete interception boundary proves it.
* Keep diagnostics CPU-only during recording, with no added GPU submission,
  waits, readbacks or ownership transfer. An instrumented build is a distinct
  configuration and cannot establish performance equivalence.

The available source shows the creation path but does not establish a safe
complete interception boundary or an audited SlangPy receipt lifetime/API.
In particular `getNativeHandle` exposes the `VkBuffer`, the device exposes
native handles, and deferred deletion complicates a borrowed receipt. A partial
patch that marks only the ordinary constructor as tracked would make a stronger
claim than these observations justify. Draft, deliberately unavailable interface patches and CPU-only model tests are
supplied for review under `tools/diagnostics/patches`; they do not implement or
prove any real observation. Those boundaries still require independent review.
