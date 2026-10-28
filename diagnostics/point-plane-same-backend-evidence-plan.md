# Point-plane same-backend parity: remaining evidence path

Status: design investigation only. The point-plane GPU production gate remains
blocked; no GPU execution, performance or qualification is claimed.

The current isolated RHI schema-2 patch obtains UUID, physical limits and the
`vkGetInstanceProcAddr` / `vkGetPhysicalDeviceProperties` addresses from the
same backend's selected physical device. In pinned RHI `src/vulkan/vk-api.cpp`,
`VulkanModule::init` opens `libvulkan.so.1`, `getFunction` uses `dlsym`, and
`initInstanceProcs` resolves instance functions through that pointer. These
addresses can therefore be loader trampolines; `dladdr` or mapped-library names
alone do not identify the executing ICD. Selecting one ICD JSON is a loader
configuration observation, not proof of which code ultimately executed.

The allocation patch records each successful ordinary RHI buffer bind, the
VkBuffer, its VkDeviceMemory, allocation requirement size, offset zero and an
owner generation. Wrapped/native and shared allocations are not admitted. This
supports a *controlled-process backend observation* while owners are held, but
cannot detect foreign native operations, prove the full Vulkan binding history,
or independently establish physical backing. Distinct native buffer handles
alone are insufficient. The C++ CPU probe uses synthetic objects; it did not
exercise a Vulkan allocation, export, device loss or concurrency.

The current point-plane runner compares git-object shader bytes and a temporary
read-only core copy before and after compilation. Its native runtime, imported
Python helpers and compiled shader bytes are not authenticated. A read-only
file mode is not an immutable snapshot. On this host a self-mapped temporary
file's `/proc/self/map_files/<range>` stat returned `EPERM`; `/proc/self/maps`
also reports a backing device different from `stat` on overlayfs. An
`unshare -Urnm true` probe succeeded, but this does not establish GPU access,
trusted runtime closure or unforgeable loaded-page identity in that namespace.

A concrete, still-unimplemented path is:

1. Build the paired schema-2 runtime in a fresh, source-verified environment;
   retain full compiler/linker/dependency identities and binary hashes. Start a
   fresh process in a private mount/user namespace, with a read-only verified
   source and native-runtime closure, no pre-imported local modules or bytecode,
   no access to the mutable original checkout, controlled layer/loader variables,
   and dropped capabilities. Adapt the existing CPU snapshot prototype only
   after its native extensions, dynamic linker and required GPU-device access
   are constrained and independently tested. Prove tampered/late/shadowed
   modules fail before execution; do not infer immutable loaded pages from a
   mutable pathname or a caller-provided digest.
2. Investigate explicit exclusive driver selection via
   `VK_LUNARG_direct_driver_loading`: the pinned Vulkan header declares an
   exclusive mode and a driver `pfnGetInstanceProcAddr`. This is a proposal,
   not evidence that the target loader or driver supports it. Establish support,
   exact loader and ICD binary closure, instance chain, no implicit layers,
   driver entry-point provenance and selected live device UUID on the same RHI
   before exposing a reviewed receipt. Test fail-closed on unsupported,
   ambiguous or unexpected providers. If the target cannot support or verify
   this route, retain the block rather than substituting a vendor name or JSON.
3. Extend the controlled diagnostic with actual backend-owned buffer leases:
   observe successful allocate/bind and teardown, retain owners while using
   them, verify live generations/ranges and same device, forbid external/wrapped,
   shared, sparse and untracked paths, and prohibit native foreign mutation.
   Exercise ordinary, close, failure and supported shared paths on the actual
   runtime under the canonical lock. Report this as a controlled-process
   allocation observation unless a stronger backing guarantee is established.
4. Bind the shader bytes and compilation to execution. The pinned RHI
   `src/vulkan/vk-shader-program.cpp::ShaderProgramImpl::createShaderModule`
   passes `kernelCode` directly to `vkCreateShaderModule`; an independently
   reviewed, opt-in same-backend receipt could hash those exact bytes and entry
   point alongside compiler configuration, source/import closure and live device
   generation. Merely hashing a separately compiled SPIR-V file does not prove
   those bytes were passed to Vulkan. Capture must not alter GPU submission or
   introduce hidden synchronization.
5. Only after independent review, use the canonical lock before device creation
   through teardown, verify actual same-backend limits, source/runtime/driver
   and allocation evidence before dispatch, and run full fixture input/guard,
   paired output and independent-oracle checks. Keep correctness, performance,
   source provenance and both-vendor qualification as distinct outcomes.

The direct-driver route, complete read-only runtime isolation, same-backend
shader-module capture and real allocation lifecycle have not been implemented
or executed. None of these proposals authorizes bypassing current guards.
