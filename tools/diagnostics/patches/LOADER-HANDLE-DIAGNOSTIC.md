# CPU diagnostic: RHI loader object and symbol owner

On Linux, RHI `VulkanModule::init` calls `dlopen("libvulkan.so.1", RTLD_NOW)`;
`getFunction` calls `dlsym` on that exact handle. The CPU-only probe subclasses
that real RHI module to inspect its borrowed handle while it is live, using
glibc `dlinfo(RTLD_DI_LINKMAP)`, and uses `dladdr` on the resolved synthetic
`vkGetInstanceProcAddr` address. No Vulkan function is called. This diagnostic
is not exposed by the production device and does not alter the hard GPU gate.

The local pytest builds two synthetic shared-object configurations: the symbol
is defined by the opened loader object in one, and by its linked dependency in
the other. It checks exact canonical expected paths for both cases and rejects
mismatched expectations. This demonstrates why the opened loader object alone
cannot identify the symbol's owning object. It does not identify the underlying
ICD, establish mapped-page bytes or complete dependency closure, authenticate
provider identity, or show real driver behavior. The isolated objects and
loader environment are test fixtures, not production provenance.

Build the local diagnostic target in the prepared Debug lab with
`cmake --build /dev/shm/sintergpu-allocation-correlation-lab/build --target rhi_loader_handle_cpu_probe`,
then run `/home/bjoern/miniforge3/bin/python -m pytest -q tools/diagnostics/patches/test_rhi_loader_handle_cpu_probe.py`.
The test skips if the local build or C compiler is unavailable. Locally it
passed once with both ICD environment variables set to nonexistent paths inside
child processes; assertions are mandatory at compile time. Source SHA-256:
`9b976a7faf727fab2a6dea3e4f814c4c26779ad834d777f87a167eb6b52b93b5`;
local binary SHA-256:
`157196340467d221cb1a3ead59410895e15ba8b9422006477e77f9dc3f9f1270`.
These identify this local build, not a production runtime or different host.
