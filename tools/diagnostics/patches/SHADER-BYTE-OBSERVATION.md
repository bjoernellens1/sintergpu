# Experimental same-backend shader-byte observation

These source patches are an unqualified diagnostic candidate. Apply them only
on top of the isolated creator, allocation, and device-limit receipt patch
layers; keep the upstream Slang RHI/SlangPy `Apache-2.0 WITH LLVM-exception`
licenses and existing EliGSiR/SinterGPU modification notices. The named input
and output source hashes are in `test_rhi_shader_observation_patch.py`. Local
source checks skip when the audit copies are absent and do not constitute CI
or build provenance.

The RHI patch copies the compiler-produced module bytes into an aligned,
program-owned array immediately before `vkCreateShaderModule`, passes that
array as `pCode`, and exposes it only after shader compilation and all module
creations succeed. The optional COM interface retains the program; the SlangPy
binding makes Python-owned byte copies while that interface is held. Queries
must be serialized against compilation, hot reload, close and destruction.
The source entry name and the Vulkan stage name (`main`) are reported
separately. These are observations of module creation in this instrumented
runtime, not proof that a specific subsequent pipeline or dispatch used those
modules: specialization, pipeline caching and hot reload still need a
same-backend binding and independent review. Returned historical Python bytes
are not an owner lease.

The additional copy allocates and changes the `pCode` host pointer, and therefore
this build is a distinct configuration with no performance-equivalence claim.
It introduces no deliberate GPU submission, transfer or wait. Provider/ICD,
complete runtime/compiler identity, physical buffer backing, source closure and
loaded mapped pages remain separate, unverified issues. The hard point-plane GPU
gate is unchanged. No GPU execution has been performed.

The local source patches have SHA-256
`2fdab79fe2065e9767fa59cd534d58b6861fbf19cd79e2f6b08031312a046d18` (RHI) and
`780a07f569a1c246fc3e8635568a0a809acf382feb42e550923e68a2bbbb3e4d` (SlangPy).
A CPU-only C++ probe exercises synthetic module state and COM lifetime, not
`vkCreateShaderModule` or an actual driver. Full runtime and GPU behavior need
independent review and separately controlled device tests.

The isolated Debug paired build/link passed locally with GCC 16.2.1, Python
3.13.12 and `SGL_WARNINGS_AS_ERRORS=OFF`; upstream Python/glibc macro warnings
remain. The C++ CPU probe passed synthetic not-yet-compiled, completed, failed,
out-of-range, retained-interface and synthetic conversion-exception COM-release checks without initializing Vulkan. The exception probe exercises the actual COM smart pointer, not a forced nanobind allocation failure. A
SlangPy CPU-device smoke confirmed the binding is unavailable before and after
close. The executable source checks in the patch test are local-only. The local
`libsgl.so` SHA-256 is
`caf91f01957630db6602899eada3d0fec0038280313505ea55674a4d601807bb` and
extension SHA-256 is
`728951fc5636889d6dee613b651f0ee6d66eb973ecc08b929370e054c712bfc1`; these
are not evidence about another build or machine.
