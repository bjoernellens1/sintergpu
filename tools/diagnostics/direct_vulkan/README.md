# Direct Vulkan point-plane engineering-parity scaffold

This is a separate trusted-workstation diagnostic, not SlangPy qualification,
provider/ICD authentication, proof of physical backing independence, or a
performance comparison. Both Python and native production entry points are
hard-blocked before Vulkan. The Python private test path and the native
`SINTERGPU_CPU_TEST_ONLY` compile option exist only for synthetic CPU tests;
never use either for hardware. Existing production qualification gates remain.

The proposed fixed workload uses the checked-in frozen Caspar AoSoA donor
(`computeMain`, 1024 threads) and packaged candidate (`main`, 256 threads),
the fixed 1025-point fixture, and exact pinned Slang 2026.19 compiler/SPIR-V
file hashes. The native code selects a physical device by same-instance UUID,
checks raw driver version and dispatch/descriptor limits, verifies exact SPIR-V
and fixture bytes plus all indices, allocates and binds one host-coherent Vulkan
memory object per buffer, submits explicit host/compute barriers, reads every
buffer and cleans up. The parent verifies its exclusive canonical GPU lock across both
children, validation and native teardown. Success is recorded after teardown
and lock close; failures produce a write-once failure receipt. Paired bytes, independent float64 oracle, input bytes
and all output tails are checked. Same-source snapshot and compiler-binary file
hashes do not prove loaded native dependency identity; raw driver and device
information rely on the trusted workstation.

The separate fake Vulkan test simulates allocations, descriptor bindings,
barriers, queue handling, output readback and destruction; its fake dispatch
copies a CPU-generated oracle and does not calculate shader results. CPU tests
also check invalid identity and fixture arguments, lock ownership, output guards,
write-once behavior and both production guards. Run:

```sh
/home/bjoern/miniforge3/bin/python3.13 -m pytest -q tools/diagnostics/direct_vulkan/test_runner.py tools/diagnostics/direct_vulkan/test_native_preflight.py tools/diagnostics/direct_vulkan/test_spirv_contract.py tools/diagnostics/direct_vulkan/test_verified_build.py
```

The preflight test compiles the donor with the locally pinned Slang compiler and
skips if that compiler is unavailable; a skip is not evidence. Missing before
hardware: complete native failure/teardown and lock coverage, independently
verified source/build and loaded runtime closure, independently reviewed failure
receipts and SPIR-V ABI checks, and review of the unblocked production path. The existing hard blocks must stay in
place until these are resolved. No GPU test has been run.

The scaffold pins its native source SHA-256
`24c32e16b2d560514925d70ae6d61ced4a051488ff544a32a54a6f8a7366be32`
and the **still blocked** local executable SHA-256
`c623778a0db171127b20e21bbd3ac63bbcfcae8d79691b60a4857c1a7a45f06b`.
Two local builds matched with GCC 16.2.1 using:

```sh
c++ -std=c++17 -O0 -g0 -Wall -Wextra -Werror -Wno-missing-field-initializers tools/diagnostics/direct_vulkan/point_plane.cpp -o OUTPUT -lvulkan -lcrypto
```

This local repeat is not independent build authentication or complete toolchain
closure. Removing either production guard changes the relevant source/binary
and requires new reviewed pins. The CPU-test executable links only the fake
Vulkan object, without `libvulkan`; a fake-only marker is required at link time.

`spirv_contract.py` inspects the actual compiled SPIR-V for entry point,
workgroup, descriptor sets and bindings, scalar storage types and stride, and
uniform field offsets; the uniform buffer allocates 32 bytes. Tests compile both
modules and reject targeted layout mutations. This targeted parser is not a
general SPIR-V validator; the runner also checks exact pinned shader hashes.
`verified_build.py` rebuilds the pinned native source twice with the pinned
local compiler and compares exact blocked binary bytes. The runner requires the
source from the specified Git commit and compares this rebuild with the
requested executable. Toolchain dependencies, dynamic loading and the Python
execution closure are not authenticated; both production guards remain.
Native fake tests include failed submission and failed device-wait teardown;
those cases destroy all fake allocations without publishing output.
