# RTX 4070 Ti trusted-host engineering attempt

This is a deliberately enabled, separate diagnostic. Do not run on a GPU until
its final source, build and lock path are independently reviewed. It does not
change the blocked SlangPy path, release qualification, or performance policy.
A successful result establishes only fixed-workload direct Vulkan engineering
parity on the trusted workstation; it does not authenticate the loaded ICD or
driver, prove physical backing independence, qualify SlangPy, establish model
accuracy, or report performance.

The attempt selects the previously observed RTX 4070 Ti UUID
`96d2dba0-95fd-5c00-e849-2e0610f24e06`, PCI vendor/device `10de:2782`, and raw
Vulkan driver version `2497233024` (595.99.02). It rejects other identities and
requires same-instance properties and limits. The version is historical evidence,
not a claim that the device currently has that driver; any mismatch fails closed.
The user-specified single ICD JSON is hashed as configuration only. It is not
proof of the loaded ICD or of an absence of implicit Vulkan layers.

The runner checks committed shader, fixture and local helper bytes, compiles helper
source directly from Git-verified bytes, rebuilds the native binary twice using
the pinned local toolchain, checks exact SPIR-V hashes and targeted ABI, and holds
the canonical exclusive GPU lock across both native processes and full output
validation. Its actual native module receives those exact shader bytes. Seven
separate Vulkan allocations are bound; their handles do not prove physical
independence. It compares the frozen Caspar donor against the packaged candidate
bitwise, against an independent float64 oracle, and checks inputs and output
tails. A receipt is written exclusively after teardown; error receipts describe
failure without qualification. Compiler/linker dependencies and loaded runtime
mappings are not authenticated and remain trusted-host assumptions.

After independent review and with all files committed, reproduce the native build
with `verified_build.py --commit COMMIT --output BINARY --receipt BUILD_RECEIPT`.
Set `VK_DRIVER_FILES` to exactly one trusted absolute ICD JSON path; do not set
`VK_ICD_FILENAMES`, `VK_ADD_DRIVER_FILES`, `VK_LAYER_PATH`, `VK_ADD_LAYER_PATH`,
`VK_INSTANCE_LAYERS`, or `LD_PRELOAD`. Run `runner.py --execute --commit COMMIT
--uuid 96d2dba095fd5c00e8492e0610f24e06 --driver 2497233024 --binary BINARY
--binary-sha b04a1e4d73dd2b4db1820a6d6816ecb9fc03c32f8a47f96b0bcbaa0dbbe431ad
--slangc SLANGC --icd ICD_JSON
--icd-sha ICD_JSON_SHA256 --output NEW_RECEIPT`. The execution host must have
all pinned tools and the committed source; paths above are local to the current
build environment and must be checked on the target host. The runner requires
an explicit execute flag and refuses to overwrite receipts.

CPU-only tests directly link fake Vulkan functions and never load a real Vulkan
library in the fake executable. The fake dispatch copies independent CPU oracle
output; it exercises setup, guards and cleanup, not shader computation. Run
`/home/bjoern/miniforge3/bin/python3.13 -m pytest -q tools/diagnostics/direct_vulkan/engineering_attempt`.
No GPU execution has been performed in this attempt.

Target-specific CPU preparation (no GPU execution): on `cps-wkstn-nvidia4070`,
Ubuntu GCC 13.3.0 `/usr/bin/c++` SHA-256
`1353e9bdd29a7295c7226bf6c63abccce056d8cac31f112e5cdbecc3f28c2769`
produced the pinned native binary in three local builds. A user-owned isolated
copy of Slang 2026.19 at
`/home/bjoern/.cache/sintergpu-qualification/direct-point-plane-4070-PdRxNvIk/slang-2026.19`
produced the two pinned shader hashes. Its complete regular-file hash manifest
matched the pinned local tree using `LC_ALL=C` (manifest digest
`794d004679ce9e5773143100959e149277acf221d8467bc7a17d1158a6302669`).
This verifies copied regular files at preparation time, not all symlinks, loaded
libraries or future runtime state. Target CPU tests passed 13 cases using that
compiler and existing NumPy 2.5.4 venv. The isolated source checkout and
compiler paths are preparation artifacts and may need recreation.
