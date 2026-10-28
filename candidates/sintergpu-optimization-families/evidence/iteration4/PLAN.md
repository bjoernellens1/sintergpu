# Scheduling-preserving point-plane profile

The previous struct-return composition preserved operations and exact outputs but
changed load/store/arithmetic phases. This iteration provides a shared residual,
rotation and scalar-export core while preserving each generated adapter's phases.
The original helper and all previous rejected measurements remain recoverable.

`compile/instruction-equivalence.json` compares two profiles under Slang 2026.12
and 2026.19, with precise FP and O3. It removes debug names/source, consistently
renames IDs by definition order, and sorts semantically unordered decorations.
All executable/type/constant instruction ordering, operands, decorations and
resource layout remain compared. All four comparisons pass. Equality does not
substitute for measured performance or numerical validation.

The separate native Vulkan profile uses the 2026.19 artifacts and one retained,
caller-owned descriptor/buffer set for both variants. Predeclared counts are
1,63,1023,1024,1025,4099,65539, in both AoSoA and scalar profiles. Every configuration
has three fresh native processes, ten warmups per variant per boundary and 1,000
ABBA blocks per boundary, retaining every individual invocation. CPU affinity is
31 on Radeon and 19 on the canonical RTX4070Ti host. The original median/p95
bootstrap bounds and credible-slowdown rejection remain unchanged; each of GPU,
resident completion and complete transfer boundaries must pass independently.
No averaging across configurations or runs can cancel a failure.

The native adapter requires cached host staging, reports memory allocation flags,
and includes all uploads and complete output downloads/host copies in transfer
timing. It independently restores poison before each correctness invocation and
checks byte differential and padding before/after sampling. The Python suite then
checks an independent float64 oracle against both variants and phases. Both smoke
runs passed these checks, but their ten blocks are not qualification evidence.

The local native suite used commit 45d82b5; the archive launcher additionally accepts
an explicit full revision because the remote exported source has no .git directory.
Each plan records exact source, binary, SPIR-V, fixture and output SHA256 identities.
Prior Python profiles are separate and retain their previous decisions. No default,
catalog, installed API, consumer or release change is authorized by this evidence.
