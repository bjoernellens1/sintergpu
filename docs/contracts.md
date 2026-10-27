# Numerical contracts

Contracts are versioned with the source. Different profiles are never silently
substituted. Imported source behavior is retained; see provenance and donor docs.

## Locus float4 ABI1

Bindings0–2 are float4 input buffers;3 is float4 output;4 is uint32 state;5 is
128-byte parameters (`float4 p[8]`). Image strides count float4 elements, not
bytes. Validate extents, calibration and accumulator budgets before recording.
The independent scalar reference validates fixture inputs.

- Planar Y/U/V carry byte values in float4.x, cosited chroma; this is not NV12.
- Resize uses nearest pixel centers. Bilinear resize is a distinct operation.
- Depth validity is finite and positive. Registration initializes state to
  positive-infinity float bits, atomically retains nearest positive depth, then
  resolves state. Output validity and optical right/down/forward coordinates are explicit.
- Quality reduction quantizes/clamps and caps pixels at2^20; reset counters.
- Voxel cells range[-512,511], with bounded probing and explicit overflow counters.
- Mask threshold is inclusive; outside-image morphology samples are false.

The full parameter table is retained in
[Locus's contract](../references/donors/locus/gpu/README.md).

## Caspar and SLAM

Three-component AoSoA chunks occupy4floats; six-column Jacobians split4+2.
Keep TSDF averaging denominators and finite-at-zero SE(3) branches unchanged.
SLAM BRIEF uses its original deterministic pattern, not OpenCV ORB descriptors.
Normal-equation reduction ordering and original numerical tolerances remain
binding. State initialization, capacities and resource lifetime are caller-owned.

Generated prefixes have retained definitions and hashes. Frames, keyframe
scheduling, rollback/acceptance and loop policy stay with consuming applications.

## Color and evidence

EliGSiR-sim's exact uint8 profiles retain exhaustive rounding guards and HIP
stream behavior. RGB→NV12 explicitly uses centered2×2 chroma. Portable profiles
with looser tolerances cannot satisfy exact-profile requests.
VKL projection uses row-major world-to-camera matrices, ties-even sensor lookup,
dtype-specific epsilon, strict comparisons and explicit byte-mask conversion.
