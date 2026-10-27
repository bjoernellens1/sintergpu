# VKL binary feature contracts

These experimental ports implement the frozen VKL float32 forward operations
`brief_describe` and `binary_local_match` from revision
`3384023a311651ea4260cf12ddab1567972f89f3`. Frozen Torch implementations remain
unchanged, alongside independent NumPy references in `features_numeric.py`.
Binary outputs are nondifferentiable, as in the donor; no float64 profile is
provided. SLAM's separate original BRIEF pattern and global ratio matching policy
remain separate contracts (overlap groups G15/G16).

All entries use 256 threads per group and require Vulkan `shaderInt64`, including
64-bit integer arithmetic and buffer elements. Masks and unpacked descriptors
use uint32 logical 0/1 storage. Coordinates, pair offsets and returned match
indices/distances use native signed int64. Image/projection inputs are float32.

## BRIEF

`vkl.brief.briefDescribe`: one 256-thread group per point; dispatch
`(B*P*256,1,1)`. `Params` holds uint32 `batches, channels, height, width, points,
bits, defaultPairs, usePixelsValid, useRgbValid` in that order.

| Binding | Type / scalar element count |
| --- | --- |
| rgb | float32 B×C×H×W, C=1 or 3, contiguous CHW |
| pixels | int64 B×P×2, integer x/y |
| pixelsValid | uint32 B×P when enabled, otherwise one ignored element |
| rgbValid | uint32 B×H×W when enabled, otherwise one ignored element |
| pairs | int64 D×2×2 custom offsets, otherwise one ignored element |
| descriptorOut | uint32 B×P×D unpacked comparisons |
| briefValidOut | uint32 B×P |

The center and every requested sample must lie on finite, valid RGB support.
Invalid descriptors are entirely zero. Comparison is strict first-gray < second-
gray; RGB gray uses the donor's float32 products and left-to-right sum with
weights 0.299/0.587/0.114. Custom comparison locations must be distinct. The caller
validates custom pair contents before dispatch; no GPU-data validation readback
is introduced. Bounds checks precede coordinate addition/address conversion,
including INT64_MIN/MAX offsets and invalid centers.

Default offsets use exactly VKL's affine permutation of 961×960 distinct ordered
31×31 comparisons, with D in [1,922560]. The resource-independent default-pattern
helper is in `common/brief_pattern.slang`. Custom D is limited by validated
buffer/dispatch capacities. P=0 generates no work. B,H,W must be positive.

## Local Hamming matching

`vkl.binary_match` has six entries: `packBinary`, `initializeMatch`, `matchPairs`,
`rowMinimum`, `reverseMinimum`, `finalizeMatch`. The setup-only Python
`binary_match_plan` produces their ordered, reusable lazy stage recipe. It
validates uint32 address/thread limits and the caller's device buffer/dispatch
limits without allocating any GPU resources.

Inputs are uint32 unpacked landmark/candidate descriptors B×M×D and B×N×D,
float32 projected B×M×2, int64 candidateXY B×N×2, and optional uint32 masks.
All B,M,N,D axes are nonempty. Packed words have zero unused tail bits.
Pair scratch is only B×M×min(chunk_size,N), rather than an unpacked B×M×N×D
tensor. Every candidate participates; chunk size changes memory/dispatch costs,
not ownership semantics. Candidate/landmark masks and nonfinite projections gate
pairs before distance evaluation. Candidate integer coordinates cast to float32
before spatial subtraction, matching VKL's float32 profile.

The spatial gate is inclusive squared radius; the Hamming gate is inclusive.
Row minima keep the first candidate on equal distance, including across chunks.
Mutual mode also chooses the first landmark on equal distance. Unmatched rows
return index -1, distance 0, valid 0. Outputs `matchedIndexOut`, `distanceOut`
are int64 B×M, and `matchValidOut` is uint32 B×M.

Every stage uses `Params`: uint32 `batches, landmarks, candidates, bits, words,
chunkStart, chunkCount, maxDistance, mutual, useLandmarkValid, useCandidateValid`,
then float32 `radiusSquared`. Packing and pair/final stages dispatch their linear
counts; row/reverse minima dispatch one 256-thread group per row/chunk candidate.
The plan specifies each exact binding, dtype, capacity and thread count. Reverse
scratch is B×N only in mutual mode; otherwise a one-element ignored binding.

`initializeMatch` precedes all chunks. Each chunk writes pair scratch before row
and reverse reductions read it; every reverse index is written before finalization.
The caller retains all buffers through completion. `record_feature_plan` records
prepared kernels on the caller's encoder and inserts inter-stage barriers; it
allocates, compiles, submits, waits and reads back nothing. The caller establishes
input visibility and final consumer visibility/completion. No overlapping writable
resources are permitted. ShaderInt64 and all descriptor/offset alignment limits
must be checked by the binding adapter in addition to the setup capacities.

## Float32 numerical preservation and qualification

The new feature paths use resource-independent integer binary32 helpers for
round-to-nearest/even operations. They preserve subnormals and signed zero even
when native GPU FP32 arithmetic flushes denormals. NaN arithmetic results are
canonicalized; comparisons with NaN return false. This preserves grayscale
comparison and squared-radius decisions without relaxing binary output equality.
These helpers are original implementations, not imports of unresolved RTG code.

This exact baseline may be slower than native floating-point variants. No
optimized/default selection or whole-operation hardware qualification follows
from its presence. Independent source/performance clearance precedes GPU tests;
qualification records must retain compiler/device/source identities, raw timing
samples and workload scope. All entries remain experimental. Full pipeline,
individual stages and transfers must be measured separately. Tracking trajectories,
consumer adoption and cross-vendor execution remain separate gates.
