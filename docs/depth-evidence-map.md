# Experimental depth-evidence map stage

`kernels/experimental/depth_evidence_map.slang` is an importable, resource-free
per-pixel building block for the VKL depth-evidence profile. Its source donor is
`src/vkl/kernels/reference/depth_evidence.py` at VKL commit
`3384023a311651ea4260cf12ddab1567972f89f3`; the frozen Apache-2.0 reference is
`references/donors/vkl/src/vkl/kernels/reference/depth_evidence.py`.
Frozen source SHA-256: `5976e426e33d1e1430cc5c233bd351fe7e226f2289c1cd43966a34392957fe18`.
It emits the eight per-pixel grid-reduction fields plus pair count and squared
difference sum. It makes no allocation, dispatch, synchronization or reduction.
It is not a complete depth-evidence operation and is not catalogued or selected.

Inputs are one float32 depth and validity per pixel plus the right and bottom
neighbor, and confidence in [0,1]. Use invalid neighbors at the image edge;
when confidence is absent pass 1.0. Bounds must satisfy 0 <= near < far. Limit
absolute finite depth to 1e10 so squared differences and sums per pixel are
finite. Valid depth means caller validity and finite depth, including zero or
negative values. Edge tests are strict `>`; near/far tests are strict `<`/`>`;
an edge is assigned to its top/left pixel, and stable requires positive
confidence. Batch and grid assignment, counts, support normalization, global
pair normalization by `4 * max(pairCount, 1)`, and reduction order are caller
responsibilities. Accumulated sums may still overflow for sufficiently large
images; callers must bound them. Neighbor addressing and output non-aliasing
are also caller responsibilities.

CPU tests compare a separate CPU model of the map semantics with the frozen
Torch donor; they do not execute or prove parity of the shader. Compiler tests
import the public module with both pinned Slang compilers. Remaining gates include:
GPU differential testing, output layout/dispatch ABI, reduction-order tolerance,
gradients, performance and integration remain unqualified. No no-regression or
consumer adoption claim follows from this experiment.

## Experimental dispatch and reduction

`kernels/experimental/depth_evidence.slang` imports the helper and exposes
`depthEvidenceGlobal` and `depthEvidenceGrid` as separate 64-thread dispatches.
Each active global invocation loops over one batch; each grid invocation loops
over one bin. This deliberately simple serial reduction is a correctness
candidate, not a performance choice. The global output is eight float32 values
per batch in order: valid fraction, hole fraction, near fraction, far fraction,
local variance, edge fraction, confidence mean, stable fraction. Grid output is
`[batch, gridHeight, gridWidth, 8]` row-major with support/area, near, far,
depth, squared depth, edge, confidence, stable. Inputs are contiguous flattened
`[batch, 1, height, width]` float32 depth, uint32 validity and float32 confidence.
Always bind confidence, even if `useConfidence` is zero; then values are ignored.
Grid dimensions are clamped to the image dimensions, matching the donor.

Call `checked_depth_evidence_launch` in `sintergpu.depth_evidence_contract`
first, and populate shader parameters from the returned dimensions and canonical
float32 `near`/`far` values. Dispatch its
reported groups for each entry and enforce device binding byte-range and
alignment limits. This checks u32 arithmetic, float count exactness (including
pairs), dispatch group limits, nonzero dimensions, and finite near/far bounds.
It does not inspect buffer contents: caller must ensure finite confidence in
[0,1], finite depths limited to 1e10 absolute, enough correctly sized buffers,
and no aliasing between outputs or any inputs. Both output buffers need bindings;
each entry uses only its own output. Caller owns all allocation, barriers,
submission and synchronization; results may be read only after completion.

CPU tests compare a separate scalar model with the donor, and compiler tests
reflect the dispatch entry points; neither executes or proves the shader. GPU
numerical parity, Torch gradients, reduction-order tolerance, performance and
consumer integration remain pending. CPU summation order is not the Torch or
GPU reduction order; no exact floating-point equality is promised.
