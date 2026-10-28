# Experimental VKL sparse depth validation

`kernels/experimental/sparse_depth_validate.slang` independently ports the
float32 forward profile of VKL `src/vkl/kernels/reference/sparse_geometry.py`
`sparse_depth_validate`, pinned at `3384023a311651ea4260cf12ddab1567972f89f3`
(owner-authorized Apache-2.0). The frozen donor source SHA-256 is
`c56224f9313c21c0fa4315ae09f14a1e43e61417a3cd4819a5ac096395a56fc9`.
This is unqualified, not catalogued or selected as a default.

Input depth is contiguous float32 `[B,1,H,W]`, points are contiguous int32
`[B,P,2]` in (x,y) order, and pixel-valid flags are uint32 `[B,P]`; nonzero is
true. Unlike the donor's int64 pixels, coordinates must fit int32. Output arrays
are contiguous `[B,P]`: float32 depth, uint32 validity, float32 normalized
median absolute deviation, float32 normalized max-minus-min range, uint32
neighbor count, and float32 confidence. Donor neighbor count is int64; values
are always in [0,9]. An invalid supplied pixel still gets neighborhood count
and diagnostic metrics when its coordinates are in bounds. Out-of-bounds points
have count and diagnostics zero. The center is never clamped into the image.

In-image positive finite depths form a 3x3 neighborhood. The lower median is
used for both depths and absolute deviations. Normalize by max(median,float32
epsilon). Validity requires an enabled in-bounds center with positive finite
depth, at least the requested number of neighbors, finite metrics and both
metrics <= threshold. Confidence is `(count/9)*(1-clamp(max(metrics),0,1))`
for valid points and zero otherwise. Positive infinite diagnostics become
float32 maximum on output, following donor `nan_to_num`; their corresponding
point is invalid. Empty neighborhoods output zero diagnostics. CPU and shader
float rounding and denormal handling can differ at strict decision boundaries.

Call `checked_sparse_depth_validate_launch` and use its returned canonical
float32 threshold and group count. The caller must supply in-range int32 points,
correctly sized and non-overlapping bindings, and enforce device buffer range,
alignment, dispatch and synchronization requirements. Inputs and outputs are
caller-owned; no hidden allocation or submission occurs.

CPU tests compare an independent scalar model against the frozen Torch donor;
they do not execute the shader. Both pinned compiler reflection checks are
compile-only. GPU parity, gradients, performance and consumer integration are
pending; no performance or adoption claim is made.
