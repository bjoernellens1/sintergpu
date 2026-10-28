# Experimental VKL dense depth regularization

`kernels/experimental/dense_depth_regularize.slang` is an independent float32
forward dispatch derived from VKL `src/vkl/kernels/reference/dense_depth_regularize.py`
at commit `3384023a311651ea4260cf12ddab1567972f89f3`, original owner code under
Apache-2.0. The frozen reference lives under `references/donors/vkl/`; its
SHA-256 is `586e2b2e8dfd2a0991f836fb640a7c976fefca14c39dcb6abbf64d44f52d3351`.
This is an unqualified experiment, catalogued as `vkl.dense_depth_regularize.regularizeDenseDepth`; it is not selected as a default.

Depth, uint32 validity and confidence use contiguous `[B,1,H,W]` layout; RGB
uses contiguous planar `[B,C,H,W]` with C=1 or 3. Outputs `depthOut`, `validOut`,
`confidenceOut`, `directValidOut`, `supportOut` are one float32/uint32 value per
pixel. Support is a uint32 value in [0,9], whereas the donor returns int64.
All buffers including outputs must be correctly sized, non-overlapping and
caller-owned. Call `checked_dense_depth_regularize_launch` and use its returned
canonical thresholds and group count; enforce device binding size, alignment
and dispatch limits separately. Caller validates that confidence is finite and
in [0,1], nonzero validity means true, and input buffers have these layouts. No
allocation, recording, barrier, submission or synchronization is hidden.

The zero-padded 3x3 neighborhood includes the center and excludes off-image
pixels. Eligible neighbors have nonzero validity, positive finite depth, and
luminance difference <= the threshold. Three-channel gray is `.299 R + .587 G +
.114 B`; gray nonfinite comparisons reject neighbors. Lower median is selected
for an even support count. Output validity depends on support and median, so an
invalid center may receive a valid output. Direct validity additionally requires
valid positive center depth within the relative-deviation threshold of the
median; float32 epsilon is used in that threshold. Confidence is the eligible
neighbors' mean confidence times support/9. These semantics follow the frozen
donor, subject to float32 arithmetic and reduction order.

CPU tests compare a separate scalar model to the frozen Torch donor; compiler
and reflection tests verify both pinned Slang compilers. They do not execute
this shader and do not establish GPU parity, boundary arithmetic, autograd,
performance, or application accuracy. Device-specific testing and controlled
no-regression measurements are required before adoption.
