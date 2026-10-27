# Validation scope and remaining work

Development continues in `bjoernellens1/slang-rgbd-slam`; cross-vendor comparisons
are qualification work, not a prerequisite for creating or developing that repository.

`uv run python scripts/validate.py` runs portable tests, actual Caspar lowering,
standalone SPIR-V compilation and real Vulkan numerical tests. It returns 0 when
those local checks pass and 2 when a required check fails or its dependency is missing.

Continue RGB-D projective ICP, depth/normal pyramids, iterative pose updates,
keyframes, local mapping, TUM loading and trajectory export. Track NVIDIA
Vulkan/Caspar CUDA parity and Intel coverage independently. Maintain explicit
backend, hardware, numerical and timing evidence as implementations develop.
