# Third-party sources

## VkSplat

Repository: `harry7557558/vksplat`

License: Apache-2.0.

Pinned read-only upstream implementation source for the differentiable Vulkan-compute path. `xsplat-vk` builds the projection, tile/key, scan, radix-sort, raster forward/backward, and related math required by the backend. Python bindings, trainer/application policy, dataset code, and viewer/server code are not imported.

The build applies narrow portability preprocessing outside the submodule: the radix GLSL include is inlined and its workgroup reduced from 512 to 256 threads. Projection/key generation uses the upstream int64-emulation path.

## NVIDIA vk_gaussian_splatting

Repository: `nvpro-samples/vk_gaussian_splatting`

License: Apache-2.0.

Reference for the later graphics/display path, storage/layout ideas, and Vulkan/Slang practices. The full nvpro application stack is not bundled.

## Spirula Studio

Repository: `harry7557558/spirula-studio`

License: GPLv3.

Architecture and portability lessons only. GPL implementation code is not copied into this Apache-2.0 project.
