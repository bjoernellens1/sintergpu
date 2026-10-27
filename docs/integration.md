# Integration

Python consumers install a pinned release and use `sintergpu.kernel_path(id)`.
Device functions may be included directly from the installed kernel tree.

```python
from sintergpu.adapters.slangpy import SlangPyAdapter
adapter = SlangPyAdapter(existing_device, allow_experimental=True)
adapter.prepare('geometry.backproject')  # setup only: compile/reflection/cache
adapter.record('geometry.backproject', thread_count=(width, height, 1),
               bindings=existing_bindings, command_encoder=existing_encoder)
```

The application validates resource extents/state/capabilities and owns all
barriers, submission and lifetime. Source identities are available on the adapter.
An operation without qualification requires explicit experimental opt-in.

CMake: `find_package(sintergpu CONFIG REQUIRED)` then link
`sintergpu::kernels`. `SINTERGPU_KERNEL_DIR` identifies installed shader modules.
No Daxa, SlangPy or Vulkan runtime is linked by the source-only target.

`<sintergpu/vulkan.hpp>` records caller-owned pipeline/descriptors into an existing
VkCommandBuffer. `<sintergpu/daxa.hpp>` records an existing compute pipeline using
TaskInterface.recorder; the caller declares task attachment reads/writes and
supplies reflected push constants. Workgroup counts differ from invocation counts.

The headers do not infer barriers or create pipelines/devices. Daxa consumers link
their existing `daxa::daxa` target. Compatibility pin and required upstream patches
are retained from StereoForge; full adapter GPU qualification is separate from
header compilation.
