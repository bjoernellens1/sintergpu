# CPU fixture for specialization-cache miss: investigation

This records the original design investigation, not GPU qualification. The
seeded-cache probe tests lookup only. A later compiler-backed CPU fixture tests
the success-path misses and hit described in `DISPATCH-ASSOCIATION.md`. A meaningful miss fixture must use a real
Slang component with a specialization parameter, supply the corresponding type
and component ID, and start with an empty `m_shaderCache`. The actual path is:

1. `Device::getConcretePipeline` misses its cache and calls
   `Device::specializeProgram`, which invokes the component's `specialize`.
2. Vulkan `createShaderProgram` initializes the specialized component and a
   reflection-derived `RootShaderObjectLayoutImpl`.
3. `compileShaders` obtains generated code and `createShaderModule` supplies its
   exact bytes to Vulkan. Concrete pipeline creation then inserts the result in
   the specialization cache.
4. A second request with the same arguments should return the same cached
   concrete pipeline without another specialization or pipeline creation; a
   different type must miss and produce its own independently checked code.

To execute this on CPU without a GPU or loader, the fixture needs a real Slang
compiler/session and controlled fake Vulkan function pointers for every reached
layout, descriptor, shader-module and pipeline API, plus initialized device
capabilities. Assert actual component specialization, cache population, later hit,
concrete pipeline identity, observed bytes, and failure propagation; retain a
real compiler diagnostic and test an invalid specialization. A synthetic
pre-populated shader program would not prove the miss path. Exact APIs reached
by reflection depend on the chosen shader and must be traced before enabling
callbacks. Independent review is required; the production GPU gate stays closed.

The following SHA-256 values identify files inspected in the local isolated lab
at investigation time, not a complete or immutable build closure:

| Lab input | SHA-256 |
| --- | --- |
| `rhi-source/src/device.cpp` | `081c8185aa574c4675b9f2fe95f57922839854dcea55f33a3328d5c80b4b1420` |
| `rhi-source/src/shader.cpp` | `049359c34927756b817f8aec6a649a64ba807549cdbb0ebe36ef290aa2100bfc` |
| `rhi-source/src/vulkan/vk-device.cpp` | `52a57fc7b08bba8f06402a987da3e535b714785a22018e8a42dc17172baf222b` |
| `rhi-source/src/vulkan/vk-shader-program.cpp` | `bf854f78cc08305ba345a100c3b756ee2522e06c2791a6fb7107f667d2329bd4` |
| `rhi-source/src/vulkan/vk-shader-object-layout.cpp` | `140441103520ea9bd8e0e78c98d44c52e544c6c380615422360c87dedfaee8a4` |
| `rhi-source/src/vulkan/vk-pipeline.cpp` | `1d5ef0607c3d133211f117a0c84a7283f8116aab061601a4c0829c686a1509f1` |
| Local linked `libslang-compiler.so.0.2026.12` | `195f1fbbf7342517aed3064657c8cfaaae5791065158189610253dce070c4678` |

The compiler hash is of the file named by the current local link command, not
proof of the dynamically loaded library or of the compiler's transitive native
dependencies. Before a future run, bind and verify the complete test, patched
RHI, headers, shader inputs, binary, loaded compiler and runtime closure; capture
actual mappings. The later compiler-backed CPU fixture does not establish that
complete closure and does not exercise error paths or GPU execution.
