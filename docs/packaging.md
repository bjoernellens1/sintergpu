# Packaging for Python and C++ consumers

The canonical shader sources, contracts and version are shared by every package.

| Distribution | Contents | Consumer requirements |
| --- | --- | --- |
| `sintergpu` universal Python wheel | Slang sources, generated math, catalog, fixtures, CLI and Python API | Python3.11+; no GPU runtime imported by catalog/resource discovery |
| `sintergpu[gpu]` | Base wheel plus pinned SlangPy and NumPy dependencies | Supported platform wheel for SlangPy and a compatible GPU driver |
| CMake/source bundle | Same shader tree and C++20 headers; exported `sintergpu::kernels` target | CMake3.25+; caller's Vulkan/Daxa runtime; Python unnecessary |
| Optional backend binary wheels | Future compiled C++/HIP bindings for specific qualified operations | Explicit platform/backend compatibility; never required for source-only users |

Start with GitHub release wheels/source bundles. PyPI publication requires its
own project/trusted-publisher setup; no PyPI availability is claimed. Consumers
can pin a release wheel URL and SHA256, or a tagged source archive.

C++ headers record into caller-owned command contexts. They are header-only,
so the base Python wheel needs no C++ compiler, CPython ABI binding or Daxa build.
If a measured operation later needs a compiled binding, use a separate optional
backend package built with CMake/scikit-build-core and a narrow Python binding.
Avoid imposing platform-specific compiled dependencies on catalog/shader users.

Precompiled SPIR-V can be an additional release artifact per pinned toolchain.
It does not imply qualification on every GPU, and does not replace Slang source
for device-function composition. Source/compiler/flags hashes remain evidence.

Package installation tests must run outside the checkout, with no donor source
paths, and verify shader/fixture availability. CMake install tests relocate the
prefix and consume only its exported config and installed files.

Primary references: [Python binary extension packaging](https://packaging.python.org/en/latest/guides/packaging-binary-extensions/),
[scikit-build-core](https://scikit-build-core.readthedocs.io/en/latest/),
[CMake importing/exporting](https://cmake.org/cmake/help/latest/guide/importing-exporting/index.html).

## Optional native C++ radix library

Build the native implementation from the released source bundle with
`SINTERGPU_BUILD_RADIX=ON`. Provide either a `volk` or `volk::volk` target in the
caller superproject, or an installed `volk` CMake package. Vulkan development
headers and a selected Slang compiler are build dependencies. The caller's
`volk` target does not need to join SinterGPU's install/export set.

```cmake
# Source/superproject build, after providing volk:
set(SINTERGPU_BUILD_RADIX ON CACHE BOOL "" FORCE)
set(SINTERGPU_SLANGC "/path/to/pinned/slangc" CACHE FILEPATH "")
add_subdirectory(sintergpu)
target_link_libraries(my_app PRIVATE sintergpu::radix)
```

After building/installing the native archive, another C++ project can consume it
from a relocated installation with no Python or shader compiler at consumer build
time. It must provide its own compatible `volk` target or package and Vulkan
development headers:

```cmake
find_package(sintergpu CONFIG REQUIRED)
target_link_libraries(my_app PRIVATE sintergpu::radix)
```

The installed target exposes `vk_radix_sort.h`. The existing `VrdxSorterCreateInfo`
accepts the caller's physical device/device/pipeline cache; command functions
record into the caller's command buffer. Consumers retain buffer lifetime,
submission, synchronization and runtime capability validation. No loader/device
or GPU work is initialized by CMake discovery or linking. SinterGPU does not
bundle or install the caller's `volk` dependency.

The CMake 3.25 checks compile the actual native archive and four embedded shaders
under both pinned Slang versions, remove the original producer paths, relocate
the install and link/execute CPU-only probes for both `volk` target spellings,
either already provided or found in a separate dependency prefix. This proves
installation and static linking, not Vulkan execution, sorting correctness or
performance. Native GPU qualification and binary-wheel distribution remain pending.
