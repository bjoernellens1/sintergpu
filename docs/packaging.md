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
