# SinterGPU

Composable GPU kernels. Measured performance.

A Slang-first library consolidating reusable image, depth, geometry, tracking,
reconstruction and optimization operations from existing projects. Kernels,
independent references, provenance and device-specific evidence travel together.

```sh
python -m pip install .
sinter list
sinter info image.resize_nearest
sinter test --all --backend reference
```

GPU execution requires the optional `gpu` extra. GPU skips and compilation alone
never count as hardware qualification. See [status](docs/status.md),
[contracts](docs/contracts.md), [integration](docs/integration.md),
[provenance](provenance/README.md), and [changelog](CHANGELOG.md).
The [kernel overlap audit](docs/kernel-overlap.md) compares duplicated tasks,
contract differences, canonical owners and required consolidation checks.
See [packaging](docs/packaging.md) for Python wheels and C++/CMake integration.

Core adapters borrow application devices and command contexts. Standalone test
and benchmark tools explicitly own their test resources. Production callers own
allocation, submission, synchronization and lifetimes.
