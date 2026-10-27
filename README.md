# SinterGPU

Composable GPU kernels. Measured performance.

A Slang-first library consolidating reusable image, depth, geometry, tracking,
reconstruction and optimization operations from existing projects. Kernels,
independent references, provenance and device-specific evidence travel together.

[Experimental 0.1.0 release](https://github.com/bjoernellens1/sintergpu/releases/tag/v0.1.0)
includes a Python wheel, source packages, checksums, attestations and scoped
qualification metadata. Full consolidation and cross-vendor qualification remain
in progress.

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
See [packaging](docs/packaging.md) for Python wheels and C++/CMake integration,
and [consumer migrations](docs/consumer-migrations.md) for delivery and adoption evidence.
The [VKL source inventory](docs/vkl-source-inventory.md) records all 69 public
reference candidates; [geometry/depth contracts](docs/vkl-geometry-depth.md) describe
the new forward profiles and their explicit numerical and qualification limits.
[Temporal evidence selection](docs/vkl-temporal-best.md) has its own forward contract.

Core adapters borrow application devices and command contexts. Standalone test
and benchmark tools explicitly own their test resources. Production callers own
allocation, submission, synchronization and lifetimes.
