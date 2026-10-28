# Standalone xsplat math timing methodology

The benchmark covers three existing experimental dispatch entries independently:
standalone Adam (12 fixtures), Gaussian projection (four fixtures), and the fixed
compositing-gradient diagnostic (one fixture). Run from a source checkout:

```sh
SINTERGPU_GPU_TESTS=1 PYTHONPATH=src:. python -m benchmarks.xsplat_timings \
  --gpu --iterations 50 --warmups 5 --output benchmarks/results/xsplat-cpu-gpu-DEVICE-VERSION.json
python tools/update_timings.py --input benchmarks/results/xsplat-cpu-gpu-DEVICE-VERSION.json
```

Omit `--gpu` for CPU-only samples. Hardware dispatch requires source/measurement
review and coordination with other agents. The benchmark owns its device,
allocations, submission, synchronization and readback. Library adapters still
borrow their caller's device and command context.

Adam uses31/32/1025/65539 elements and steps0/1/37, including zero gradients,
nonzero prior moments and positive second moments. Step0 follows the existing
shader's max(step,1) bias correction. Every sample resets values and both moments;
repeated dispatch is not allowed to change the measured workload. The shader's
32-thread profile is retained. Prepared random inputs are outside both regions.

Projection uses127/128/1025/65539 Gaussians, nonunit quaternions, anisotropic scales,
a nontrivial row-major camera rotation/translation, and visible, off-screen and
behind-camera points. Storage has80-byte Gaussian and48-byte projected strides;
camera uniform storage is96 bytes. The collector verifies actual runtime field
offsets, strides and workgroup sizes before dispatch. It requests explicit
row-major matrix compilation. Input and complete output storage have13 extra
records. Defined projected fields are compared; padding within a live struct is
unspecified and excluded. Complete records beyond the live count must retain all
seed bytes. Visibility flags are exact.

The benchmark-only independent NumPy oracles use float64 mathematical evaluation
on float32 inputs/uniforms, then produce float32 fields. Projection is also
compared with the retained, unmodified xsplat-vk C++ reference on their shared
finite valid domain. The newly retained original headers carry the same pinned
source provenance and license. The shader retains camera-space depth for rejected
points, whereas the C++ reference returns an all-zero result; separate hand-derived
tests preserve this distinction. No zero quaternion, nonfinite input, covariance
singularity, huge step, optimizer training convergence or renderer qualification
is claimed. The portable fixtures initially require2e-6 relative/absolute error
for defined floats, without changing any existing kernel tolerance. The existing
fixed gradient test retains its original2e-6 bounds. Projection's exact centered
CPU example is retained by independent analytical tests.

The backward probe always computes one fixed compositing derivative and launches
exactly one thread. Its result contains five defined floats in32-byte storage;
the remaining live padding is unspecified. This is a diagnostic cost, not a
scalable rasterizer or full-gradient benchmark.

CPU wall time includes oracle computation, temporary arrays and copying seeded
output storage. This is not an optimized CPU alternative. GPU timestamps cover
one entry after compilation/allocation/warmup and full output/inout reset. Host
submission/completion time is separate. Transfer-inclusive time uploads every
bound input/output seed and camera uniform buffer, dispatches, waits and reads all
complete output allocations. Parity is checked before and after each region.
Raw samples bind workload bytes, all uniforms, ordered bindings, launch extent,
strides, compiler matrix layout, exact committed collector/reference/helper hashes,
shader/reference trees, actual bundled compiler, device and software environment.
The8MiB per-buffer budget is a conservative fixture guard. These snapshots do not
select an optimized variant or establish controlled speedups, full-domain device
qualification, full renderer latency or training throughput.
