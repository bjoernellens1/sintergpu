# Native point-plane measurement adapter

This standalone caller owns its Vulkan context, shared descriptor set, seven device-local
buffers, retained host staging, pipelines, command buffer, fence, and timestamp queries.
It adds a **separate native profile**. Existing Python results and failures remain evidence;
a native result does not change the verdict of a Python profile or establish promotion.

Build (CPU only, requires C++17, Vulkan headers and loader):

```sh
candidates/sintergpu-optimization-families/native/build.sh /tmp/sintergpu-point-plane-ab
python candidates/sintergpu-optimization-families/native/cpu_tests.py --binary /tmp/sintergpu-point-plane-ab
```

Export a device-free fixture using the same representative arrays and poison as the
Python benchmark. `params.bin` contains six actual uint fields and two zero padding words;
the harness also accepts24/28-byte Params files and pads to32 bytes. All words are little endian.

```sh
python candidates/sintergpu-optimization-families/benchmarks/native_fixture.py \
  --fixture-dir /tmp/point-1025 --count 1025 --profile aosoa
/tmp/sintergpu-point-plane-ab --baseline /path/point-aosoa-baseline.spv \
  --candidate /path/point-aosoa-candidate.spv --fixture-dir /tmp/point-1025 --count 1025 \
  --vendor 0x1002 --device 0x1586 --output /tmp/point-1025-native.json --validate-only
```

Only after the physical GPU slot is granted, remove `--validate-only` to execute.
The runner acquires nonblocking exclusive `flock /tmp/sintergpu-gpu.lock` **before any
Vulkan instance/device work**; do not wrap it in a caller-held exclusive lock. It records
its PID/hostname and active/released state in `<output>.ownership.json`. Allowed adapter
identities are AMD1002:1586 and NVIDIA10de:2782; vendor/device must be explicit. CPU/software
adapters and ambiguous duplicate devices are rejected. Defaults are1000 ABBA blocks and10
warmups per variant per boundary. The order is independently baseline,candidate,candidate,baseline
for each block of each boundary. Each individual invocation is retained; no batching averages.

The SPIR-V must come from the separately recorded pinned Slang compiler lane, with entry
`main`, literal1024×1×1 local size, six storage buffers at set0 bindings0..5, and one
uniform Params at binding6. CPU validation checks SPIR-V framing, Shader-only capability,
entry, workgroup size, and binding decoration contract. It is deliberately not a complete
SPIR-V validator or reflection/type checker. The pinned shader reflection/compilation evidence
must establish descriptor types and numerical source identity. Compile artifacts, compiler
SHA256/version/flags, and profile must accompany native evidence. The runtime does not compile
or choose a different baseline. Vulkan1.2 and SPIR-V<=1.5 are required. No optional shader
arithmetic capability is accepted by the CPU ABI guard.

All host vectors, Vulkan allocations, pipeline creation, and descriptor updates happen
outside measurement. Both variants use exactly the same buffers/descriptors/staging.
Host staging **requires HOST_VISIBLE|HOST_CACHED**, preferring HOST_COHERENT; the run fails
if cached memory is unavailable. Memory types, flag bitmasks and allocation byte counts are
reported. Noncoherent maps use full-allocation flush/invalidate ranges (offset0 and
VK_WHOLE_SIZE, valid for the device's reported nonCoherentAtomSize).

The independent measured boundaries are:

* `gpu_timestamp`: one dispatch bracketed by compute-stage timestamps. The recorded wall
  duration also includes query reset/record/submit/fence; timestamp result retrieval occurs
  after that timer. Device timestamp valid-bit wrap is handled.
* `resident_completion`: CPU wall time from retained command/fence reset through recording,
  queue submission and fence completion. No uploads, downloads or query commands.
* `transfer_inclusive`: CPU wall time including memcpy of all six initial arrays plus32-byte
  Params into retained cached upload staging, required flush, command/fence reset, seven full
  GPU uploads, barriers, dispatch, full out_0 and out_1 downloads including poison padding,
  submit/fence completion, required invalidate and final CPU memcpy of both full outputs.

Before and after sampling, baseline and candidate each perform the complete transfer operation.
The runner writes `<output>.<before|after>.<baseline|candidate>.<out_0|out_1>.bin`, compares
all bytes (including padding), and checks untouched poison against the fixture. Any mismatch
aborts with nonzero status. The float64 oracle remains independent and required:

```sh
python candidates/sintergpu-optimization-families/benchmarks/native_fixture.py \
  --count 1025 --profile aosoa --check-native-output /tmp/point-1025-native.json
```

Raw JSON records include boundary/block/position/variant, `wall_ns`, and `gpu_ns` (null
outside the GPU boundary). Hardware/queue/driver/API, timestamp period, affinity, fixture
and SPIR-V FNV1a64 identities, current executable identity, and runtime source identity are
included. FNV1a is explicitly noncryptographic; retain external SHA256 artifact manifests.
A runtime source hash is not a build provenance assertion. Output dumps are outside timing.
The runner reports samples only after both differential checks pass; a nonzero process
status invalidates that run. Use fresh output paths to avoid confusing old reports with a
failed rerun. Apply the existing unchanged paired-block bootstrap and tail gates externally.

CPU-only validation and fixture/oracle tests do not validate Vulkan barriers, drivers, GPU
execution, native latency, performance gates, or independent shader numerical correctness.
