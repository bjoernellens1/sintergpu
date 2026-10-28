# Proposed allocation and device correlation

The current V1 allocation query exposes a per-buffer generation, `VkBuffer`,
`VkDeviceMemory`, and sizes. It does not expose its owning `DeviceImpl` or that
device's creation-receipt generation. Comparing its numeric buffer and memory
handles with a separate device receipt therefore does not establish their
association. The existing hard execution gate remains required.

The isolated experimental RHI and SlangPy patches add a **new** COM IID and
schema while retaining V1. `BufferImpl` queries its actual `DeviceImpl` and
returns V1 allocation fields with the receipt's device generation and numeric
`VkDevice`; the backend checks the receipt's handle type before returning it.
The query rejects missing, unfinalized, external or mismatched device receipts,
unknown or zero UUID and allocations rejected by V1. The Python binding uses
RAII, checks SGL liveness and returns a value copy. The caller must retain both
the actual Device and Buffer owners and serialize close, export, and native
operations across checks and use; an RHI interface or copied dictionary is not
a device lease.

The isolated Debug RHI and SlangPy build passed. Synthetic CPU objects test
schema, absent/unfinalized/external/mismatched receipts, zero UUID, taint,
retained COM references and exception unwind. An isolated CPU-device smoke
rejects the query before and after SGL close. These do not test actual Vulkan
allocation, close, wrapped/shared/export or cross-device paths. The patches
require independent review and hardware preflight before any GPU use. This
correlation cannot establish the Vulkan loader,
ICD or driver binary identity, mapped bytes, absence of foreign native mutation,
or physical backing independence. No GPU or qualification claim follows from
matching handles or receipt generations.

The existing source seam is `src/vulkan/vk-buffer.cpp` `getObservation`, with
`src/vulkan/vk-device.h`'s private `m_creationReceipt`; SGL's device tracks its
receipt and clears it at close in `src/sgl/device/device.cpp`. The allocation
and device must be observed on that same live backend, not inferred by UUID,
name, separate `vulkaninfo` or a caller-supplied association.

The local RHI patch SHA-256 is
`aa6c1102fa6317e570aa7ab74e954fddb985511b2f4415f6d1d8292a4bed276b`; the
SlangPy patch is `3ad53e3474ce4f8aa6957c41b2eaee91fae270102f14a082ef56e8928bfba1e0`.
`test_rhi_buffer_device_correlation_patch.py` checks the exact local layered
input and post-apply file hashes; its local run passed two cases and skips when
the audited source is missing. That is not CI or complete build provenance.
The isolated Debug build uses GCC 16.2.1, Python 3.13.12 and
`SGL_WARNINGS_AS_ERRORS=OFF`. Reproduction in the prepared isolated lab:

```sh
cmake --build /dev/shm/sintergpu-allocation-correlation-lab/build --target rhi_buffer_device_correlation_cpu_probe slangpy_ext -j 8
VK_DRIVER_FILES=/nonexistent/sintergpu-no-vulkan-icd.json VK_ICD_FILENAMES=/nonexistent/sintergpu-no-vulkan-icd.json /dev/shm/sintergpu-allocation-correlation-lab/build/rhi_buffer_device_correlation_cpu_probe
PYTHONPATH=/dev/shm/sintergpu-allocation-correlation-lab/slangpy-source LD_LIBRARY_PATH=/dev/shm/sintergpu-allocation-correlation-lab/build:/home/bjoern/git/caspar-slang/.venv/lib/python3.13/site-packages/slangpy VK_DRIVER_FILES=/nonexistent/sintergpu-no-vulkan-icd.json VK_ICD_FILENAMES=/nonexistent/sintergpu-no-vulkan-icd.json /home/bjoern/miniforge3/bin/python tools/diagnostics/patches/slangpy-buffer-device-correlation-cpu-smoke.py
```

The local CPU probe binary SHA-256 is
`c27b6c69e1b08a7a6042734eaed8810e3bc7b5cb4842da77ea414668c7d517e6`.
These local paths and hashes do not authenticate another runtime or its loader.
