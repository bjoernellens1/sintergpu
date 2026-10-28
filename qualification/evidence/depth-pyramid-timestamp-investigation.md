# 4070 Ti depth-pyramid timestamp investigation

The resident timing pilot at source `3ed55cb590eaba21e090805ef767f481967ff9f1` showed substantial between-loop drift and integer timestamp results on a 1,024-tick grid; it cannot establish a no-regression result. The 50-pair SlangPy diagnostic at `27da8270b276019d0ee8c7f4418ccbd0e2c42ee5` retained absolute integer ticks, all multiples of 1,024, at a reported 1 GHz frequency.

On the same reported GPU UUID and NVIDIA driver, the native Vulkan diagnostic at `ffd3c58bbdbf1582e86165cba39d623991d7aa92` captured 50 adjacent pairs at each of TOP_OF_PIPE and BOTTOM_OF_PIPE. Both stages contained all 32 multiples of 32 modulo 1,024; within-pair deltas were 64 or 96 ticks. Its period was 1 ns and queue timestampValidBits was 64. These observations show that stage alone does not explain the different grids in these captures. Queue equivalence to SlangPy, command composition, simultaneous conditions, and loaded driver binary identity were not established, so no cause or timer-resolution claim follows.

A read-only inspection of the 4070 qualification environment's SlangPy 0.43.1 installation recorded these binaries (actual process mappings were not observed):

| Artifact | SHA-256 | ELF build ID |
| --- | --- | --- |
| `slangpy_ext.cpython-312-x86_64-linux-gnu.so` | `eb0bb466acc78c5ed96cda1057cc4617e44f80ecd7c2933923e3dfba271a46c7` | `b4f92399c43d38ac4b3928ba59997efb014b96bd` |
| `libsgl.so` | `e525e1be8cf3198ac1c0c0fb2d90c19c9dcf6de2c15a6fd0bb8edcc0615588ff` | `fa27395328ae456355974543ee3bc56b374d9425` |
| `libslang-compiler.so.0.2026.12` | `195f1fbbf7342517aed3064657c8cfaaae5791065158189610253dce070c4678` | `c8ede7a266718b915eaa221d3fa9597389b40140` |

Static inspection found integer `get_results` in the packaged interface, double conversion in the separate `get_timestamp_results` function, and a Vulkan RHI query-result path with 8-byte stride and flags 3. These observations do not establish which Vulkan function was called dynamically or prove the source revision. The next discriminating evidence is a same-process, same-queue and matched-command native/SlangPy capture with loaded binary identities. Neither this investigation nor the resident pilot qualifies the candidate for adoption.
