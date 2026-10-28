# RTX 4070 Ti direct Vulkan point-plane engineering check

The reviewed fixed-fixture attempt at SinterGPU commit
`c07b54ccef3921377e7aad8db9074960144ea1b4` exited 0 on the trusted
`cps-wkstn-nvidia4070` host. The runner compared all donor and packaged-candidate
buffer bytes, checked the independent float64 oracle, input immutability and
output tails, and wrote `trusted_host_engineering_parity_only` to the included
receipt. The entry points used 1024 and 256 threads respectively.

The GPU UUID and driver matched the pinned 4070 Ti identity. Immediately before
the attempt, `preflight-gpu.txt` reported zero GPU utilization and
`preflight-compute.txt` was empty; this observation does not prove the absence
of unrelated activity throughout the run. The reviewed runner acquired and
verified the canonical lock around child execution and validation. The native
subprocesses completed device teardown before readback publication. Both
attempt output streams were empty, and the process exited 0.

`engineering-receipt.json` contains source, compiler, shader and per-buffer
hashes; `build-receipt.json` records the target native rebuild. The raw output
buffers were inside the runner's temporary directory and were deleted after
validation; their hashes are retained but the bytes cannot be independently
rechecked from this artifact. Original preparation and receipts are also at
`/home/bjoern/.cache/sintergpu-qualification/direct-point-plane-4070-PdRxNvIk/`
on the 4070 host.

This is a single trusted-workstation engineering check. It does not establish
provider/ICD identity, physical memory independence, SlangPy parity, trajectory
accuracy, performance, or device qualification. It does not change release
qualification or kernel-selection gates.
