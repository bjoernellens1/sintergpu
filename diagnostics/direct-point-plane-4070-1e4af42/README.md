# RTX 4070 Ti direct Vulkan point-plane engineering check

The reviewed single-fixture attempt at commit
`1e4af42e4ce1c2f491e0a3620d05a27edbdba513` exited 0 on the trusted
`cps-wkstn-nvidia4070` workstation. Donor and packaged candidate complete
readbacks match byte-for-byte and pass the independent float64 oracle, input
immutability and output-tail checks. The donor and candidate use 1024- and
256-thread workgroups respectively.

The raw packed input, both complete readbacks, subprocess stdout/stderr, and
`events.jsonl` are retained in `retained-receipt.json.artifacts/`. Both output
files have SHA-256
`5420738f46011e48fc62a80d4139f1f15b0b210544e8e7cfec403282cea045dd`.
After copying the files, the input, both outputs, oracle, per-buffer hashes and
four zero process exits were checked again locally. The event journal records
seven successful canonical lock checks and lock close before validation receipt.
Preflight reported zero GPU utilization and no compute applications immediately
before the attempt; that observation does not prove the absence of unrelated GPU
activity throughout the run. The initial c07b54c run is a separate record and
its deleted readbacks are not recovered by this run.

This is a fixed-fixture trusted-host engineering check. It does not authenticate
the ICD/provider, prove physical memory independence, qualify SlangPy or device
behavior, establish trajectory accuracy, or measure performance. Release and
kernel-selection gates remain unchanged. The original remote files are under
`/home/bjoern/.cache/sintergpu-qualification/direct-point-plane-4070-PdRxNvIk/`.
