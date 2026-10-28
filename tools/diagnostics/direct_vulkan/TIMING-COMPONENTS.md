# Experimental point-plane timing components

These components are **not a runnable benchmark** and provide no performance evidence or qualification. They do not change the existing GPU qualification gates.

- `vulkan_timestamp_pair.h` records four raw Vulkan timestamps around the resident recording and compute stage. The compute-stage markers are stage-level timestamps, not proof of dispatch-exclusive time; timestamp resolution and driver behavior are unverified.
- `vulkan_transfer_pass.h` records explicit staging copies and memory dependencies. `vulkan_point_plane_record.h` records the point-plane dispatch and resident or transfer pass.
- `vulkan_point_plane_timing.h` uses caller-provided Vulkan functions to submit and wait. Its resident host bound covers submission through queue completion. Its transfer interval covers host uploads, command recording and checks, submission, wait, and raw host readbacks; validation and resets are outside it. An operation failure poisons the timing instance.
- `paired_timing_protocol.py` provides same-process old/candidate/old ordering, raw samples and descriptive summaries. Caller-provided validation and identity are assertions, not verified evidence; its results cannot select or qualify an implementation. Three loops in one process are not independent controlled runs.

The caller must separately verify source and device identity, hold the canonical GPU lock, provision valid non-aliased buffers and coherent mappings, poison/reset and independently validate every relevant input, output and guard before each reset, and preserve evidence. For each timestamp interval, its independently measured host completion upper bound must be shorter than one full timestamp counter wrap; the protocol rejects ambiguous intervals. These obligations and the native components are not yet integrated into the fixed-fixture executable. No actual GPU timing has been collected by this scaffold.

CPU-only checks, from the repository root, compile fake Vulkan callbacks without linking a Vulkan loader:

```sh
python3 -m pytest -q tools/diagnostics/direct_vulkan/test_paired_timing_protocol.py
for t in test_vulkan_timestamp_pair test_vulkan_transfer_pass test_vulkan_point_plane_record test_vulkan_point_plane_timing; do
  c++ -std=c++17 -Wall -Wextra -Werror "tools/diagnostics/direct_vulkan/$t.cpp" -o "/tmp/sinter-$t" && "/tmp/sinter-$t" || exit 1
done
```

Fake callbacks check command order and failure handling; they do not exercise a Vulkan driver, shader correctness, physical memory independence, or hardware timing.

## Integrated work in progress

`timing_executable.cpp` is a separately compiled, **unconditionally blocked production executable**. Its fixed-fixture implementation creates one selected device, seven upload/device/download buffer triplets, two point-plane pipelines, timestamp queries and explicit reset/readback transfers. Each resident and transfer output is checked against the scalar oracle, all six returned fixture buffers and parameters are checked, and each old/candidate/old triple must match byte-for-byte. Each of three same-process loops performs five validated warmups and at least 50 measured iterations. Those loops are not independent controlled runs. The per-row journal contains raw query ticks, host completion bounds, transfer times and full-output hashes; individual rows may precede completion of a triple. A consumer must reject incomplete triples or failed runs. The per-row journal write and `fsync` occur outside reported intervals but may perturb subsequent samples; the journal does not retain full readback bytes or guarantee directory-entry durability.

The test-only build accepts synthetic shader words and calls a fake-only marker. `test_timing_executable_fake.py` links its fake Vulkan functions directly without a Vulkan loader. The fake copies previously retained donor output during recording, so it tests orchestration and failure handling, not actual Vulkan execution, shader results or hardware timing. The fake links distinct pipeline handles and checks their workgroup counts. Simulated wait failures after resident submission exercise the drain-before-query-destruction ordering; on an unsuccessful device drain the child exits immediately without running destructors. The parent must retain the lock until child exit and treat such an exit as a failure.

Before any GPU attempt, an independently reviewed parent runner must authenticate the complete committed source and compiled shader ABI, the native executable and its build, hold and verify the canonical lock across child execution and teardown, verify target and peer conditions, retain full raw readbacks and source/device evidence, and publish explicit failure/success receipts. It must validate complete triples, timestamp bounds and statistical acceptance; the current three loops, quantization and stage-level markers do not establish independent performance qualification. Review of failure handling, filesystem and runtime/provider assumptions is also pending. The hard production block must remain until that review clears an explicitly scoped attempt.

Additional CPU-only tests:

```sh
python3 -m pytest -q tools/diagnostics/direct_vulkan/test_paired_timing_protocol.py tools/diagnostics/direct_vulkan/test_timing_executable_fake.py
for t in test_timing_resource_plan test_timing_resources test_timing_pipelines test_timing_oracle test_timing_readback test_timing_tick_bound test_timing_drain; do
  c++ -std=c++17 -Wall -Wextra -Werror "tools/diagnostics/direct_vulkan/$t.cpp" -o "/tmp/sinter-$t" && "/tmp/sinter-$t" || exit 1
done
```

The oracle test reads the retained engineering fixture files in `diagnostics/direct-point-plane-4070-1e4af42/`; their presence is required for this local test. No GPU execution or performance result is claimed by these checks.
