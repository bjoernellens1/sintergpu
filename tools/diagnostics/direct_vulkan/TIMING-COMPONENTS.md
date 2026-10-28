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

`timing_executable.cpp` is a separately compiled, **unconditionally blocked production executable**. Its fixed-fixture implementation creates one selected device, seven upload/device/download buffer triplets, two point-plane pipelines, timestamp queries and explicit reset/readback transfers. Each resident and transfer output is checked against the scalar oracle, all six returned fixture buffers and parameters are checked, and each old/candidate/old triple must match byte-for-byte. Each native process performs five validated warmups and at least 50 measured ABA iterations. The experimental parent launches three separate processes; process isolation alone does not establish independent environmental controls. The per-row journal contains raw query ticks, host completion bounds, transfer times and full-output hashes; complete resident and transfer fixture and parameter readbacks are saved separately; individual rows may precede completion of a triple. A consumer must reject incomplete triples or failed runs. The per-row and raw-readback journal writes and `fsync` occur outside reported intervals but may perturb subsequent samples; the native journal does not guarantee directory-entry durability.

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

`timing_parent.py` is a draft, independently reviewed only after its current changes; both its CLI and `run` remain unconditionally blocked. Its intended source-checked path stages the native source closure from a specified commit, builds twice with the pinned compiler, checks shader identity and ABI, holds a parent lock through three child processes, and validates ordered complete samples and every raw readback against the independent reference. CPU mock tests exercise child failure and ordering; they do not establish target-host build, process isolation, runtime/provider identity, comprehensive peer activity, or hardware timing. Early setup failures, failure-receipt publication and all lock/teardown paths require independent review before enabling. `nvidia-smi` compute-process checks are limited to reported compute processes and do not prove absence of graphics or other activity. No statistical improvement or no-regression decision has been made.

Additional CPU checks: `TMPDIR=/dev/shm/sinter-timing-tests python3 -m pytest -q tools/diagnostics/direct_vulkan/test_timing_parent.py tools/diagnostics/direct_vulkan/test_timing_report.py tools/diagnostics/direct_vulkan/test_timing_executable_fake.py` (create the temporary directory first). These tests may write tens of megabytes of fake readbacks.

## Separate 4070 trusted-host candidate

`timing_4070_trusted.cpp` explicitly enables the narrow native diagnostic without changing the default executable gate. `timing_4070_parent.py` has a provisional target binary pin and remains unconditionally blocked pending review. Both paths are restricted to exactly 50 measured ABA triples per process; the parent starts three separate processes under one lease. Process separation, `nvidia-smi` compute-process snapshots and the available clocks do not establish independent environmental controls or a no-regression result. Each sample retains its resident and transfer readbacks, including parameters; the parent checks full bytes, oracle and ordering. No GPU execution is authorized until the exact committed sources and target build have been reviewed.

A CPU-only target build of the current uncommitted candidate source on `cps-wkstn-nvidia4070.local`, in `/home/bjoern/.cache/sinter-timing-build.mjw6YlJS`, produced identical binary SHA-256 `f8fbce518a37ee0a22e4b74700820b6aa526bfbab273e9984620dffce3ffdca5` on two builds (also on an absolute-source-path build). The compiler `/usr/bin/c++` SHA-256 was `1353e9bdd29a7295c7226bf6c63abccce056d8cac31f112e5cdbecc3f28c2769`; the source archive SHA-256 was `2295d13431b1d28431c524d3ccd68634ed9625d2ea5ad08f5ccb7f3d7f43d539`. Flags were `-std=c++17 -O0 -g0 -Wall -Wextra -Werror -Wno-missing-field-initializers`, linked with `-lvulkan -lcrypto`. This is reproducible build evidence for those bytes only, not committed-source, loader, driver or runtime attestation. Rebuild and independently review after the source is committed. The target binary was not run.

CPU-only candidate checks use the same directly linked fake backend and require no Vulkan loader:

```sh
mkdir -p /dev/shm/sinter-timing-tests
TMPDIR=/dev/shm/sinter-timing-tests python3 -m pytest -q tools/diagnostics/direct_vulkan/test_timing_4070_parent.py tools/diagnostics/direct_vulkan/test_timing_parent.py tools/diagnostics/direct_vulkan/test_timing_report.py tools/diagnostics/direct_vulkan/test_timing_executable_fake.py
```

The same CPU fake suite passed on the target host in the isolated `test-root` under the build directory above: 12 tests passed. The transferred test archive SHA-256 was `0f3544cf5ba52a1e3daaba6e358f462142e439c7ce4b103325841aed6ab48d13`. These tests used the fake Vulkan callbacks and did not execute the target GPU. They cover the uncommitted source snapshot; committed-source verification remains outstanding.

A separate proposed entry point, `timing_4070_reviewed_parent.py`, removes only the candidate parent's unconditional review block and includes its own bytes in the committed source closure. It requires `--execute`; do not run it on GPU before independent review of its exact commit, target rebuild and controls. The original `timing_4070_parent.py` and default timing parent stay blocked. On the 4070 host, the isolated `source-retained` checkout was advanced to commit `347b4ef27ff64c57032e9b00143088a62df278c8`; its checked 20-file source closure rebuilt twice to the provisional `f8fbce...` binary above and its CPU-only fake suite passed 12 tests. The proposed separate entry point was added after that commit and therefore needs its own committed-source and target recheck.
