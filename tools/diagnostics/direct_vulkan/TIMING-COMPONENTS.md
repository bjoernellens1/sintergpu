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
