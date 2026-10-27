# CPU and GPU timing reference

The user's request extends the approved consolidation measurement task: provide
CPU and GPU execution times for every kernel in a reusable table/database. My
format choice is Git-tracked JSON plus generated Markdown, requiring no database
server or new runtime dependency. Every current catalog operation/variant appears.

Records retain exact workload/input and shader/reference hashes, CPU/GPU/compiler
identity, warmups, raw millisecond samples, median and p95. CPU reference wall time,
GPU kernel timestamps, resident pipeline, host submission and transfer-inclusive
costs are distinct. Missing/stale sources display explicit pending/stale status,
never zero or donor timings. Multiple workloads and machines remain separate rows;
do not select the fastest measurement or combine unmatched CPU/GPU workloads.

First reuse all 28 existing Locus fixtures for its 17 operations. Small correctness
fixtures are labeled, not claimed as production throughput. CPU timing includes
reference output construction with prepared inputs. GPU mutable output/state must
reset per invocation outside kernel timing; transfer-inclusive timing includes
all uploads/initialization, dispatch/completion and final output/state readbacks.
Compilation and allocation are setup. Reuse independent reference and parity
checks before recording timings. Then extend standalone recipes and matching CPU
references across every remaining catalog entry. A complete list of placeholders
alone does not fulfill the user's request or the end-to-end goal.
