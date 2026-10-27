# Consumer migrations

Release [0.1.0](https://github.com/bjoernellens1/sintergpu/releases/tag/v0.1.0)
is immutable. Initial registration pins its exact source commit, source archive
and manifest digests. Registration provides dependency integrity checks; it does
not replace consumer kernels or establish native application qualification.

| Consumer | Delivery | Hosted integrity check |
| --- | --- | --- |
| stereoforge | [Registration PR](https://github.com/bjoernellens1/stereoforge/pull/2) | passed |
| eligsir-sim | [Registration PR](https://github.com/bjoernellens1/eligsir-sim/pull/42) | blocked-before-job-start |
| eligsir-native | [Registration PR](https://github.com/bjoernellens1/eligsir-native/pull/11) | blocked-before-job-start |
| vkl | [Registration PR](https://github.com/bjoernellens1/vkl/pull/131) | blocked-before-job-start |
| slang-rgbd-slam | [Registration PR](https://github.com/bjoernellens1/slang-rgbd-slam/pull/2) | blocked-before-job-start |
| caspar-slang | [Registration PR](https://github.com/bjoernellens1/caspar-slang/pull/2) | blocked-before-job-start |
| locus | [Registration PR](https://github.com/bjoernellens1/locus/pull/13) | blocked-before-job-start |

All seven PRs were reused by a real second delivery, with identical URLs and
branch commit hashes and no new commits. See the machine-readable
[delivery record](../qualification/delivery-0.1.0.json). Six private repository
jobs could not start because GitHub reported failed account payments or a
spending limit. These are pending checks, not correctness failures or passes.

## First shared-source integrations

- [StereoForge](https://github.com/bjoernellens1/stereoforge/pull/3): opt-in uint
  minimum, one channel, fixed 256-thread tree. Eighteen released-wheel checks
  passed locally, including 16 complete-output GPU comparisons on Radeon 8060S.
- [EliGSiR-native](https://github.com/bjoernellens1/eligsir-native/pull/10): opt-in
  float sum, five channels, same tree. Six released-wheel checks passed locally,
  including four forward/reduce/backward comparisons of partials, totals and
  gradients on Radeon 8060S. HIP runtime execution remains pending.

Both consume the identical released `common.block_reduce` module bytes:
`b963fc6f1cad8365068c60579f57b55c73c1ad3c3a696a1d3573b0cf627d9388` (SHA256). The wheel is pinned by URL and
SHA256, and the helper enters compile/cache identity. Caller resource, barrier,
submission and synchronization ownership remain explicit. Default routes remain
unchanged. Downstream reviewers retain merge/adoption control.

Native GPU comparisons, CPU source/cache CI, release integrity verification and
controlled performance qualification are distinct gates. Neither registration
nor a CPU runner enables an optimized route for an untested configuration.

## Updating a consumer

1. Verify the immutable release, checksums and exact publisher-workflow
   attestations before changing a pin.
2. Select modules and named numerical profiles; consult the
   [overlap audit](kernel-overlap.md) before treating similar operations as
   interchangeable. Preserve layout, initialization, overflow and invalid-value
   semantics.
3. Compare frozen donor and released implementations on identical inputs,
   including gradients and full pipeline results where applicable.
4. Run the consumer's checks. Record source/compiler/device/configuration
   identities and controlled resident/transfer-inclusive benchmark samples.
5. Review and merge downstream. Only then register actual consumed modules and
   native verification argv; never infer adoption from a notification PR.
