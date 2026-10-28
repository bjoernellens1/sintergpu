# Contributing kernels

SinterGPU is the upstream home for reusable kernel improvements across consuming
projects. Push a dedicated branch and open a PR; do not directly change a
consumer's pinned release or mark an untested variant qualified.

For new or upgraded kernels, use the [candidate intake protocol](docs/superpowers/specs/2026-10-28-kernel-candidate-intake-design.md): dedicated candidate branches and draft PRs, explicit validation evidence, then a separate reviewed promotion PR. Unqualified candidates stay outside the installed catalog. Rewrites must outperform or stay on par with equivalent incumbents across their supported configurations.

Include:

- Independent CPU reference, meaningful correctness tests and numerical contract.
- Immutable donor revision, source paths/hashes, authorship and compatible license.
- Reproducible benchmark evidence, separating kernel/pipeline/transfer costs.
- Capability requirements, resource layouts, scratch/initialization requirements.
- Changelog fragment classified as compatible or breaking, including numerical changes.

Preserve specialized implementations when faster. Changed source, compiler,
flags or layouts require fresh qualification for the affected configuration.
Missing hardware and skipped tests remain pending, never passes.

Run `python -m pytest tests -q`, `sinter validate` and both pinned
compiler lanes. Run GPU tests explicitly with `SINTERGPU_GPU_TESTS=1` on an
available dedicated device. Performance review precedes GPU qualification.

The source-only API and optional adapters must not create hidden devices/queues,
submit work or add readbacks. Allocation and synchronization stay with callers.
Keep application orchestration outside the reusable library.
