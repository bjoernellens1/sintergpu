# Immutable releases and consumer delivery

`VERSION` is the release version. `src/sintergpu/data/VERSION` must match it;
Python package metadata reads the root version dynamically. Tags are strict
`vMAJOR.MINOR.PATCH`, beginning with `v0.1.0`. Before 1.0, incompatible API,
numerical, ordering or layout changes increment minor; compatible additions and
fixes increment patch. Commit a dated, populated `CHANGELOG.md` release section
before tagging. An `unreleased` heading deliberately blocks publication.

The release workflow resolves the existing remote tag to a commit and runs the
maintained CPU contracts, catalog/provenance validation, two pinned compiler and
reflection lanes, and isolated wheel/source installation. Compiler archives are
SHA256 pinned. Actual compiler output, executable hashes, per-operation source
hashes and reflection are recorded. Source archives use `git archive` and gzip
with zero timestamp. The manifest binds source commit, shader-tree digest,
compiler records, qualifications, pending gates, payload sizes and SHA256 values.
`SHA256SUMS` covers all payloads and the manifest. The manifest excludes itself
and checksum file, avoiding a recursive hash dependency.

The workflow reads the versioned record under `qualification/`, labels the initial release experimental and explicitly records
pending hardware, consumer migration and controlled performance gates. This
is not evidence that those gates passed. To promote qualification, provide
reviewed per-operation records including adapter, compiler, source SHA256,
driver, device and result, and preserve the real benchmark adoption criteria.
Missing NVIDIA execution remains pending. No registration check qualifies a
kernel or authorizes automatic runtime selection.

`actions/attest@v4` attests payloads, manifest and checksums. Its bundle is
uploaded separately as `attestation.jsonl`, outside its own subject set.
Publication requires the repository's immutable-release setting. The publisher
creates a draft, uploads without clobber, compares every uploaded digest/size,
verifies source-bound workflow attestations and only then publishes. Existing
draft assets must match byte-for-byte; mismatches stop for investigation. Before uploading, the workflow retains the complete prepared payload and original attestation bundle as a 90-day Actions artifact. Draft retries restore that exact set from its original run, rather than rebuilding it. An expired/missing recovery set stops delivery until the original bytes are restored. Never
move a tag, delete/recreate a release, or overwrite an asset to repair a failure.
A published retry downloads and verifies the existing immutable release, without
building, committing, uploading or repackaging it.

For a tag run, publication and downstream delivery share one workflow and one
per-tag concurrency group. Do not rely on `release: published` events generated
by `GITHUB_TOKEN`. To retry delivery for an existing publication:

```bash
gh workflow run release.yml --ref v0.1.0 -f tag=v0.1.0
```

For an unpublished-tag manual run, `--ref` must identify that tag so the workflow
provenance source SHA equals the tagged source. Manual dispatch from an unrelated
branch is rejected. Release attestations require the GitHub-hosted runner;
`gh attestation verify` denies self-hosted signing runners.

## Initial consumer registration

`consumers.json` allowlists exactly seven repositories under `bjoernellens1`:
StereoForge, EliGSiR-sim, EliGSiR-native, VKL, slang-rgbd-slam, Caspar-Slang and
Locus. Actual repository names and observed default branches are stored in the
registry. Commands are argument arrays. Registration records contain no consumed
modules and no invented native test commands.

Use the locally authenticated `gh` session for explicit initial registration.
It needs permission to introduce the downstream workflow. Credentials remain in
the existing session; do not copy OAuth credentials into Actions secrets.

```bash
python3 tools/update_consumers.py --tag v0.1.0 --dry-run --register
python3 tools/update_consumers.py --tag v0.1.0 --deliver --register
```

`--dry-run` verifies publication and reads consumer state, but creates no remote
branches, commits or PRs. `--register` adds `.sintergpu/lock.json`, a dependency-free
`.sintergpu/verify_lock.py`, and `.github/workflows/sintergpu-dependency.yml`.
The downstream-owned verifier validates exact version/tag/source commit,
manifest/artifact URLs and hashes, manifest consistency, archive paths, immutable
release integrity and the source-bound publisher workflow attestation. It never
extracts or runs downloaded packages or shaders.

The PR title explicitly states "kernel migration pending". Registration proves
integrity and reviewable delivery only. A migration owner must later wire the
identical pinned shader digest into the consumer's real build/runtime, run its
existing differential tests, and provide exact native verification argv before
changing `integration_state` to `integrated`. StereoForge uint-MIN and
EliGSiR-native float-SUM require old/new output, tree and barrier parity.

## Recurring delivery

Install a dedicated GitHub App on only the seven registered consumers with
Contents write, Pull requests write and Attestations read. Configure `SINTERGPU_APP_ID` as an Actions
variable and `SINTERGPU_APP_PRIVATE_KEY` as its secret. Normal delivery only
updates the registered lock, so the App needs no Workflows permission. Each matrix
job mints a short-lived single-repository token with automatic revocation.
Missing App configuration is a failed job with an uploaded result, never a
successful skipped automation claim.

```bash
python3 tools/update_consumers.py --tag v0.1.0 --dry-run
python3 tools/update_consumers.py --tag v0.1.0 --deliver
```

The updater first verifies the published immutable release. The deterministic
branch is `sintergpu/v0.1.0`; a machine marker identifies the PR independently
of its title. All PR states are queried. Merged PRs require the expected lock at
the merge commit. Closed unmerged PRs are declined and never recreated. Matching
open PRs are reused. Human-modified generated files, branch commits or PR bodies
are conflicts. Existing generated branches with differing bytes stop for review;
updates never force a ref or touch the default branch. Creation races and
ambiguous PR-create responses are resolved by rereading state.

All seven consumers are processed even after partial failures. Each job uploads
`delivery-results.json` and writes a summary including PR, head/base commit and
check URLs where available. Delivery failure never replaces or rolls back the
published release. CI checks remain pending until their real downstream results
arrive; PR creation does not mean consumer tests passed. There is no auto-merge,
auto-approval or branch-protection bypass. Existing older release PRs remain open
for their reviewers.

## Local preparation and verification

Preparation requires a clean tagged checkout, prebuilt packages in `dist`, an
explicit qualification JSON record, and both compiler output JSON records
augmented with `pinned_release` (`2026.12` and `2026.19`). Compiler source coverage
must exactly match the catalog. Wheel and sdist source bytes must match the tracked tagged sources; matching package versions alone are insufficient.

```bash
python3 tools/release.py prepare --tag v0.1.0 --qualification qualification.json \
  --compiler compile-2026.12/results.json --compiler compile-2026.19/results.json
python3 tools/release.py verify --tag v0.1.0 --output /tmp/sintergpu-verified
python3 -m pytest tests/test_release.py tests/test_consumers.py -q
```

Operational acceptance still requires an authorized real immutable release,
seven reviewable registration PRs and a repeat delivery creating zero new PRs or
commits. Fake-API tests prove local state handling, not GitHub configuration,
attestation issuance, App installation or downstream/native execution.

## Publisher authorization

The repository immutability setting requires Administration read. The standard
workflow token cannot grant that permission. Configure a separate GitHub App
installed only on SinterGPU with Administration read, Contents write and
Attestations read. Set `SINTERGPU_PUBLISHER_APP_ID` and
`SINTERGPU_PUBLISHER_APP_PRIVATE_KEY`. The publishing step mints a short-lived
repository-scoped token; attestation signing continues under the workflow token.
Without this App, automatic publication fails visibly after retaining the signed
prepared set. An authorized owner can recover that original artifact locally and
publish it using the existing authenticated CLI session, without copying secrets
into Actions or rebuilding its payloads.

The first 0.1.0 run exposed this boundary and the published-tag endpoint's exclusion
of drafts. Its initial publication used owner-authorized local recovery of the
original hosted, attested bytes. Draft inspection uses the release ID returned by
the authenticated release list. Automatic publisher-App and downstream-App
execution remain pending configuration.
