# v0.2.6 downstream delivery observations

All seven registered consumers received update pull requests; see `delivery-0.2.6.json` for immutable heads and links. The delivery receipt records CI as pending at creation time and is not evidence of consumer qualification.

On 2026-10-28, StereoForge's release-integrity check and both CPU jobs passed. The six private-repository PRs reported failed checks. The release-integrity job in each of the six private repositories had no execution steps; every GitHub check annotation says the job was not started because recent account payments failed or the spending limit needs to be increased. These six integrity checks therefore failed at GitHub Actions startup, without running tests. Other failed jobs require their own inspection before attributing the same cause. Re-run downstream checks after the account issue is resolved; do not infer native or GPU qualification from delivery alone.

Evidence: https://github.com/bjoernellens1/eligsir-sim/actions/runs/40735284779 and https://github.com/bjoernellens1/stereoforge/pull/17/checks.
