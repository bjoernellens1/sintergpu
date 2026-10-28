# v0.2.6 downstream delivery observations

All seven registered consumers received update pull requests; see `delivery-0.2.6.json` for immutable heads and links. The delivery receipt records CI as pending at creation time and is not evidence of consumer qualification.

On 2026-10-28, StereoForge's release-integrity check passed and its two CPU jobs were still running. The six private-repository PRs reported failed checks. The EliGSiR-sim release-integrity job had no execution steps; its GitHub check annotation says the job was not started because recent account payments failed or the spending limit needs to be increased. This is a GitHub Actions billing/startup failure, not an observed test failure. Other private job failures require their own inspection before attributing the same cause. Re-run downstream checks after the account issue is resolved; do not infer native or GPU qualification from delivery alone.

Evidence: https://github.com/bjoernellens1/eligsir-sim/actions/runs/40735284779 and https://github.com/bjoernellens1/stereoforge/pull/17/checks.
