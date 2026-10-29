# Sampled-monitor 4070 Ti point-plane diagnostic

This unqualified trusted-host diagnostic ran commit `9e28b073ae89986c334cf7388f439a538caac927` on the RTX 4070 Ti. Three separate child processes each recorded 50 old/candidate/old triples after five warmups. All 450 measured resident and transfer readbacks, including parameters, passed the fixed fixture's independent scalar oracle and paired byte comparisons. The parent recorded its lease, guardian, child handshake, source and binary identities, shader reflection, policy and complete samples.

**The diagnostic does not support selecting the candidate.** Its preregistered descriptive analysis flags the transfer-inclusive no-regression condition as false in all three runs. Each p95 below is the nearest-rank estimate from 50 samples; the comparison uses the lower of the two old controls.

| Run | Old-before p95 | Candidate p95 | Old-after p95 | Candidate vs lower old p95 |
| --- | ---: | ---: | ---: | ---: |
| 0 | 344.977 µs | 331.814 µs | 313.781 µs | +5.75% |
| 1 | 289.162 µs | 317.589 µs | 321.077 µs | +9.83% |
| 2 | 301.988 µs | 318.087 µs | 324.463 µs | +5.33% |

The resident-pipeline and compute-stage descriptive flags passed, but transfer-inclusive bootstrap upper bounds were positive in all runs. These figures are not a hardware qualification or a general performance verdict: the p95 threshold is a point estimate, bootstrap assumptions and process independence are unverified, and compute timestamps are stage-level.

The predeclared policy allowed 0–3000 MHz SM clocks, memory clock at least 0 MHz, temperature up to 85 °C and power up to 285 W; it excluded other reported compute PIDs while dynamically allowing the gated child. It did not lock clocks. Five telemetry snapshots per child were roughly 297–309 ms apart, and the monitor cannot detect all short-lived work or all graphics/Vulkan peers. Xorg and GNOME Shell contexts remained present; the desktop session reported idle before and after. Telemetry polling and per-sample fsync may perturb timings. The selected ICD JSON and trusted-host build do not authenticate the loaded provider or physical backing.

`receipt.json.artifacts/` retains the sample journals, monitor traces, preregistration, source/build identity, analysis, shader binaries, executable, input and child logs. The three original 22,290,000-byte raw readback files were byte-identical, with SHA-256 `906b63472b9a44ac7e6f3de6b9c399ed66814f6e6c591958847cdd524a7e3f1c`; `shared-readbacks.raw.xz` is a lossless deduplicated copy of each. The original three raw files remain on `cps-wkstn-nvidia4070.local` under `/home/bjoern/.cache/sintergpu-qualification/direct-point-plane-4070-PdRxNvIk/monitored-9e28.9hSSy8QI/receipt.json.artifacts/`. `SHA256SUMS` covers the retained local files.
