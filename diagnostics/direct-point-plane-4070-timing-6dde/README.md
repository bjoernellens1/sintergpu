# RTX 4070 Ti point-plane timing diagnostic

This is an unqualified trusted-host engineering diagnostic, not a performance acceptance or no-regression result. On commit `6dde8dc965cd37364b9b22c5fca1f1d2ef2870f2`, three separate child processes each recorded 50 old/candidate/old triples after five warmups. All 450 measured resident and transfer readbacks, parameters, tails and inputs passed the fixed fixture's independent scalar oracle and paired byte comparisons. The parent held and verified the canonical lock; each child exited successfully. The receipt records source, shader ABI, compiler, binary, device and ICD configuration identities.

The 4070 was also hosting Xorg and GNOME Shell graphics contexts. Its desktop session reported idle and compute-process queries were empty; the recorded snapshots do not establish absence of all interference. The per-sample journal and raw-readback fsyncs occur outside measured intervals and can affect subsequent samples. Compute timestamps are stage-level; the runs lack independent environmental controls and statistical acceptance. The selected ICD JSON and trusted-host build do not authenticate the loaded provider or physical memory backing. No variant was selected from this result.

The sample journals and complete per-pass raw buffers are retained here. The three 22,290,000-byte raw readbacks were byte-identical: their common uncompressed SHA-256 is `906b63472b9a44ac7e6f3de6b9c399ed66814f6e6c591958847cdd524a7e3f1c`. `receipt.json.artifacts/shared-readbacks.raw.xz` is a lossless deduplicated copy; decompression recreates each run's raw file. All three original raw files remain on `cps-wkstn-nvidia4070.local` under `/home/bjoern/.cache/sintergpu-qualification/direct-point-plane-4070-PdRxNvIk/timing-6dde.kj2TwCLg/receipt.json.artifacts/`. The accompanying SHA-256 manifest covers the retained local files.

For context, the median times in microseconds, ordered old-before / candidate / old-after, were:

| Run | GPU compute stage | Resident pipeline | Host transfer-inclusive |
| --- | --- | --- | --- |
| 0 | 8.144 / 5.088 / 8.224 | 8.544 / 5.632 / 8.464 | 259.969 / 244.161 / 261.377 |
| 1 | 7.936 / 4.896 / 8.032 | 8.320 / 5.264 / 8.400 | 265.565 / 262.511 / 265.043 |
| 2 | 8.080 / 4.880 / 7.920 | 8.416 / 5.376 / 8.208 | 255.184 / 250.958 / 255.076 |

These descriptive medians are not the requested performance qualification. Raw samples, timestamp period, valid bits and completion bounds are retained for independent analysis.
