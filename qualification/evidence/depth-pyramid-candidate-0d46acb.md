# Experimental depth-pyramid candidate parity

Source commit: `0d46acb558267a089d1a6695e9b0271e5816e5fb`. The incumbent and shared-core candidate were executed on identical test inputs and compared bitwise, alongside independent CPU references. These results cover the retained fixture cases and do not qualify performance, full-domain behavior, application trajectories, or consumer integration.

| Device | Result | Driver | Evidence |
| --- | --- | --- | --- |
| Radeon 8060S | 67 passed, 0 skipped | RADV Mesa 26.2.3 | `depth-pyramid-candidate-8060s-0d46acb/` |
| RTX 4070 Ti | 67 passed, 0 skipped | NVIDIA 595.99.02 | `depth-pyramid-candidate-4070ti-0d46acb/` |

Both runs held the shared GPU lock and retained device and source identities. The Radeon before/after source manifests use different `./` path prefixes; comparison after path normalization confirms all 1,248 source hashes match. No performance-based selection was made.
