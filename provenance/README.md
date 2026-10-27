# Source provenance

Each donor JSON pins an immutable source revision and records source path/hash,
destination/hash and modifications. Importing uses Git blobs, excluding dirty
working-tree contents. `tools/import_sources.py` performs reproducible imports;
`tools/materialize_shaders.py` composes dispatchable modules without application
runtime dependencies. Retained snapshots are reference and verification resources.

The user authorized public Apache-2.0 redistribution of original code they own
on 2026-10-27. Upstream permissive licenses and notices remain intact. Missing
third-party redistribution permission is not supplied by fork ownership.

Research-only simple-knn-rocm and unresolved RTG-SLAM-cuda_utils code are excluded.
VKKT and video-optimize-cuda contribute consumer contracts/use cases, not custom
kernel bodies. Full ML engines and renderer orchestration remain dependencies or
future consumers.
