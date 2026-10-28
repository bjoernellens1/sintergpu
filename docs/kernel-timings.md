# CPU and GPU kernel timing reference

Explicit workload/configuration timings, not optimization or application qualification. CPU reference implementations are not necessarily optimized CPU alternatives.

Median; p95 uses linear interpolation at (n-1)*0.95. Raw samples retained.

Coverage: 0 / 137 operations have a source-matched paired CPU/GPU baseline. Pending entries remain part of the required measurement work.

Times are median / p95 **milliseconds**. Each row is one workload and environment; pipeline costs are never copied onto individual kernels. Small correctness fixtures are not production-throughput benchmarks. Historical sources are explicitly stale. Raw samples and full identities are in [the timing registry](../src/sintergpu/data/timings.json).

| Operation / variant | Workload | CPU reference | GPU kernel | GPU pipeline | Transfer-inclusive | Environment | State |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| caspar_slang.gn.reducePartials / portable-port | pending | pending | pending | pending | pending | pending | pending |
| caspar_slang.gn.reduceResidual / portable-port | pending | pending | pending | pending | pending | pending | pending |
| caspar_slang.gn.reduceResidualSubgroup / portable-port | pending | pending | pending | pending | pending | pending | pending |
| caspar_slang.gn.solveSystem / portable-port | pending | pending | pending | pending | pending | pending | pending |
| caspar_slang.point_plane_residual.computeMain / portable-port | pending | pending | pending | pending | pending | pending | pending |
| caspar_slang.pose_graph.checkCost / portable-port | pending | pending | pending | pending | pending | pending | pending |
| caspar_slang.pose_graph.evaluateEdges / portable-port | pending | pending | pending | pending | pending | pending | pending |
| caspar_slang.pose_graph.initializeCG / portable-port | pending | pending | pending | pending | pending | pending | pending |
| caspar_slang.pose_graph.initializeMeasurements / portable-port | pending | pending | pending | pending | pending | pending | pending |
| caspar_slang.pose_graph.linearizeEdges / portable-port | pending | pending | pending | pending | pending | pending | pending |
| caspar_slang.pose_graph.operatorEdges / portable-port | pending | pending | pending | pending | pending | pending | pending |
| caspar_slang.pose_graph.operatorNodes / portable-port | pending | pending | pending | pending | pending | pending | pending |
| caspar_slang.pose_graph.pcgStep / portable-port | pending | pending | pending | pending | pending | pending | pending |
| caspar_slang.pose_graph.resetEdgeCount / portable-port | pending | pending | pending | pending | pending | pending | pending |
| caspar_slang.pose_graph.resetGraph / portable-port | pending | pending | pending | pending | pending | pending | pending |
| caspar_slang.pose_graph.rollbackNodes / portable-port | pending | pending | pending | pending | pending | pending | pending |
| caspar_slang.pose_graph.updateNodes / portable-port | pending | pending | pending | pending | pending | pending | pending |
| caspar_slang.reprojection.buildReprojection / portable-port | pending | pending | pending | pending | pending | pending | pending |
| caspar_slang.reprojection.evaluateReprojection / portable-port | pending | pending | pending | pending | pending | pending | pending |
| caspar_slang.reprojection.resetReprojection / portable-port | pending | pending | pending | pending | pending | pending | pending |
| caspar_slang.reprojection.resetReprojectionCount / portable-port | pending | pending | pending | pending | pending | pending | pending |
| caspar_slang.reprojection.updateReprojection / portable-port | pending | pending | pending | pending | pending | pending | pending |
| depth.bilateral / portable-imported | locus-fixture/bilateral-invalid | stale | stale | pending | stale | AMD RYZEN AI MAX+ PRO 395 w/ Radeon 8060S; AMD Radeon 8060S Graphics (RADV STRIX_HALO); 2026-10-27T20:20:04.561625+00:00 | stale source |
| depth.bilateral / portable-imported | locus-fixture/bilateral-invalid | stale | stale | pending | stale | AMD RYZEN AI MAX+ PRO 395 w/ Radeon 8060S; AMD Radeon 8060S Graphics (RADV STRIX_HALO); 2026-10-27T22:17:46.284752+00:00 | stale source |
| depth.register / portable-imported | locus-fixture/identity-registration | stale | stale | pending | stale | AMD RYZEN AI MAX+ PRO 395 w/ Radeon 8060S; AMD Radeon 8060S Graphics (RADV STRIX_HALO); 2026-10-27T20:20:04.561625+00:00 | stale source |
| depth.register / portable-imported | locus-fixture/nearest-depth-collision | stale | stale | pending | stale | AMD RYZEN AI MAX+ PRO 395 w/ Radeon 8060S; AMD Radeon 8060S Graphics (RADV STRIX_HALO); 2026-10-27T20:20:04.561625+00:00 | stale source |
| depth.register / portable-imported | locus-fixture/identity-registration | stale | stale | pending | stale | AMD RYZEN AI MAX+ PRO 395 w/ Radeon 8060S; AMD Radeon 8060S Graphics (RADV STRIX_HALO); 2026-10-27T22:17:46.284752+00:00 | stale source |
| depth.register / portable-imported | locus-fixture/nearest-depth-collision | stale | stale | pending | stale | AMD RYZEN AI MAX+ PRO 395 w/ Radeon 8060S; AMD Radeon 8060S Graphics (RADV STRIX_HALO); 2026-10-27T22:17:46.284752+00:00 | stale source |
| depth.resolve / portable-imported | locus-fixture/resolve-invalid | stale | stale | pending | stale | AMD RYZEN AI MAX+ PRO 395 w/ Radeon 8060S; AMD Radeon 8060S Graphics (RADV STRIX_HALO); 2026-10-27T20:20:04.561625+00:00 | stale source |
| depth.resolve / portable-imported | locus-fixture/resolve-invalid | stale | stale | pending | stale | AMD RYZEN AI MAX+ PRO 395 w/ Radeon 8060S; AMD Radeon 8060S Graphics (RADV STRIX_HALO); 2026-10-27T22:17:46.284752+00:00 | stale source |
| depth.validity / portable-imported | locus-fixture/invalid-depth-nan | stale | stale | pending | stale | AMD RYZEN AI MAX+ PRO 395 w/ Radeon 8060S; AMD Radeon 8060S Graphics (RADV STRIX_HALO); 2026-10-27T20:20:04.561625+00:00 | stale source |
| depth.validity / portable-imported | locus-fixture/invalid-depth-nan | stale | stale | pending | stale | AMD RYZEN AI MAX+ PRO 395 w/ Radeon 8060S; AMD Radeon 8060S Graphics (RADV STRIX_HALO); 2026-10-27T22:17:46.284752+00:00 | stale source |
| geometry.backproject / portable-imported | locus-fixture/optical-backproject | stale | stale | pending | stale | AMD RYZEN AI MAX+ PRO 395 w/ Radeon 8060S; AMD Radeon 8060S Graphics (RADV STRIX_HALO); 2026-10-27T20:20:04.561625+00:00 | stale source |
| geometry.backproject / portable-imported | locus-fixture/optical-backproject | stale | stale | pending | stale | AMD RYZEN AI MAX+ PRO 395 w/ Radeon 8060S; AMD Radeon 8060S Graphics (RADV STRIX_HALO); 2026-10-27T22:17:46.284752+00:00 | stale source |
| geometry.coverage_heatmap / portable-imported | locus-fixture/coverage-pitched | stale | stale | pending | stale | AMD RYZEN AI MAX+ PRO 395 w/ Radeon 8060S; AMD Radeon 8060S Graphics (RADV STRIX_HALO); 2026-10-27T20:20:04.561625+00:00 | stale source |
| geometry.coverage_heatmap / portable-imported | locus-fixture/coverage-pitched | stale | stale | pending | stale | AMD RYZEN AI MAX+ PRO 395 w/ Radeon 8060S; AMD Radeon 8060S Graphics (RADV STRIX_HALO); 2026-10-27T22:17:46.284752+00:00 | stale source |
| geometry.coverage_points / portable-imported | locus-fixture/coverage-point-preview | stale | stale | pending | stale | AMD RYZEN AI MAX+ PRO 395 w/ Radeon 8060S; AMD Radeon 8060S Graphics (RADV STRIX_HALO); 2026-10-27T20:20:04.561625+00:00 | stale source |
| geometry.coverage_points / portable-imported | locus-fixture/coverage-point-preview | stale | stale | pending | stale | AMD RYZEN AI MAX+ PRO 395 w/ Radeon 8060S; AMD Radeon 8060S Graphics (RADV STRIX_HALO); 2026-10-27T22:17:46.284752+00:00 | stale source |
| geometry.transform_points / portable-imported | locus-fixture/transform-validity | stale | stale | pending | stale | AMD RYZEN AI MAX+ PRO 395 w/ Radeon 8060S; AMD Radeon 8060S Graphics (RADV STRIX_HALO); 2026-10-27T20:20:04.561625+00:00 | stale source |
| geometry.transform_points / portable-imported | locus-fixture/transform-validity | stale | stale | pending | stale | AMD RYZEN AI MAX+ PRO 395 w/ Radeon 8060S; AMD Radeon 8060S Graphics (RADV STRIX_HALO); 2026-10-27T22:17:46.284752+00:00 | stale source |
| image.crop_normalize / portable-imported | locus-fixture/crop-normalize | stale | stale | pending | stale | AMD RYZEN AI MAX+ PRO 395 w/ Radeon 8060S; AMD Radeon 8060S Graphics (RADV STRIX_HALO); 2026-10-27T20:20:04.561625+00:00 | stale source |
| image.crop_normalize / portable-imported | locus-fixture/crop-normalize | stale | stale | pending | stale | AMD RYZEN AI MAX+ PRO 395 w/ Radeon 8060S; AMD Radeon 8060S Graphics (RADV STRIX_HALO); 2026-10-27T22:17:46.284752+00:00 | stale source |
| image.nv12_to_rgb / portable-port | color-pitched/nv12_decode/6x4-to-6x4/bt601-limited | stale | stale | pending | stale | AMD RYZEN AI MAX+ PRO 395 w/ Radeon 8060S; AMD Radeon 8060S Graphics (RADV STRIX_HALO); 2026-10-27T20:55:54.662521+00:00 | stale source |
| image.nv12_to_rgb / portable-port | color-pitched/nv12_decode/6x4-to-6x4/bt601-full | stale | stale | pending | stale | AMD RYZEN AI MAX+ PRO 395 w/ Radeon 8060S; AMD Radeon 8060S Graphics (RADV STRIX_HALO); 2026-10-27T20:55:54.662521+00:00 | stale source |
| image.nv12_to_rgb / portable-port | color-pitched/nv12_decode/6x4-to-6x4/bt709-limited | stale | stale | pending | stale | AMD RYZEN AI MAX+ PRO 395 w/ Radeon 8060S; AMD Radeon 8060S Graphics (RADV STRIX_HALO); 2026-10-27T20:55:54.662521+00:00 | stale source |
| image.nv12_to_rgb / portable-port | color-pitched/nv12_decode/6x4-to-6x4/bt709-full | stale | stale | pending | stale | AMD RYZEN AI MAX+ PRO 395 w/ Radeon 8060S; AMD Radeon 8060S Graphics (RADV STRIX_HALO); 2026-10-27T20:55:54.662521+00:00 | stale source |
| image.nv12_to_rgb / portable-port | color-pitched/nv12_decode/640x480-to-640x480/bt709-limited | stale | stale | pending | stale | AMD RYZEN AI MAX+ PRO 395 w/ Radeon 8060S; AMD Radeon 8060S Graphics (RADV STRIX_HALO); 2026-10-27T20:55:54.662521+00:00 | stale source |
| image.nv12_to_rgb / portable-port | color-pitched/nv12_decode/6x4-to-6x4/bt601-limited | stale | stale | pending | stale | AMD RYZEN AI MAX+ PRO 395 w/ Radeon 8060S; AMD Radeon 8060S Graphics (RADV STRIX_HALO); 2026-10-27T22:18:17.867070+00:00 | stale source |
| image.nv12_to_rgb / portable-port | color-pitched/nv12_decode/6x4-to-6x4/bt601-full | stale | stale | pending | stale | AMD RYZEN AI MAX+ PRO 395 w/ Radeon 8060S; AMD Radeon 8060S Graphics (RADV STRIX_HALO); 2026-10-27T22:18:17.867070+00:00 | stale source |
| image.nv12_to_rgb / portable-port | color-pitched/nv12_decode/6x4-to-6x4/bt709-limited | stale | stale | pending | stale | AMD RYZEN AI MAX+ PRO 395 w/ Radeon 8060S; AMD Radeon 8060S Graphics (RADV STRIX_HALO); 2026-10-27T22:18:17.867070+00:00 | stale source |
| image.nv12_to_rgb / portable-port | color-pitched/nv12_decode/6x4-to-6x4/bt709-full | stale | stale | pending | stale | AMD RYZEN AI MAX+ PRO 395 w/ Radeon 8060S; AMD Radeon 8060S Graphics (RADV STRIX_HALO); 2026-10-27T22:18:17.867070+00:00 | stale source |
| image.nv12_to_rgb / portable-port | color-pitched/nv12_decode/640x480-to-640x480/bt709-limited | stale | stale | pending | stale | AMD RYZEN AI MAX+ PRO 395 w/ Radeon 8060S; AMD Radeon 8060S Graphics (RADV STRIX_HALO); 2026-10-27T22:18:17.867070+00:00 | stale source |
| image.planar_yuv_to_rgb / portable-imported | locus-fixture/yuv-matrix0-range0 | stale | stale | pending | stale | AMD RYZEN AI MAX+ PRO 395 w/ Radeon 8060S; AMD Radeon 8060S Graphics (RADV STRIX_HALO); 2026-10-27T20:20:04.561625+00:00 | stale source |
| image.planar_yuv_to_rgb / portable-imported | locus-fixture/yuv-matrix0-range1 | stale | stale | pending | stale | AMD RYZEN AI MAX+ PRO 395 w/ Radeon 8060S; AMD Radeon 8060S Graphics (RADV STRIX_HALO); 2026-10-27T20:20:04.561625+00:00 | stale source |
| image.planar_yuv_to_rgb / portable-imported | locus-fixture/yuv-matrix1-range0 | stale | stale | pending | stale | AMD RYZEN AI MAX+ PRO 395 w/ Radeon 8060S; AMD Radeon 8060S Graphics (RADV STRIX_HALO); 2026-10-27T20:20:04.561625+00:00 | stale source |
| image.planar_yuv_to_rgb / portable-imported | locus-fixture/yuv-matrix1-range1 | stale | stale | pending | stale | AMD RYZEN AI MAX+ PRO 395 w/ Radeon 8060S; AMD Radeon 8060S Graphics (RADV STRIX_HALO); 2026-10-27T20:20:04.561625+00:00 | stale source |
| image.planar_yuv_to_rgb / portable-imported | locus-fixture/yuv-matrix0-range0 | stale | stale | pending | stale | AMD RYZEN AI MAX+ PRO 395 w/ Radeon 8060S; AMD Radeon 8060S Graphics (RADV STRIX_HALO); 2026-10-27T22:17:46.284752+00:00 | stale source |
| image.planar_yuv_to_rgb / portable-imported | locus-fixture/yuv-matrix0-range1 | stale | stale | pending | stale | AMD RYZEN AI MAX+ PRO 395 w/ Radeon 8060S; AMD Radeon 8060S Graphics (RADV STRIX_HALO); 2026-10-27T22:17:46.284752+00:00 | stale source |
| image.planar_yuv_to_rgb / portable-imported | locus-fixture/yuv-matrix1-range0 | stale | stale | pending | stale | AMD RYZEN AI MAX+ PRO 395 w/ Radeon 8060S; AMD Radeon 8060S Graphics (RADV STRIX_HALO); 2026-10-27T22:17:46.284752+00:00 | stale source |
| image.planar_yuv_to_rgb / portable-imported | locus-fixture/yuv-matrix1-range1 | stale | stale | pending | stale | AMD RYZEN AI MAX+ PRO 395 w/ Radeon 8060S; AMD Radeon 8060S Graphics (RADV STRIX_HALO); 2026-10-27T22:17:46.284752+00:00 | stale source |
| image.remap_bilinear / portable-port | color-pitched/remap/6x4-to-7x5/bt601-limited | stale | stale | pending | stale | AMD RYZEN AI MAX+ PRO 395 w/ Radeon 8060S; AMD Radeon 8060S Graphics (RADV STRIX_HALO); 2026-10-27T20:55:54.662521+00:00 | stale source |
| image.remap_bilinear / portable-port | color-pitched/remap/6x4-to-7x5/bt601-full | stale | stale | pending | stale | AMD RYZEN AI MAX+ PRO 395 w/ Radeon 8060S; AMD Radeon 8060S Graphics (RADV STRIX_HALO); 2026-10-27T20:55:54.662521+00:00 | stale source |
| image.remap_bilinear / portable-port | color-pitched/remap/6x4-to-7x5/bt709-limited | stale | stale | pending | stale | AMD RYZEN AI MAX+ PRO 395 w/ Radeon 8060S; AMD Radeon 8060S Graphics (RADV STRIX_HALO); 2026-10-27T20:55:54.662521+00:00 | stale source |
| image.remap_bilinear / portable-port | color-pitched/remap/6x4-to-7x5/bt709-full | stale | stale | pending | stale | AMD RYZEN AI MAX+ PRO 395 w/ Radeon 8060S; AMD Radeon 8060S Graphics (RADV STRIX_HALO); 2026-10-27T20:55:54.662521+00:00 | stale source |
| image.remap_bilinear / portable-port | color-pitched/remap/640x480-to-64x48/bt709-limited | stale | stale | pending | stale | AMD RYZEN AI MAX+ PRO 395 w/ Radeon 8060S; AMD Radeon 8060S Graphics (RADV STRIX_HALO); 2026-10-27T20:55:54.662521+00:00 | stale source |
| image.remap_bilinear / portable-port | color-pitched/remap/6x4-to-7x5/bt601-limited | stale | stale | pending | stale | AMD RYZEN AI MAX+ PRO 395 w/ Radeon 8060S; AMD Radeon 8060S Graphics (RADV STRIX_HALO); 2026-10-27T22:18:17.867070+00:00 | stale source |
| image.remap_bilinear / portable-port | color-pitched/remap/6x4-to-7x5/bt601-full | stale | stale | pending | stale | AMD RYZEN AI MAX+ PRO 395 w/ Radeon 8060S; AMD Radeon 8060S Graphics (RADV STRIX_HALO); 2026-10-27T22:18:17.867070+00:00 | stale source |
| image.remap_bilinear / portable-port | color-pitched/remap/6x4-to-7x5/bt709-limited | stale | stale | pending | stale | AMD RYZEN AI MAX+ PRO 395 w/ Radeon 8060S; AMD Radeon 8060S Graphics (RADV STRIX_HALO); 2026-10-27T22:18:17.867070+00:00 | stale source |
| image.remap_bilinear / portable-port | color-pitched/remap/6x4-to-7x5/bt709-full | stale | stale | pending | stale | AMD RYZEN AI MAX+ PRO 395 w/ Radeon 8060S; AMD Radeon 8060S Graphics (RADV STRIX_HALO); 2026-10-27T22:18:17.867070+00:00 | stale source |
| image.remap_bilinear / portable-port | color-pitched/remap/640x480-to-64x48/bt709-limited | stale | stale | pending | stale | AMD RYZEN AI MAX+ PRO 395 w/ Radeon 8060S; AMD Radeon 8060S Graphics (RADV STRIX_HALO); 2026-10-27T22:18:17.867070+00:00 | stale source |
| image.resize_bilinear / portable-port | color-pitched/resize/6x4-to-7x5/bt601-limited | stale | stale | pending | stale | AMD RYZEN AI MAX+ PRO 395 w/ Radeon 8060S; AMD Radeon 8060S Graphics (RADV STRIX_HALO); 2026-10-27T20:55:54.662521+00:00 | stale source |
| image.resize_bilinear / portable-port | color-pitched/resize/6x4-to-7x5/bt601-full | stale | stale | pending | stale | AMD RYZEN AI MAX+ PRO 395 w/ Radeon 8060S; AMD Radeon 8060S Graphics (RADV STRIX_HALO); 2026-10-27T20:55:54.662521+00:00 | stale source |
| image.resize_bilinear / portable-port | color-pitched/resize/6x4-to-7x5/bt709-limited | stale | stale | pending | stale | AMD RYZEN AI MAX+ PRO 395 w/ Radeon 8060S; AMD Radeon 8060S Graphics (RADV STRIX_HALO); 2026-10-27T20:55:54.662521+00:00 | stale source |
| image.resize_bilinear / portable-port | color-pitched/resize/6x4-to-7x5/bt709-full | stale | stale | pending | stale | AMD RYZEN AI MAX+ PRO 395 w/ Radeon 8060S; AMD Radeon 8060S Graphics (RADV STRIX_HALO); 2026-10-27T20:55:54.662521+00:00 | stale source |
| image.resize_bilinear / portable-port | color-pitched/resize/640x480-to-64x48/bt709-limited | stale | stale | pending | stale | AMD RYZEN AI MAX+ PRO 395 w/ Radeon 8060S; AMD Radeon 8060S Graphics (RADV STRIX_HALO); 2026-10-27T20:55:54.662521+00:00 | stale source |
| image.resize_bilinear / portable-port | color-pitched/resize/6x4-to-7x5/bt601-limited | stale | stale | pending | stale | AMD RYZEN AI MAX+ PRO 395 w/ Radeon 8060S; AMD Radeon 8060S Graphics (RADV STRIX_HALO); 2026-10-27T22:18:17.867070+00:00 | stale source |
| image.resize_bilinear / portable-port | color-pitched/resize/6x4-to-7x5/bt601-full | stale | stale | pending | stale | AMD RYZEN AI MAX+ PRO 395 w/ Radeon 8060S; AMD Radeon 8060S Graphics (RADV STRIX_HALO); 2026-10-27T22:18:17.867070+00:00 | stale source |
| image.resize_bilinear / portable-port | color-pitched/resize/6x4-to-7x5/bt709-limited | stale | stale | pending | stale | AMD RYZEN AI MAX+ PRO 395 w/ Radeon 8060S; AMD Radeon 8060S Graphics (RADV STRIX_HALO); 2026-10-27T22:18:17.867070+00:00 | stale source |
| image.resize_bilinear / portable-port | color-pitched/resize/6x4-to-7x5/bt709-full | stale | stale | pending | stale | AMD RYZEN AI MAX+ PRO 395 w/ Radeon 8060S; AMD Radeon 8060S Graphics (RADV STRIX_HALO); 2026-10-27T22:18:17.867070+00:00 | stale source |
| image.resize_bilinear / portable-port | color-pitched/resize/640x480-to-64x48/bt709-limited | stale | stale | pending | stale | AMD RYZEN AI MAX+ PRO 395 w/ Radeon 8060S; AMD Radeon 8060S Graphics (RADV STRIX_HALO); 2026-10-27T22:18:17.867070+00:00 | stale source |
| image.resize_nearest / portable-imported | locus-fixture/resize-odd-pitched | stale | stale | pending | stale | AMD RYZEN AI MAX+ PRO 395 w/ Radeon 8060S; AMD Radeon 8060S Graphics (RADV STRIX_HALO); 2026-10-27T20:20:04.561625+00:00 | stale source |
| image.resize_nearest / portable-imported | locus-fixture/resize-odd-pitched | stale | stale | pending | stale | AMD RYZEN AI MAX+ PRO 395 w/ Radeon 8060S; AMD Radeon 8060S Graphics (RADV STRIX_HALO); 2026-10-27T22:17:46.284752+00:00 | stale source |
| image.rgb_to_nv12 / portable-port | color-pitched/nv12_encode/6x4-to-6x4/bt601-limited | stale | stale | pending | stale | AMD RYZEN AI MAX+ PRO 395 w/ Radeon 8060S; AMD Radeon 8060S Graphics (RADV STRIX_HALO); 2026-10-27T20:55:54.662521+00:00 | stale source |
| image.rgb_to_nv12 / portable-port | color-pitched/nv12_encode/6x4-to-6x4/bt601-full | stale | stale | pending | stale | AMD RYZEN AI MAX+ PRO 395 w/ Radeon 8060S; AMD Radeon 8060S Graphics (RADV STRIX_HALO); 2026-10-27T20:55:54.662521+00:00 | stale source |
| image.rgb_to_nv12 / portable-port | color-pitched/nv12_encode/6x4-to-6x4/bt709-limited | stale | stale | pending | stale | AMD RYZEN AI MAX+ PRO 395 w/ Radeon 8060S; AMD Radeon 8060S Graphics (RADV STRIX_HALO); 2026-10-27T20:55:54.662521+00:00 | stale source |
| image.rgb_to_nv12 / portable-port | color-pitched/nv12_encode/6x4-to-6x4/bt709-full | stale | stale | pending | stale | AMD RYZEN AI MAX+ PRO 395 w/ Radeon 8060S; AMD Radeon 8060S Graphics (RADV STRIX_HALO); 2026-10-27T20:55:54.662521+00:00 | stale source |
| image.rgb_to_nv12 / portable-port | color-pitched/nv12_encode/640x480-to-640x480/bt709-limited | stale | stale | pending | stale | AMD RYZEN AI MAX+ PRO 395 w/ Radeon 8060S; AMD Radeon 8060S Graphics (RADV STRIX_HALO); 2026-10-27T20:55:54.662521+00:00 | stale source |
| image.rgb_to_nv12 / portable-port | color-pitched/nv12_encode/6x4-to-6x4/bt601-limited | stale | stale | pending | stale | AMD RYZEN AI MAX+ PRO 395 w/ Radeon 8060S; AMD Radeon 8060S Graphics (RADV STRIX_HALO); 2026-10-27T22:18:17.867070+00:00 | stale source |
| image.rgb_to_nv12 / portable-port | color-pitched/nv12_encode/6x4-to-6x4/bt601-full | stale | stale | pending | stale | AMD RYZEN AI MAX+ PRO 395 w/ Radeon 8060S; AMD Radeon 8060S Graphics (RADV STRIX_HALO); 2026-10-27T22:18:17.867070+00:00 | stale source |
| image.rgb_to_nv12 / portable-port | color-pitched/nv12_encode/6x4-to-6x4/bt709-limited | stale | stale | pending | stale | AMD RYZEN AI MAX+ PRO 395 w/ Radeon 8060S; AMD Radeon 8060S Graphics (RADV STRIX_HALO); 2026-10-27T22:18:17.867070+00:00 | stale source |
| image.rgb_to_nv12 / portable-port | color-pitched/nv12_encode/6x4-to-6x4/bt709-full | stale | stale | pending | stale | AMD RYZEN AI MAX+ PRO 395 w/ Radeon 8060S; AMD Radeon 8060S Graphics (RADV STRIX_HALO); 2026-10-27T22:18:17.867070+00:00 | stale source |
| image.rgb_to_nv12 / portable-port | color-pitched/nv12_encode/640x480-to-640x480/bt709-limited | stale | stale | pending | stale | AMD RYZEN AI MAX+ PRO 395 w/ Radeon 8060S; AMD Radeon 8060S Graphics (RADV STRIX_HALO); 2026-10-27T22:18:17.867070+00:00 | stale source |
| image.undistort / portable-imported | locus-fixture/identity-undistort | stale | stale | pending | stale | AMD RYZEN AI MAX+ PRO 395 w/ Radeon 8060S; AMD Radeon 8060S Graphics (RADV STRIX_HALO); 2026-10-27T20:20:04.561625+00:00 | stale source |
| image.undistort / portable-imported | locus-fixture/distortion-outside | stale | stale | pending | stale | AMD RYZEN AI MAX+ PRO 395 w/ Radeon 8060S; AMD Radeon 8060S Graphics (RADV STRIX_HALO); 2026-10-27T20:20:04.561625+00:00 | stale source |
| image.undistort / portable-imported | locus-fixture/identity-undistort | stale | stale | pending | stale | AMD RYZEN AI MAX+ PRO 395 w/ Radeon 8060S; AMD Radeon 8060S Graphics (RADV STRIX_HALO); 2026-10-27T22:17:46.284752+00:00 | stale source |
| image.undistort / portable-imported | locus-fixture/distortion-outside | stale | stale | pending | stale | AMD RYZEN AI MAX+ PRO 395 w/ Radeon 8060S; AMD Radeon 8060S Graphics (RADV STRIX_HALO); 2026-10-27T22:17:46.284752+00:00 | stale source |
| image.yuyv_remap_fused / portable-port | color-pitched/fused_remap/6x4-to-7x5/bt601-limited | stale | stale | pending | stale | AMD RYZEN AI MAX+ PRO 395 w/ Radeon 8060S; AMD Radeon 8060S Graphics (RADV STRIX_HALO); 2026-10-27T20:55:54.662521+00:00 | stale source |
| image.yuyv_remap_fused / portable-port | color-pitched/fused_remap/6x4-to-7x5/bt601-full | stale | stale | pending | stale | AMD RYZEN AI MAX+ PRO 395 w/ Radeon 8060S; AMD Radeon 8060S Graphics (RADV STRIX_HALO); 2026-10-27T20:55:54.662521+00:00 | stale source |
| image.yuyv_remap_fused / portable-port | color-pitched/fused_remap/6x4-to-7x5/bt709-limited | stale | stale | pending | stale | AMD RYZEN AI MAX+ PRO 395 w/ Radeon 8060S; AMD Radeon 8060S Graphics (RADV STRIX_HALO); 2026-10-27T20:55:54.662521+00:00 | stale source |
| image.yuyv_remap_fused / portable-port | color-pitched/fused_remap/6x4-to-7x5/bt709-full | stale | stale | pending | stale | AMD RYZEN AI MAX+ PRO 395 w/ Radeon 8060S; AMD Radeon 8060S Graphics (RADV STRIX_HALO); 2026-10-27T20:55:54.662521+00:00 | stale source |
| image.yuyv_remap_fused / portable-port | color-pitched/fused_remap/640x480-to-64x48/bt709-limited | stale | stale | pending | stale | AMD RYZEN AI MAX+ PRO 395 w/ Radeon 8060S; AMD Radeon 8060S Graphics (RADV STRIX_HALO); 2026-10-27T20:55:54.662521+00:00 | stale source |
| image.yuyv_remap_fused / portable-port | color-pitched/fused_remap/6x4-to-7x5/bt601-limited | stale | stale | pending | stale | AMD RYZEN AI MAX+ PRO 395 w/ Radeon 8060S; AMD Radeon 8060S Graphics (RADV STRIX_HALO); 2026-10-27T22:18:17.867070+00:00 | stale source |
| image.yuyv_remap_fused / portable-port | color-pitched/fused_remap/6x4-to-7x5/bt601-full | stale | stale | pending | stale | AMD RYZEN AI MAX+ PRO 395 w/ Radeon 8060S; AMD Radeon 8060S Graphics (RADV STRIX_HALO); 2026-10-27T22:18:17.867070+00:00 | stale source |
| image.yuyv_remap_fused / portable-port | color-pitched/fused_remap/6x4-to-7x5/bt709-limited | stale | stale | pending | stale | AMD RYZEN AI MAX+ PRO 395 w/ Radeon 8060S; AMD Radeon 8060S Graphics (RADV STRIX_HALO); 2026-10-27T22:18:17.867070+00:00 | stale source |
| image.yuyv_remap_fused / portable-port | color-pitched/fused_remap/6x4-to-7x5/bt709-full | stale | stale | pending | stale | AMD RYZEN AI MAX+ PRO 395 w/ Radeon 8060S; AMD Radeon 8060S Graphics (RADV STRIX_HALO); 2026-10-27T22:18:17.867070+00:00 | stale source |
| image.yuyv_remap_fused / portable-port | color-pitched/fused_remap/640x480-to-64x48/bt709-limited | stale | stale | pending | stale | AMD RYZEN AI MAX+ PRO 395 w/ Radeon 8060S; AMD Radeon 8060S Graphics (RADV STRIX_HALO); 2026-10-27T22:18:17.867070+00:00 | stale source |
| image.yuyv_to_rgb / portable-port | color-pitched/yuyv_decode/6x4-to-6x4/bt601-limited | stale | stale | pending | stale | AMD RYZEN AI MAX+ PRO 395 w/ Radeon 8060S; AMD Radeon 8060S Graphics (RADV STRIX_HALO); 2026-10-27T20:55:54.662521+00:00 | stale source |
| image.yuyv_to_rgb / portable-port | color-pitched/yuyv_decode/6x4-to-6x4/bt601-full | stale | stale | pending | stale | AMD RYZEN AI MAX+ PRO 395 w/ Radeon 8060S; AMD Radeon 8060S Graphics (RADV STRIX_HALO); 2026-10-27T20:55:54.662521+00:00 | stale source |
| image.yuyv_to_rgb / portable-port | color-pitched/yuyv_decode/6x4-to-6x4/bt709-limited | stale | stale | pending | stale | AMD RYZEN AI MAX+ PRO 395 w/ Radeon 8060S; AMD Radeon 8060S Graphics (RADV STRIX_HALO); 2026-10-27T20:55:54.662521+00:00 | stale source |
| image.yuyv_to_rgb / portable-port | color-pitched/yuyv_decode/6x4-to-6x4/bt709-full | stale | stale | pending | stale | AMD RYZEN AI MAX+ PRO 395 w/ Radeon 8060S; AMD Radeon 8060S Graphics (RADV STRIX_HALO); 2026-10-27T20:55:54.662521+00:00 | stale source |
| image.yuyv_to_rgb / portable-port | color-pitched/yuyv_decode/640x480-to-640x480/bt709-limited | stale | stale | pending | stale | AMD RYZEN AI MAX+ PRO 395 w/ Radeon 8060S; AMD Radeon 8060S Graphics (RADV STRIX_HALO); 2026-10-27T20:55:54.662521+00:00 | stale source |
| image.yuyv_to_rgb / portable-port | color-pitched/yuyv_decode/6x4-to-6x4/bt601-limited | stale | stale | pending | stale | AMD RYZEN AI MAX+ PRO 395 w/ Radeon 8060S; AMD Radeon 8060S Graphics (RADV STRIX_HALO); 2026-10-27T22:18:17.867070+00:00 | stale source |
| image.yuyv_to_rgb / portable-port | color-pitched/yuyv_decode/6x4-to-6x4/bt601-full | stale | stale | pending | stale | AMD RYZEN AI MAX+ PRO 395 w/ Radeon 8060S; AMD Radeon 8060S Graphics (RADV STRIX_HALO); 2026-10-27T22:18:17.867070+00:00 | stale source |
| image.yuyv_to_rgb / portable-port | color-pitched/yuyv_decode/6x4-to-6x4/bt709-limited | stale | stale | pending | stale | AMD RYZEN AI MAX+ PRO 395 w/ Radeon 8060S; AMD Radeon 8060S Graphics (RADV STRIX_HALO); 2026-10-27T22:18:17.867070+00:00 | stale source |
| image.yuyv_to_rgb / portable-port | color-pitched/yuyv_decode/6x4-to-6x4/bt709-full | stale | stale | pending | stale | AMD RYZEN AI MAX+ PRO 395 w/ Radeon 8060S; AMD Radeon 8060S Graphics (RADV STRIX_HALO); 2026-10-27T22:18:17.867070+00:00 | stale source |
| image.yuyv_to_rgb / portable-port | color-pitched/yuyv_decode/640x480-to-640x480/bt709-limited | stale | stale | pending | stale | AMD RYZEN AI MAX+ PRO 395 w/ Radeon 8060S; AMD Radeon 8060S Graphics (RADV STRIX_HALO); 2026-10-27T22:18:17.867070+00:00 | stale source |
| perception.cosine_similarity / portable-imported | locus-fixture/similarity-count0 | stale | stale | pending | stale | AMD RYZEN AI MAX+ PRO 395 w/ Radeon 8060S; AMD Radeon 8060S Graphics (RADV STRIX_HALO); 2026-10-27T20:20:04.561625+00:00 | stale source |
| perception.cosine_similarity / portable-imported | locus-fixture/similarity-count3 | stale | stale | pending | stale | AMD RYZEN AI MAX+ PRO 395 w/ Radeon 8060S; AMD Radeon 8060S Graphics (RADV STRIX_HALO); 2026-10-27T20:20:04.561625+00:00 | stale source |
| perception.cosine_similarity / portable-imported | locus-fixture/similarity-nan-zero | stale | stale | pending | stale | AMD RYZEN AI MAX+ PRO 395 w/ Radeon 8060S; AMD Radeon 8060S Graphics (RADV STRIX_HALO); 2026-10-27T20:20:04.561625+00:00 | stale source |
| perception.cosine_similarity / portable-imported | locus-fixture/similarity-count0 | stale | stale | pending | stale | AMD RYZEN AI MAX+ PRO 395 w/ Radeon 8060S; AMD Radeon 8060S Graphics (RADV STRIX_HALO); 2026-10-27T22:17:46.284752+00:00 | stale source |
| perception.cosine_similarity / portable-imported | locus-fixture/similarity-count3 | stale | stale | pending | stale | AMD RYZEN AI MAX+ PRO 395 w/ Radeon 8060S; AMD Radeon 8060S Graphics (RADV STRIX_HALO); 2026-10-27T22:17:46.284752+00:00 | stale source |
| perception.cosine_similarity / portable-imported | locus-fixture/similarity-nan-zero | stale | stale | pending | stale | AMD RYZEN AI MAX+ PRO 395 w/ Radeon 8060S; AMD Radeon 8060S Graphics (RADV STRIX_HALO); 2026-10-27T22:17:46.284752+00:00 | stale source |
| perception.mask_postprocess / portable-imported | locus-fixture/mask-mode0 | stale | stale | pending | stale | AMD RYZEN AI MAX+ PRO 395 w/ Radeon 8060S; AMD Radeon 8060S Graphics (RADV STRIX_HALO); 2026-10-27T20:20:04.561625+00:00 | stale source |
| perception.mask_postprocess / portable-imported | locus-fixture/mask-mode1 | stale | stale | pending | stale | AMD RYZEN AI MAX+ PRO 395 w/ Radeon 8060S; AMD Radeon 8060S Graphics (RADV STRIX_HALO); 2026-10-27T20:20:04.561625+00:00 | stale source |
| perception.mask_postprocess / portable-imported | locus-fixture/mask-mode2 | stale | stale | pending | stale | AMD RYZEN AI MAX+ PRO 395 w/ Radeon 8060S; AMD Radeon 8060S Graphics (RADV STRIX_HALO); 2026-10-27T20:20:04.561625+00:00 | stale source |
| perception.mask_postprocess / portable-imported | locus-fixture/mask-mode0 | stale | stale | pending | stale | AMD RYZEN AI MAX+ PRO 395 w/ Radeon 8060S; AMD Radeon 8060S Graphics (RADV STRIX_HALO); 2026-10-27T22:17:46.284752+00:00 | stale source |
| perception.mask_postprocess / portable-imported | locus-fixture/mask-mode1 | stale | stale | pending | stale | AMD RYZEN AI MAX+ PRO 395 w/ Radeon 8060S; AMD Radeon 8060S Graphics (RADV STRIX_HALO); 2026-10-27T22:17:46.284752+00:00 | stale source |
| perception.mask_postprocess / portable-imported | locus-fixture/mask-mode2 | stale | stale | pending | stale | AMD RYZEN AI MAX+ PRO 395 w/ Radeon 8060S; AMD Radeon 8060S Graphics (RADV STRIX_HALO); 2026-10-27T22:17:46.284752+00:00 | stale source |
| primitives.compact.compactPrepare / portable-port | pending | pending | pending | pending | pending | pending | pending |
| primitives.compact.compactScatter / portable-port | pending | pending | pending | pending | pending | pending | pending |
| primitives.scan.scanBlocks / portable-port | pending | pending | pending | pending | pending | pending | pending |
| primitives.scan.scanCarry / portable-port | pending | pending | pending | pending | pending | pending | pending |
| primitives.scan.scanExclusive / portable-port | pending | pending | pending | pending | pending | pending | pending |
| primitives.scan_uint.scanBlocks / portable-port | pending | pending | pending | pending | pending | pending | pending |
| primitives.scan_uint.scanCarry / portable-port | pending | pending | pending | pending | pending | pending | pending |
| primitives.scan_uint.scanExclusive / portable-port | pending | pending | pending | pending | pending | pending | pending |
| quality.metrics / portable-imported | locus-fixture/quality-metrics | stale | stale | pending | stale | AMD RYZEN AI MAX+ PRO 395 w/ Radeon 8060S; AMD Radeon 8060S Graphics (RADV STRIX_HALO); 2026-10-27T20:20:04.561625+00:00 | stale source |
| quality.metrics / portable-imported | locus-fixture/quality-nan | stale | stale | pending | stale | AMD RYZEN AI MAX+ PRO 395 w/ Radeon 8060S; AMD Radeon 8060S Graphics (RADV STRIX_HALO); 2026-10-27T20:20:04.561625+00:00 | stale source |
| quality.metrics / portable-imported | locus-fixture/quality-metrics | stale | stale | pending | stale | AMD RYZEN AI MAX+ PRO 395 w/ Radeon 8060S; AMD Radeon 8060S Graphics (RADV STRIX_HALO); 2026-10-27T22:17:46.284752+00:00 | stale source |
| quality.metrics / portable-imported | locus-fixture/quality-nan | stale | stale | pending | stale | AMD RYZEN AI MAX+ PRO 395 w/ Radeon 8060S; AMD Radeon 8060S Graphics (RADV STRIX_HALO); 2026-10-27T22:17:46.284752+00:00 | stale source |
| quality.reduce / portable-imported | locus-fixture/quality-reduction | stale | stale | pending | stale | AMD RYZEN AI MAX+ PRO 395 w/ Radeon 8060S; AMD Radeon 8060S Graphics (RADV STRIX_HALO); 2026-10-27T20:20:04.561625+00:00 | stale source |
| quality.reduce / portable-imported | locus-fixture/quality-reduction | stale | stale | pending | stale | AMD RYZEN AI MAX+ PRO 395 w/ Radeon 8060S; AMD Radeon 8060S Graphics (RADV STRIX_HALO); 2026-10-27T22:17:46.284752+00:00 | stale source |
| slang_rgbd_slam.dense_ba.commitDensePoses / portable-port | pending | pending | pending | pending | pending | pending | pending |
| slang_rgbd_slam.dense_ba.finalizeDense / portable-port | pending | pending | pending | pending | pending | pending | pending |
| slang_rgbd_slam.dense_ba.initializeDense / portable-port | pending | pending | pending | pending | pending | pending | pending |
| slang_rgbd_slam.dense_ba.latchDenseFailure / portable-port | pending | pending | pending | pending | pending | pending | pending |
| slang_rgbd_slam.dense_ba.linearizeSchur / portable-port | pending | pending | pending | pending | pending | pending | pending |
| slang_rgbd_slam.dense_ba.reduceSchur / portable-port | pending | pending | pending | pending | pending | pending | pending |
| slang_rgbd_slam.dense_ba.rollbackDense / portable-port | pending | pending | pending | pending | pending | pending | pending |
| slang_rgbd_slam.dense_ba.saveInitialCost / portable-port | pending | pending | pending | pending | pending | pending | pending |
| slang_rgbd_slam.dense_ba.solveSchur / portable-port | pending | pending | pending | pending | pending | pending | pending |
| slang_rgbd_slam.dense_ba.updateDenseDepth / portable-port | pending | pending | pending | pending | pending | pending | pending |
| slang_rgbd_slam.dense_ba.updateDensePoses / portable-port | pending | pending | pending | pending | pending | pending | pending |
| slang_rgbd_slam.dense_tracker_subgroup.associate / portable-port | pending | pending | pending | pending | pending | pending | pending |
| slang_rgbd_slam.dense_tracker_subgroup.downsample / portable-port | pending | pending | pending | pending | pending | pending | pending |
| slang_rgbd_slam.dense_tracker_subgroup.downsampleIntensity / portable-port | pending | pending | pending | pending | pending | pending | pending |
| slang_rgbd_slam.dense_tracker_subgroup.fusedAssociate / portable-port | pending | pending | pending | pending | pending | pending | pending |
| slang_rgbd_slam.dense_tracker_subgroup.makeNormals / portable-port | pending | pending | pending | pending | pending | pending | pending |
| slang_rgbd_slam.dense_tracker_subgroup.makeVertices / portable-port | pending | pending | pending | pending | pending | pending | pending |
| slang_rgbd_slam.dense_tracker_subgroup.resetCount / portable-port | pending | pending | pending | pending | pending | pending | pending |
| slang_rgbd_slam.dense_tracker_subgroup.rgbIntensity / portable-port | pending | pending | pending | pending | pending | pending | pending |
| slang_rgbd_slam.dense_tracker_subgroup.updatePose / portable-port | pending | pending | pending | pending | pending | pending | pending |
| slang_rgbd_slam.dense_tracker_workgroup.associate / portable-port | pending | pending | pending | pending | pending | pending | pending |
| slang_rgbd_slam.dense_tracker_workgroup.downsample / portable-port | pending | pending | pending | pending | pending | pending | pending |
| slang_rgbd_slam.dense_tracker_workgroup.downsampleIntensity / portable-port | pending | pending | pending | pending | pending | pending | pending |
| slang_rgbd_slam.dense_tracker_workgroup.fusedAssociate / portable-port | pending | pending | pending | pending | pending | pending | pending |
| slang_rgbd_slam.dense_tracker_workgroup.makeNormals / portable-port | pending | pending | pending | pending | pending | pending | pending |
| slang_rgbd_slam.dense_tracker_workgroup.makeVertices / portable-port | pending | pending | pending | pending | pending | pending | pending |
| slang_rgbd_slam.dense_tracker_workgroup.resetCount / portable-port | pending | pending | pending | pending | pending | pending | pending |
| slang_rgbd_slam.dense_tracker_workgroup.rgbIntensity / portable-port | pending | pending | pending | pending | pending | pending | pending |
| slang_rgbd_slam.dense_tracker_workgroup.updatePose / portable-port | pending | pending | pending | pending | pending | pending | pending |
| slang_rgbd_slam.features.describe / portable-port | pending | pending | pending | pending | pending | pending | pending |
| slang_rgbd_slam.features.downsampleGray / portable-port | pending | pending | pending | pending | pending | pending | pending |
| slang_rgbd_slam.features.fastDetect / portable-port | pending | pending | pending | pending | pending | pending | pending |
| slang_rgbd_slam.features.filterMatches / portable-port | pending | pending | pending | pending | pending | pending | pending |
| slang_rgbd_slam.features.matchForward / portable-port | pending | pending | pending | pending | pending | pending | pending |
| slang_rgbd_slam.features.matchReverse / portable-port | pending | pending | pending | pending | pending | pending | pending |
| slang_rgbd_slam.features.nonmaxHarris / portable-port | pending | pending | pending | pending | pending | pending | pending |
| slang_rgbd_slam.features.resetFeatures / portable-port | pending | pending | pending | pending | pending | pending | pending |
| slang_rgbd_slam.features.resetMatchCount / portable-port | pending | pending | pending | pending | pending | pending | pending |
| slang_rgbd_slam.features.rgbToGray / portable-port | pending | pending | pending | pending | pending | pending | pending |
| slang_rgbd_slam.features.selectCells / portable-port | pending | pending | pending | pending | pending | pending | pending |
| slang_rgbd_slam.loop_verification.buildRefinement / portable-port | pending | pending | pending | pending | pending | pending | pending |
| slang_rgbd_slam.loop_verification.compactPairs / portable-port | pending | pending | pending | pending | pending | pending | pending |
| slang_rgbd_slam.loop_verification.finishVerification / portable-port | pending | pending | pending | pending | pending | pending | pending |
| slang_rgbd_slam.loop_verification.resetVerification / portable-port | pending | pending | pending | pending | pending | pending | pending |
| slang_rgbd_slam.loop_verification.scoreHypotheses / portable-port | pending | pending | pending | pending | pending | pending | pending |
| slang_rgbd_slam.loop_verification.selectHypothesis / portable-port | pending | pending | pending | pending | pending | pending | pending |
| slang_rgbd_slam.tracker.associate / portable-port | pending | pending | pending | pending | pending | pending | pending |
| slang_rgbd_slam.tracker.downsample / portable-port | pending | pending | pending | pending | pending | pending | pending |
| slang_rgbd_slam.tracker.makeNormals / portable-port | pending | pending | pending | pending | pending | pending | pending |
| slang_rgbd_slam.tracker.makeVertices / portable-port | pending | pending | pending | pending | pending | pending | pending |
| slang_rgbd_slam.tracker.resetCount / portable-port | pending | pending | pending | pending | pending | pending | pending |
| slang_rgbd_slam.tracker.updatePose / portable-port | pending | pending | pending | pending | pending | pending | pending |
| slang_rgbd_slam.tsdf.allocateBlocks / portable-port | pending | pending | pending | pending | pending | pending | pending |
| slang_rgbd_slam.tsdf.collectBlocks / portable-port | pending | pending | pending | pending | pending | pending | pending |
| slang_rgbd_slam.tsdf.extractSurface / portable-port | pending | pending | pending | pending | pending | pending | pending |
| slang_rgbd_slam.tsdf.initialize / portable-port | pending | pending | pending | pending | pending | pending | pending |
| slang_rgbd_slam.tsdf.integrateVoxels / portable-port | pending | pending | pending | pending | pending | pending | pending |
| slang_rgbd_slam.tsdf.prepareFusion / portable-port | pending | pending | pending | pending | pending | pending | pending |
| slang_rgbd_slam.tsdf.raycast / portable-port | pending | pending | pending | pending | pending | pending | pending |
| slang_rgbd_slam.tsdf.resetFrame / portable-port | pending | pending | pending | pending | pending | pending | pending |
| slang_rgbd_slam.tsdf.resetSurface / portable-port | pending | pending | pending | pending | pending | pending | pending |
| spatial.voxel_insert / portable-imported | locus-fixture/voxel-contention | stale | stale | pending | stale | AMD RYZEN AI MAX+ PRO 395 w/ Radeon 8060S; AMD Radeon 8060S Graphics (RADV STRIX_HALO); 2026-10-27T20:20:04.561625+00:00 | stale source |
| spatial.voxel_insert / portable-imported | locus-fixture/voxel-capacity-overflow | stale | stale | pending | stale | AMD RYZEN AI MAX+ PRO 395 w/ Radeon 8060S; AMD Radeon 8060S Graphics (RADV STRIX_HALO); 2026-10-27T20:20:04.561625+00:00 | stale source |
| spatial.voxel_insert / portable-imported | locus-fixture/voxel-contention | stale | stale | pending | stale | AMD RYZEN AI MAX+ PRO 395 w/ Radeon 8060S; AMD Radeon 8060S Graphics (RADV STRIX_HALO); 2026-10-27T22:17:46.284752+00:00 | stale source |
| spatial.voxel_insert / portable-imported | locus-fixture/voxel-capacity-overflow | stale | stale | pending | stale | AMD RYZEN AI MAX+ PRO 395 w/ Radeon 8060S; AMD Radeon 8060S Graphics (RADV STRIX_HALO); 2026-10-27T22:17:46.284752+00:00 | stale source |
| vkl.binary_match.finalizeMatch / portable-port | vkl-features/finalizeMatch/chunk-0-129 | stale | stale | pending | stale | AMD RYZEN AI MAX+ PRO 395 w/ Radeon 8060S; AMD Radeon 8060S Graphics (RADV STRIX_HALO); 2026-10-27T22:17:29.471115+00:00 | stale source |
| vkl.binary_match.initializeMatch / portable-port | vkl-features/initializeMatch/chunk-0-129 | stale | stale | pending | stale | AMD RYZEN AI MAX+ PRO 395 w/ Radeon 8060S; AMD Radeon 8060S Graphics (RADV STRIX_HALO); 2026-10-27T22:17:29.471115+00:00 | stale source |
| vkl.binary_match.matchPairs / portable-port | vkl-features/matchPairs/chunk-0-129 | stale | stale | pending | stale | AMD RYZEN AI MAX+ PRO 395 w/ Radeon 8060S; AMD Radeon 8060S Graphics (RADV STRIX_HALO); 2026-10-27T22:17:29.471115+00:00 | stale source |
| vkl.binary_match.matchPairs / portable-port | vkl-features/matchPairs/chunk-258-1 | stale | stale | pending | stale | AMD RYZEN AI MAX+ PRO 395 w/ Radeon 8060S; AMD Radeon 8060S Graphics (RADV STRIX_HALO); 2026-10-27T22:17:29.471115+00:00 | stale source |
| vkl.binary_match.packBinary / portable-port | vkl-features/packBinary/chunk-0-129 | stale | stale | pending | stale | AMD RYZEN AI MAX+ PRO 395 w/ Radeon 8060S; AMD Radeon 8060S Graphics (RADV STRIX_HALO); 2026-10-27T22:17:29.471115+00:00 | stale source |
| vkl.binary_match.reverseMinimum / portable-port | vkl-features/reverseMinimum/chunk-0-129 | stale | stale | pending | stale | AMD RYZEN AI MAX+ PRO 395 w/ Radeon 8060S; AMD Radeon 8060S Graphics (RADV STRIX_HALO); 2026-10-27T22:17:29.471115+00:00 | stale source |
| vkl.binary_match.reverseMinimum / portable-port | vkl-features/reverseMinimum/chunk-258-1 | stale | stale | pending | stale | AMD RYZEN AI MAX+ PRO 395 w/ Radeon 8060S; AMD Radeon 8060S Graphics (RADV STRIX_HALO); 2026-10-27T22:17:29.471115+00:00 | stale source |
| vkl.binary_match.rowMinimum / portable-port | vkl-features/rowMinimum/chunk-0-129 | stale | stale | pending | stale | AMD RYZEN AI MAX+ PRO 395 w/ Radeon 8060S; AMD Radeon 8060S Graphics (RADV STRIX_HALO); 2026-10-27T22:17:29.471115+00:00 | stale source |
| vkl.binary_match.rowMinimum / portable-port | vkl-features/rowMinimum/chunk-258-1 | stale | stale | pending | stale | AMD RYZEN AI MAX+ PRO 395 w/ Radeon 8060S; AMD Radeon 8060S Graphics (RADV STRIX_HALO); 2026-10-27T22:17:29.471115+00:00 | stale source |
| vkl.brief.briefDescribe / portable-port | vkl-features/briefDescribe/chunk-0-0 | stale | stale | pending | stale | AMD RYZEN AI MAX+ PRO 395 w/ Radeon 8060S; AMD Radeon 8060S Graphics (RADV STRIX_HALO); 2026-10-27T22:17:29.471115+00:00 | stale source |
| vkl.depth_convert.depthConvert / portable-port | pending | pending | pending | pending | pending | pending | pending |
| vkl.depth_fusion.fuseDepth / portable-port | pending | pending | pending | pending | pending | pending | pending |
| vkl.depth_fusion.temporalDepthFuse / portable-port | pending | pending | pending | pending | pending | pending | pending |
| vkl.depth_pyramid.depthPyramidLevel / portable-port | pending | pending | pending | pending | pending | pending | pending |
| vkl.depth_resampling.densifyDepth / portable-port | pending | pending | pending | pending | pending | pending | pending |
| vkl.depth_resampling.depthUpsample / portable-port | pending | pending | pending | pending | pending | pending | pending |
| vkl.depth_resampling.initializeDepth / portable-port | pending | pending | pending | pending | pending | pending | pending |
| vkl.depth_resampling.labelDirectDepth / portable-port | pending | pending | pending | pending | pending | pending | pending |
| vkl.geometry.depthGeometry / portable-port | pending | pending | pending | pending | pending | pending | pending |
| vkl.geometry.worldTransform / portable-port | pending | pending | pending | pending | pending | pending | pending |
| vkl.photometric_isp.photometricISP / portable-port | pending | pending | pending | pending | pending | pending | pending |
| vkl.project_depth_evidence.projectDepthEvidence / portable-port | pending | pending | pending | pending | pending | pending | pending |
| vkl.temporal_best.temporalBest / portable-port | pending | pending | pending | pending | pending | pending | pending |
| xsplat_vk.adam.main / portable-port | pending | pending | pending | pending | pending | pending | pending |
| xsplat_vk.backward_probe.backwardProbeMain / portable-imported | pending | pending | pending | pending | pending | pending | pending |
| xsplat_vk.project.projectMain / portable-imported | pending | pending | pending | pending | pending | pending | pending |

## Additional native variants

These executable variants are outside the portable dispatch catalog. They also require CPU/GPU timing baselines; no build or historical upstream time substitutes for execution.

| Variant | CPU reference | GPU kernel | State |
| --- | --- | --- | --- |
| vrdx.upsweep | pending | pending | Optional native Vulkan runtime/standalone CPU-GPU timing recipe pending; C++ build only is not a timing measurement. |
| vrdx.spine | pending | pending | Optional native Vulkan runtime/standalone CPU-GPU timing recipe pending; C++ build only is not a timing measurement. |
| vrdx.downsweep | pending | pending | Optional native Vulkan runtime/standalone CPU-GPU timing recipe pending; C++ build only is not a timing measurement. |
| vrdx.downsweep_key_value | pending | pending | Optional native Vulkan runtime/standalone CPU-GPU timing recipe pending; C++ build only is not a timing measurement. |
