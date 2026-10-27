# CPU and GPU kernel timing reference

Explicit workload/configuration timings, not optimization or application qualification. CPU reference implementations are not necessarily optimized CPU alternatives.

Median; p95 uses linear interpolation at (n-1)*0.95. Raw samples retained.

Coverage: 17 / 130 operations have a source-matched paired CPU/GPU baseline. Pending entries remain part of the required measurement work.

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
| depth.bilateral / portable-imported | locus-fixture/bilateral-invalid | 0.011161 / 0.014750 | 0.003166 / 0.004048 | pending | 0.169209 / 0.230082 | AMD RYZEN AI MAX+ PRO 395 w/ Radeon 8060S; AMD Radeon 8060S Graphics (RADV STRIX_HALO); 2026-10-27T20:20:04.561625+00:00 | current source |
| depth.register / portable-imported | locus-fixture/identity-registration | 0.010270 / 0.010962 | 0.002525 / 0.002831 | pending | 0.155027 / 0.264139 | AMD RYZEN AI MAX+ PRO 395 w/ Radeon 8060S; AMD Radeon 8060S Graphics (RADV STRIX_HALO); 2026-10-27T20:20:04.561625+00:00 | current source |
| depth.register / portable-imported | locus-fixture/nearest-depth-collision | 0.005370 / 0.006081 | 0.002565 / 0.002707 | pending | 0.144839 / 0.305497 | AMD RYZEN AI MAX+ PRO 395 w/ Radeon 8060S; AMD Radeon 8060S Graphics (RADV STRIX_HALO); 2026-10-27T20:20:04.561625+00:00 | current source |
| depth.resolve / portable-imported | locus-fixture/resolve-invalid | 0.004894 / 0.007077 | 0.002284 / 0.002511 | pending | 0.165381 / 0.202455 | AMD RYZEN AI MAX+ PRO 395 w/ Radeon 8060S; AMD Radeon 8060S Graphics (RADV STRIX_HALO); 2026-10-27T20:20:04.561625+00:00 | current source |
| depth.validity / portable-imported | locus-fixture/invalid-depth-nan | 0.004027 / 0.004279 | 0.002405 / 0.002631 | pending | 0.157658 / 0.214394 | AMD RYZEN AI MAX+ PRO 395 w/ Radeon 8060S; AMD Radeon 8060S Graphics (RADV STRIX_HALO); 2026-10-27T20:20:04.561625+00:00 | current source |
| geometry.backproject / portable-imported | locus-fixture/optical-backproject | 0.005170 / 0.007157 | 0.002485 / 0.002631 | pending | 0.161380 / 0.213591 | AMD RYZEN AI MAX+ PRO 395 w/ Radeon 8060S; AMD Radeon 8060S Graphics (RADV STRIX_HALO); 2026-10-27T20:20:04.561625+00:00 | current source |
| geometry.coverage_heatmap / portable-imported | locus-fixture/coverage-pitched | 0.004384 / 0.004744 | 0.002364 / 0.003342 | pending | 0.153153 / 0.185562 | AMD RYZEN AI MAX+ PRO 395 w/ Radeon 8060S; AMD Radeon 8060S Graphics (RADV STRIX_HALO); 2026-10-27T20:20:04.561625+00:00 | current source |
| geometry.coverage_points / portable-imported | locus-fixture/coverage-point-preview | 0.002785 / 0.003536 | 0.002445 / 0.003429 | pending | 0.183406 / 0.222742 | AMD RYZEN AI MAX+ PRO 395 w/ Radeon 8060S; AMD Radeon 8060S Graphics (RADV STRIX_HALO); 2026-10-27T20:20:04.561625+00:00 | current source |
| geometry.transform_points / portable-imported | locus-fixture/transform-validity | 0.005941 / 0.006588 | 0.002445 / 0.002956 | pending | 0.170656 / 0.268091 | AMD RYZEN AI MAX+ PRO 395 w/ Radeon 8060S; AMD Radeon 8060S Graphics (RADV STRIX_HALO); 2026-10-27T20:20:04.561625+00:00 | current source |
| image.crop_normalize / portable-imported | locus-fixture/crop-normalize | 0.008296 / 0.009277 | 0.002485 / 0.003615 | pending | 0.151811 / 0.174180 | AMD RYZEN AI MAX+ PRO 395 w/ Radeon 8060S; AMD Radeon 8060S Graphics (RADV STRIX_HALO); 2026-10-27T20:20:04.561625+00:00 | current source |
| image.nv12_to_rgb / portable-port | pending | pending | pending | pending | pending | pending | pending |
| image.planar_yuv_to_rgb / portable-imported | locus-fixture/yuv-matrix0-range0 | 0.008451 / 0.009926 | 0.002445 / 0.002605 | pending | 0.169590 / 0.297986 | AMD RYZEN AI MAX+ PRO 395 w/ Radeon 8060S; AMD Radeon 8060S Graphics (RADV STRIX_HALO); 2026-10-27T20:20:04.561625+00:00 | current source |
| image.planar_yuv_to_rgb / portable-imported | locus-fixture/yuv-matrix0-range1 | 0.007024 / 0.007553 | 0.002405 / 0.002649 | pending | 0.161124 / 0.247098 | AMD RYZEN AI MAX+ PRO 395 w/ Radeon 8060S; AMD Radeon 8060S Graphics (RADV STRIX_HALO); 2026-10-27T20:20:04.561625+00:00 | current source |
| image.planar_yuv_to_rgb / portable-imported | locus-fixture/yuv-matrix1-range0 | 0.006953 / 0.008290 | 0.002445 / 0.003120 | pending | 0.161569 / 0.212053 | AMD RYZEN AI MAX+ PRO 395 w/ Radeon 8060S; AMD Radeon 8060S Graphics (RADV STRIX_HALO); 2026-10-27T20:20:04.561625+00:00 | current source |
| image.planar_yuv_to_rgb / portable-imported | locus-fixture/yuv-matrix1-range1 | 0.007304 / 0.007817 | 0.002405 / 0.002667 | pending | 0.156470 / 0.202207 | AMD RYZEN AI MAX+ PRO 395 w/ Radeon 8060S; AMD Radeon 8060S Graphics (RADV STRIX_HALO); 2026-10-27T20:20:04.561625+00:00 | current source |
| image.remap_bilinear / portable-port | pending | pending | pending | pending | pending | pending | pending |
| image.resize_bilinear / portable-port | pending | pending | pending | pending | pending | pending | pending |
| image.resize_nearest / portable-imported | locus-fixture/resize-odd-pitched | 0.005240 / 0.005717 | 0.002364 / 0.003006 | pending | 0.157662 / 0.215923 | AMD RYZEN AI MAX+ PRO 395 w/ Radeon 8060S; AMD Radeon 8060S Graphics (RADV STRIX_HALO); 2026-10-27T20:20:04.561625+00:00 | current source |
| image.rgb_to_nv12 / portable-port | pending | pending | pending | pending | pending | pending | pending |
| image.undistort / portable-imported | locus-fixture/identity-undistort | 0.007313 / 0.008188 | 0.002505 / 0.003252 | pending | 0.168848 / 0.248310 | AMD RYZEN AI MAX+ PRO 395 w/ Radeon 8060S; AMD Radeon 8060S Graphics (RADV STRIX_HALO); 2026-10-27T20:20:04.561625+00:00 | current source |
| image.undistort / portable-imported | locus-fixture/distortion-outside | 0.007659 / 0.008456 | 0.002525 / 0.003046 | pending | 0.167440 / 0.248135 | AMD RYZEN AI MAX+ PRO 395 w/ Radeon 8060S; AMD Radeon 8060S Graphics (RADV STRIX_HALO); 2026-10-27T20:20:04.561625+00:00 | current source |
| image.yuyv_remap_fused / portable-port | pending | pending | pending | pending | pending | pending | pending |
| image.yuyv_to_rgb / portable-port | pending | pending | pending | pending | pending | pending | pending |
| perception.cosine_similarity / portable-imported | locus-fixture/similarity-count0 | 0.008611 / 0.011149 | 0.002284 / 0.002587 | pending | 0.155543 / 0.188263 | AMD RYZEN AI MAX+ PRO 395 w/ Radeon 8060S; AMD Radeon 8060S Graphics (RADV STRIX_HALO); 2026-10-27T20:20:04.561625+00:00 | current source |
| perception.cosine_similarity / portable-imported | locus-fixture/similarity-count3 | 0.013485 / 0.014370 | 0.002505 / 0.002671 | pending | 0.154251 / 0.246233 | AMD RYZEN AI MAX+ PRO 395 w/ Radeon 8060S; AMD Radeon 8060S Graphics (RADV STRIX_HALO); 2026-10-27T20:20:04.561625+00:00 | current source |
| perception.cosine_similarity / portable-imported | locus-fixture/similarity-nan-zero | 0.012954 / 0.013552 | 0.002525 / 0.002924 | pending | 0.150784 / 0.185336 | AMD RYZEN AI MAX+ PRO 395 w/ Radeon 8060S; AMD Radeon 8060S Graphics (RADV STRIX_HALO); 2026-10-27T20:20:04.561625+00:00 | current source |
| perception.mask_postprocess / portable-imported | locus-fixture/mask-mode0 | 0.006117 / 0.006708 | 0.002364 / 0.002569 | pending | 0.149527 / 0.197255 | AMD RYZEN AI MAX+ PRO 395 w/ Radeon 8060S; AMD Radeon 8060S Graphics (RADV STRIX_HALO); 2026-10-27T20:20:04.561625+00:00 | current source |
| perception.mask_postprocess / portable-imported | locus-fixture/mask-mode1 | 0.010921 / 0.011450 | 0.003046 / 0.003250 | pending | 0.194351 / 0.423968 | AMD RYZEN AI MAX+ PRO 395 w/ Radeon 8060S; AMD Radeon 8060S Graphics (RADV STRIX_HALO); 2026-10-27T20:20:04.561625+00:00 | current source |
| perception.mask_postprocess / portable-imported | locus-fixture/mask-mode2 | 0.011001 / 0.011267 | 0.003126 / 0.003767 | pending | 0.146105 / 0.233966 | AMD RYZEN AI MAX+ PRO 395 w/ Radeon 8060S; AMD Radeon 8060S Graphics (RADV STRIX_HALO); 2026-10-27T20:20:04.561625+00:00 | current source |
| primitives.compact.compactPrepare / portable-port | pending | pending | pending | pending | pending | pending | pending |
| primitives.compact.compactScatter / portable-port | pending | pending | pending | pending | pending | pending | pending |
| primitives.scan.scanBlocks / portable-port | pending | pending | pending | pending | pending | pending | pending |
| primitives.scan.scanCarry / portable-port | pending | pending | pending | pending | pending | pending | pending |
| primitives.scan.scanExclusive / portable-port | pending | pending | pending | pending | pending | pending | pending |
| primitives.scan_uint.scanBlocks / portable-port | pending | pending | pending | pending | pending | pending | pending |
| primitives.scan_uint.scanCarry / portable-port | pending | pending | pending | pending | pending | pending | pending |
| primitives.scan_uint.scanExclusive / portable-port | pending | pending | pending | pending | pending | pending | pending |
| quality.metrics / portable-imported | locus-fixture/quality-metrics | 0.026981 / 0.027827 | 0.002565 / 0.002805 | pending | 0.160697 / 0.233787 | AMD RYZEN AI MAX+ PRO 395 w/ Radeon 8060S; AMD Radeon 8060S Graphics (RADV STRIX_HALO); 2026-10-27T20:20:04.561625+00:00 | current source |
| quality.metrics / portable-imported | locus-fixture/quality-nan | 0.025909 / 0.032065 | 0.002805 / 0.003014 | pending | 0.155463 / 0.199178 | AMD RYZEN AI MAX+ PRO 395 w/ Radeon 8060S; AMD Radeon 8060S Graphics (RADV STRIX_HALO); 2026-10-27T20:20:04.561625+00:00 | current source |
| quality.reduce / portable-imported | locus-fixture/quality-reduction | 0.005781 / 0.006102 | 0.002485 / 0.002685 | pending | 0.163043 / 0.193784 | AMD RYZEN AI MAX+ PRO 395 w/ Radeon 8060S; AMD Radeon 8060S Graphics (RADV STRIX_HALO); 2026-10-27T20:20:04.561625+00:00 | current source |
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
| spatial.voxel_insert / portable-imported | locus-fixture/voxel-contention | 0.011477 / 0.012216 | 0.002725 / 0.002885 | pending | 0.154251 / 0.205692 | AMD RYZEN AI MAX+ PRO 395 w/ Radeon 8060S; AMD Radeon 8060S Graphics (RADV STRIX_HALO); 2026-10-27T20:20:04.561625+00:00 | current source |
| spatial.voxel_insert / portable-imported | locus-fixture/voxel-capacity-overflow | 0.003737 / 0.004432 | 0.002685 / 0.003414 | pending | 0.151646 / 0.231761 | AMD RYZEN AI MAX+ PRO 395 w/ Radeon 8060S; AMD Radeon 8060S Graphics (RADV STRIX_HALO); 2026-10-27T20:20:04.561625+00:00 | current source |
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
