# Duplicate and overlapping GPU kernels

Consolidate operation ownership and shared algebra with explicitly named numerical/layout/capability profiles. G18 now uses one maintained symbolic factory behind two owned export adapters. Preserve frozen references and alternate contracts; route consumers through qualified profiles only after differential and performance gates.

The original targeted audit was read-only and did not qualify GPU execution. Its historical classifications of **22 semantic groups** are retained below; this document now also records the follow-up G18 consolidation. That follow-up does not establish whole-family consolidation or exhaustive coverage. The original classifications are: **14 same operation different contract**, **5 shared algorithm device helper**, **1 exact duplicate**, **2 related only**. The unit is an audited group, not a kernel/file count. No claim is made that all inventory entries were audited.

## Evidence baseline

Audit input: imported and newly implemented consolidation sources. Source citations below refer to repository-relative paths; donor lineage is pinned by the following frozen revisions. The release manifest binds the final packaged source hashes. Canonical port adaptations are not assumed byte-identical to their donors.

| Donor | Frozen revision | Provenance |
|---|---|---|
| locus | `66f67d38e2e34ceec626dd13857ca5de76deafba` | `provenance/locus.json` |
| caspar-slang | `cab53390aa111cd5526cb30c9aef484e73521186` | `provenance/caspar-slang.json` |
| slang-rgbd-slam | `e35e52b35f692b8337696ea3c9d6721d1cc2bc6b` | `provenance/slang-rgbd-slam.json` |
| vkl | `3384023a311651ea4260cf12ddab1567972f89f3` | `provenance/vkl.json` |
| eligsir-sim | `57a6fc17edfb523f90647dcbccf182db1245a1db` | `provenance/eligsir-sim.json` |
| xsplat-vk | `5c41a47c9e6ec1ad666d06abd239f1c02424624b` | `provenance/xsplat-vk.json` |
| vrdx | `d2f603251b1b007c94ee07dd9c5579b7492af6fe` | `provenance/vrdx.json` |
| primitives | `f90d51e718aa52cbaeaf3f7f74569689026f662e` | `provenance/primitives.json` |

## Intentional copies versus maintained duplication

The structural comparison checked **36 source/frozen shader pairs**, of which **34 are byte-identical** (Locus18, Caspar3, SLAM9, XSplat4). The two adapted XSplat files are `adam.slang` and `composite.slang`. These copies preserve imported evidence; byte identity is not an instruction to delete them. The JSON lists every compared pair.

Generated math and composed standalone shader outputs are derivative build/release artifacts. They must be regenerated from one owner input and hash checked, never independently edited. Repeated generated prefixes across composed modules are not counted as separate maintained implementations.

The historical exact algebra duplicate G18 is now consolidated through one maintained symbolic factory and two owned export adapters; frozen donor reference copies are intentional. “Same operation, different contract” groups call for named profiles and reusable helpers, not forced numerical equivalence.

## Audited semantic groups

### G01: YUV/RGB decode and NV12 encode

**Classification: same operation different contract.** The camera shader decodes pitched byte NV12/YUYV into packed RGB8 and explicitly rounds ties to even; Locus takes three float4 planes containing byte-valued floats and outputs normalized float RGB using rounded decimal coefficients. EliGSiR supports interpolated left/center chroma and P010, with fixed-point fast paths and fp64 guard-band fallbacks targeting zero-LSB parity. Even nearest chroma is not enough to prove FP32 and fp64 outputs identical. Encoding shares 2x2 centered chroma averaging, but float32 versus double input normalization/operation order remain observable.

**Source evidence:** `src/sintergpu/kernels/image/camera_color.slang:21-48`; `src/sintergpu/kernels/image/camera_color.slang:61-77`; `src/sintergpu/kernels/image/camera_color.slang:141-166`; `src/sintergpu/kernels/locus/preprocess/yuv_to_rgb.slang:5-13`; `references/donors/eligsir-sim/src/eligsir_sim/video/colour.py:826-857`; `references/donors/eligsir-sim/src/eligsir_sim/video/_colour_hip_ext.py:87-121`.

**Owner and retained variants:** image/camera_color.slang owns the portable packed-byte profile; retained EliGSiR implementation owns the exact HIP/P010 profile. Factor matrix/range metadata, not the rounding chain. Preserve normalized float Locus and exact HIP routes as named profiles until callers can change layout/exactness deliberately.

**Required before deduplication:** Exhaustive YUV byte/tie cases, both matrices/ranges, left/center/nearest/interpolated chroma, chroma edges, P010, padded pitches, RGB encode corner accumulation order and zero-LSB profile gates.

### G02: Resize, remap and fused decode/remap

**Classification: same operation different contract.** Camera resize uses half-pixel coordinates with -0.5, clamped borders and bilinear RGB8 interpolation. Locus resize maps output centers without -0.5 then truncates to nearest texel. VKL RGB pyramid uses antialiased bilinear at ceil-sized scales; canonical_rescale also normalizes by resized validity weights. Camera fusedYUYVRemap intentionally reproduces decode-to-RGB8 then remap by quantizing each source corner first.

**Source evidence:** `src/sintergpu/kernels/image/camera_color.slang:101-117`; `src/sintergpu/kernels/image/camera_color.slang:171-209`; `src/sintergpu/kernels/locus/preprocess/resize.slang:5-7`; `src/sintergpu/references/vkl/pyramid.py:14-18`; `src/sintergpu/references/vkl/pyramid.py:34-37`; `src/sintergpu/references/vkl/rescale.py:153-182`.

**Owner and retained variants:** image family owns a coordinate/border policy vocabulary and sampling helpers; retain separate nearest, bilinear-RGB8 and antialiased validity-weighted profiles. Fused versus two-pass is intentional pipeline equivalence, not redundant independently maintained math: they already call the same readYUYV/bilinear helpers.

**Required before deduplication:** Impulse/checkerboard, up/downsample, odd/1-pixel sizes, invalid maps, all borders, validity holes, RGB8 quantization ties; fused/separate bit equality before any fusion arithmetic change.

### G03: Brown-Conrady distortion/rectification

**Classification: shared algorithm device helper.** Locus and VKL implement the same forward radial/tangential polynomial. Their complete operations differ: Locus uses same-grid intrinsics and nearest sampling; VKL separates reusable grids/sampling and additionally performs a five-step inverse distortion iteration. Generic camera remap consumes a precomputed grid and does not compute distortion itself.

**Source evidence:** `src/sintergpu/kernels/locus/preprocess/undistort.slang:6-11`; `src/sintergpu/references/vkl/camera_registration.py:107-135`; `src/sintergpu/references/vkl/camera_registration.py:234-267`; `src/sintergpu/kernels/image/camera_color.slang:171-179`.

**Owner and retained variants:** camera geometry should own the forward polynomial helper plus independently named inverse/grid builders; image sampler owns remap. Preserve inverse direction, iteration count, coefficient ordering k1,k2,p1,p2,k3 versus Locus parameter placement, nearest versus bilinear.

**Required before deduplication:** Zero coefficients, radial/tangential fixtures, forward/inverse roundtrip with tolerances, pixel-center/border agreement, warped validity and Torch gradient checks for floating sampling.

### G04: Depth validity, representation conversion and bilateral cleanup

**Classification: same operation different contract.** All use finite positive depth as a basic predicate, but Locus invalid_depth also applies inclusive min/max after scale and outputs (depth,valid) in float4. VKL depthConvert accepts depth/inverse/disparity, explicit uint validity/confidence, returns depth and inverse and leaves metric status to host policy. SLAM makeVertices checks z>0 at shader level and relies on validated upstream input. Bilateral is a 3x3 spatial/range Gaussian weighted average, not a substitute for either conversion or a pyramid discontinuity rule.

**Source evidence:** `src/sintergpu/kernels/locus/common/abi.slang:9-9`; `src/sintergpu/kernels/locus/preprocess/invalid_depth.slang:5-8`; `src/sintergpu/kernels/locus/preprocess/bilateral.slang:5-11`; `src/sintergpu/kernels/vkl/depth_convert.slang:18-25`; `src/sintergpu/references/vkl/depth_convert.py:52-77`; `src/sintergpu/kernels/slang_rgbd_slam/tracker.slang:23-28`.

**Owner and retained variants:** depth family owns finite-positive predicate and explicit conversion/validity profiles; keep bilateral a separate filter. Do not silently add/drop range masks, confidence, metric-status policy or reciprocal/disparity handling.

**Required before deduplication:** NaN/Inf/negative/zero, exact range endpoints, reciprocal overflow, confidence propagation, unit scales, padded rows, holes and bilateral edge/invalid-center cases.

### G05: Depth pyramids

**Classification: same operation different contract.** VKL keeps any supported block valid: uses nearest positive depth unless all four samples are valid and the spread is within eps*32, in which case it averages and propagates mean confidence. Its reference masks missing samples to INF before maximum. SLAM averages the valid members when spread <= configurable depthJump, otherwise returns zero; it assumes exactly 2x input dimensions and has no confidence output. They can differ on partial support containing multiple unequal values, odd-sized border blocks and depth discontinuities. A block with only one valid finite value has the same depth value, but output confidence/layout still differ.

**Source evidence:** `src/sintergpu/kernels/vkl/depth_pyramid.slang:15-35`; `src/sintergpu/references/vkl/depth_pyramid.py:13-33`; `src/sintergpu/kernels/slang_rgbd_slam/tracker.slang:10-19`.

**Owner and retained variants:** depth/pyramid owns two documented policies: validity-preserving nearest-surface and tracking average-or-reject. Preserve ceil versus divisible sizing, sparse support, confidence winner selection, absolute epsilon versus application metre threshold.

**Required before deduplication:** All 16 valid masks, odd sizes, ties for nearest confidence, near-equal values around eps*32, jump threshold endpoints, confidence and no-support outputs.

### G06: Grayscale, quality luminance and image pyramids

**Classification: same operation different contract.** SLAM feature gray uses .299/.587/.114 in byte-scale values and its next level uses separable 5x5 [1,4,6,4,1]/256. Dense tracking uses the same gray coefficients divided by255, and a 2x2 mean with negative invalid sentinel. VKL/Locus quality luminance uses .2126/.7152/.0722; VKL resolution pyramid is antialiased bilinear RGB. The gray dot product is shareable only with explicit weights/scale; pyramid filters are different.

**Source evidence:** `src/sintergpu/kernels/slang_rgbd_slam/features.slang:25-38`; `src/sintergpu/kernels/slang_rgbd_slam/dense_tracker.slang:26-40`; `src/sintergpu/kernels/locus/quality/metrics.slang:8-14`; `src/sintergpu/references/vkl/dense_track_reduce.py:34-35`; `src/sintergpu/references/vkl/pyramid.py:34-37`.

**Owner and retained variants:** image family owns configurable luminance helper; retain named pyramid filters and intensity validity conventions. Preserve brightness scale because FAST/photometric thresholds are calibrated against it; no replacement of Gaussian with box/bilinear.

**Required before deduplication:** Color primaries, constant gray, impulse frequency response, boundary clamps, negative sentinel propagation, odd dimensions and unchanged detector/tracker outputs.

### G07: Pinhole backprojection

**Classification: shared algorithm device helper.** The same algebra ((u-cx)*z/fx,(v-cy)*z/fy,z) appears in generated Caspar scalar math, Locus and VKL. Dispatch contracts differ: Caspar scalar helper is unguarded; Locus applies depth unit/calibration and finite xyz checks with float4 validity; VKL supports batched planar Torch geometry and masks invalid depth. SLAM already calls caspar_backproject instead of defining the algebra again.

**Source evidence:** `references/donors/caspar-slang/src/caspar_slang/primitives.py:79-81`; `src/sintergpu/kernels/locus/geometry/backproject.slang:5-9`; `src/sintergpu/references/vkl/geometry.py:39-55`; `src/sintergpu/kernels/slang_rgbd_slam/tracker.slang:23-28`.

**Owner and retained variants:** generated Caspar geometry scalar helper owns the common algebra; keep caller-specific validation/layout wrappers and independent CPU reference. Changing division/multiplication order may change FP32 rounding; unit scale, finite checks and optical conventions remain wrapper contracts.

**Required before deduplication:** Same calibrated pixels/depth in all layouts, non-unit focal length, invalid/huge inputs, unit conversion, numeric error/bit profile, and gradients before replacing Torch routes.

### G08: Rigid point transforms

**Classification: shared algorithm device helper.** All compute Rp+t, but Locus returns zero if input validity is absent or transformed xyz is nonfinite; VKL world_transform always applies translation even to a zero coordinate and its depth_geometry wrapper remasks afterward. Caspar scalar functions take explicit row-major coefficients; XSplat uses matrix multiplication on Gaussian means, then projects covariance too.

**Source evidence:** `src/sintergpu/kernels/locus/common/abi.slang:14-14`; `src/sintergpu/kernels/locus/geometry/transform_points.slang:5-6`; `src/sintergpu/references/vkl/geometry.py:81-102`; `references/donors/caspar-slang/src/caspar_slang/primitives.py:84-104`; `src/sintergpu/kernels/xsplat_vk/project.slang:15-32`.

**Owner and retained variants:** geometry scalar helper for Rp+t with explicit row-major contract; validity remains in dispatch wrappers. Do not translate invalid origin points into valid positions; maintain c2w/w2c direction and scalar/matrix memory layout.

**Required before deduplication:** Nonzero translation with invalid points, random rigid transforms, row/column orientation, pose inverses, nonfinite output and differential/gradient parity.

### G09: Depth normal maps

**Classification: same operation different contract.** VKL uses forward (right/down) differences, a relative 25% depth-jump gate, eps1e-8 normalization, and flips normals to face the camera. SLAM uses central neighbors, an absolute configurable depthJump, length>1e-10, and no camera-facing flip. For a fronto-parallel optical-camera plane the formulas therefore imply opposite signs. Boundary invalid regions also differ (right/bottom only versus all four image edges).

**Source evidence:** `src/sintergpu/references/vkl/geometry.py:17-21`; `src/sintergpu/references/vkl/geometry.py:54-67`; `src/sintergpu/kernels/slang_rgbd_slam/tracker.slang:32-42`.

**Owner and retained variants:** geometry/normal_map should expose explicit stencil/orientation/discontinuity profiles, preserving current consumers. These are not interchangeable merely because ICP can sometimes absorb a sign flip; orientation and support masks are observable.

**Required before deduplication:** Plane sign/orientation, all boundaries, sparse holes, depth-dependent discontinuities, degeneracy, confidence and downstream correspondence/pose parity.

### G10: Depth registration and z-buffer reprojection

**Classification: same operation different contract.** Locus projects undistorted positive z with floor(uv+.5), accepts half-pixel edge bounds, atomic-mins positive float bits, and requires INF state initialization. VKL registration supports distortion and depth ranges, uses torch round ties-to-even, and records deterministic smallest-source-index winners for equal z. VKL rasterize_reprojection additionally aggregates confidence/support over epsilon-close depth winners. Shared nearest-z idea does not establish matching pixels or metadata.

**Source evidence:** `src/sintergpu/kernels/locus/preprocess/depth_register.slang:5-12`; `src/sintergpu/references/vkl/camera_registration.py:433-452`; `src/sintergpu/references/vkl/camera_registration.py:465-517`; `src/sintergpu/references/vkl/reproject.py:85-106`.

**Owner and retained variants:** geometry/depth_register owns projection and z-buffer helpers with explicit rounding, support and winner profiles. Preserve ties, half-pixel support, distortion, source-index provenance and confidence; reset/resolve is a caller-visible multipass contract.

**Required before deduplication:** Half-integer projections including negative edge, equal-depth collisions/source indices, epsilon-close winners, distortion, ranges, empty state, sparse confidence/support and two-pass barriers.

### G11: Workgroup reductions and quality accumulators

**Classification: same operation different contract.** SinterGPU block_reduce is a fixed256 stride-halving tree over channel-major caller storage. Caspar GN uses64 lanes,43 terms and AoSoA Jacobian access; subgroup variants do WaveActiveSum before the tree. Dense tracker changes term construction and uses32/64/128/256 specializations. Locus quality uses uint atomic sums of clamped fixed-point metrics, not floating normal-equation sums. VKL solve uses adjacent-pair left-to-right reduction, a different addition order from stride-halving.

**Source evidence:** `src/sintergpu/kernels/common/block_reduce.slang:2-17`; `src/sintergpu/kernels/caspar_slang/gn.slang:8-38`; `src/sintergpu/kernels/slang_rgbd_slam/dense_tracker.slang:68-87`; `src/sintergpu/kernels/locus/quality/reduce.slang:7-10`; `src/sintergpu/references/vkl/tracking_solve.py:10-16`.

**Owner and retained variants:** primitives owns parameterized reduction implementation only when lane count, layout, identity, wave policy and exact addition tree are preserved. Retain numerical variants; changing trees or introducing waves changes rounding, and fixed-point counts have separate saturation/overflow semantics.

**Required before deduplication:** Tail lengths around every block size, zero count, NaN policy, cancellation/high dynamic range, wave32/64, integer overflow limits and unchanged optimizer/status/pose gates.

### G12: Segmented scan versus radix-sort internal prefix sums

**Classification: shared algorithm device helper.** Public scan supports contiguous nondecreasing segment IDs, sum/product, float/uint and hierarchical carries; exclusive output is a separate pass. VRDX prefix sums are integer histogram offsets embedded in radix passes with512-lane workgroups and radix256, including full-workgroup barrier requirements. They share prefix-sum mechanics but have different storage, launch and stability obligations.

**Source evidence:** `src/sintergpu/kernels/primitives/scan.slang:6-25`; `src/sintergpu/kernels/primitives/scan.slang:27-66`; `third_party/vrdx/src/shader/spine.slang:25-59`; `third_party/vrdx/src/shader/spine.slang:62-90`.

**Owner and retained variants:** primitives/scan owns public generic scan; keep VRDX vendor implementation independently versioned/patched. Do not replace vendor prefix stages merely to remove repeated sums; sorting stability and synchronization are part of the full algorithm.

**Required before deduplication:** Segment boundaries spanning blocks, repeated versus monotone IDs rejection, sum/product identities/zeros, exclusive behavior, uint wrap policy and full stable radix-sort parity under device wave sizes.

### G13: Stable compaction versus feature append/selection

**Classification: same operation different contract.** Public compact uses inclusive uint prefixes to scatter payloads stably and reports total selection count; output capacity is caller-owned. VKL select_compact is batched, arbitrary payload dtype/shape, bounded capacity, zero-filled unused slots and returns source indices/action codes; its count is truncated retained count. SLAM cell selection uses an atomic reservation per cell and top-K tie-breaking, so cross-cell append order is not a stable original-input ordering guarantee.

**Source evidence:** `src/sintergpu/kernels/primitives/compact.slang:2-25`; `src/sintergpu/references/vkl/compact.py:17-36`; `src/sintergpu/kernels/slang_rgbd_slam/features.slang:99-119`.

**Owner and retained variants:** primitives/compact owns stable flags/prefix/scatter; add a bounded batched wrapper for VKL instead of changing the count contract. Preserve capacity truncation, total-versus-retained count, zero-fill and ordering; feature detection selection remains a distinct producer policy.

**Required before deduplication:** Zero/all/none selected, stable indices, capacities0/less-than-selected, arbitrary payload fields, batch boundaries, poisoned padding and feature-order independence/descriptor association.

### G14: FAST detection and spatial keypoint selection

**Classification: same operation different contract.** Both evaluate the same16-point radius3 ring/FAST9 predicate. SLAM score is maximum over arc minima with byte-scale threshold, border19; NMS resolves equal scores by smaller linear index, then Harris scoring and bounded cell quota with depth gating. VKL supports arc1..16, normalized threshold, scores maximum ring contrast after detection, keeps equal local-max plateaus, and uses bounded tensor top-K/margin/domain erosion.

**Source evidence:** `src/sintergpu/kernels/slang_rgbd_slam/features.slang:22-23`; `src/sintergpu/kernels/slang_rgbd_slam/features.slang:45-73`; `src/sintergpu/kernels/slang_rgbd_slam/features.slang:81-119`; `src/sintergpu/references/vkl/keypoint_selectors.py:77-79`; `src/sintergpu/references/vkl/keypoint_selectors.py:175-200`; `src/sintergpu/references/vkl/keypoint_selectors.py:103-125`.

**Owner and retained variants:** features owns shared ring predicate, while detector score/NMS/Harris/quota are explicit profiles. Same FAST name does not mean identical candidate ranking or selected features. Preserve margin, tie-breaks, depth support and normalized/byte thresholds.

**Required before deduplication:** Threshold ties, all16 arc starts, plateaus, image borders, depth holes, per-cell quotas and exact selected coordinates/ranking under each profile.

### G15: BRIEF descriptors

**Classification: same operation different contract.** SLAM rotates an original256-pair Gaussian-disk pattern by intensity-centroid orientation and packs eight uint32 words; samples clamp to image bounds. VKL default is an affine enumeration of distinct ordered pairs over31x31, no orientation, arbitrary bit count and unpacked bool output; descriptor validity requires every comparison sample to be in-bounds/finite/valid. Packing conversion alone cannot make these descriptors comparable.

**Source evidence:** `references/donors/slang-rgbd-slam/src/slang_rgbd_slam/features.py:1-21`; `src/sintergpu/kernels/slang_rgbd_slam/features.slang:123-138`; `src/sintergpu/kernels/slang_rgbd_slam/features.slang:23-23`; `src/sintergpu/references/vkl/brief.py:15-50`; `src/sintergpu/references/vkl/brief.py:87-125`.

**Owner and retained variants:** features/brief owns explicit pattern/version/orientation/border/packing contracts; retain separate original patterns. Do not call either OpenCV-ORB-compatible or compare Hamming distances between different patterns.

**Required before deduplication:** Pattern hash/version, fixed image/keypoints, orientation rotations, exact bit order/packing, equality comparisons, invalid sample and border masks, arbitrary bit counts.

VKL now has a separate float32/int64 Slang port and exact frozen-donor CPU/GPU probes; see [binary feature contracts](vkl-binary-features.md). The original SLAM pattern remains unchanged.

### G16: Binary matching

**Classification: same operation different contract.** SLAM globally matches fixed256-bit packed descriptors with absolute Hamming bound, strict best/second ratio and mutual nearest-neighbor; it requires at least two target descriptors. VKL uses boolean arbitrary-bit descriptors and geometric radius gating, optional mutual ownership and no ratio criterion; chunking is intended to preserve exact results. Both keep first index on equal best distance through strict improvement, but candidate domains differ.

**Source evidence:** `src/sintergpu/kernels/slang_rgbd_slam/features.slang:145-183`; `src/sintergpu/references/vkl/local_match.py:25-30`; `src/sintergpu/references/vkl/local_match.py:110-145`.

**Owner and retained variants:** features owns packed Hamming distance/tiled search helper and separately named global-ratio/local-radius matching policies. Keep ratio, mutual, spatial gates, descriptor format and missing-match sentinel semantics.

**Required before deduplication:** Empty/single targets, ties/best-second equality, radius boundary, ratio boundary, max Hamming endpoints, mutual collisions, packing and chunk-size invariance.

The VKL port retains radius gating, first-index ties, optional mutual ownership and bounded candidate chunking. Its seven dispatch entries include BRIEF plus six matching stages; these are retained variants, not duplicate aliases of SLAM.

### G17: ICP association and fused normal-equation assembly

**Classification: same operation different contract.** SLAM unfused association uses projective nearest match, distance+normal gates and square-root Huber weights before Caspar GN. Fused tracking computes the same geometric residual/Jacobian directly and optionally adds photometric terms, avoiding materialized residual tensors; it is an optimization variant of that pipeline. VKL dense reduction shares point-to-plane algebra but orders translation before rotation; Caspar/SLAM order rotation before translation. VKL camera-local c2w right update also differs from SLAM left action on reference_from_current. Sparse VKL uses point-to-point3D residuals and explicitly adjusts sign to its pose convention.

**Source evidence:** `src/sintergpu/kernels/slang_rgbd_slam/tracker.slang:70-115`; `src/sintergpu/kernels/slang_rgbd_slam/dense_tracker.slang:43-87`; `src/sintergpu/references/vkl/dense_track_reduce.py:38-54`; `src/sintergpu/references/vkl/tracking_reference.py:56-79`; `src/sintergpu/references/vkl/sparse_pose_reduce.py:22-26`; `src/sintergpu/references/vkl/sparse_pose_reduce.py:107-113`; `references/donors/caspar-slang/src/caspar_slang/gn.py:53-59`.

**Owner and retained variants:** optimization owns explicit Jacobian/tangent/frame convention metadata and common term helper; retain fused/unfused profiles. Do not directly feed VKL H,b/delta into Caspar solver/update without permutation and pose-frame/sign transforms; retain application acceptance/rollback policies.

**Required before deduplication:** Independent finite differences, tangent permutation/adjoint checks, asymmetric poses, correspondence masks/counts, H,b,cost and final pose, robust thresholds, fused/no-photo parity, near singular cases.

### G18: Duplicate point-to-plane symbolic definitions

**Classification: exact duplicate.** Caspar gn.residual_kernel and reprojection.factor_kernels.point_plane independently define the same r=n dot(p+omega cross p+translation-q), differentiated at zero, with rotation-first6-vector. One returns a residual+V6 for AoSoA dispatch, the other seven scalars for inline fusion. This is exact duplicated symbolic algebra, not identical dispatch ABI. Those donor definitions remain frozen independent references. The maintained vector/AoSoA and scalar adapters in `tools/codegen/caspar_point_plane.py` now consume one `sintergpu.codegen.point_plane` factory. All five existing generated modules remain byte-identical; generated emitted/composed copies are a separate provenance concern.

**Source evidence:** `references/donors/caspar-slang/src/caspar_slang/gn.py:43-59`; `references/donors/caspar-slang/src/caspar_slang/reprojection.py:85-94`; `references/donors/caspar-slang/src/caspar_slang/caspar_adapter.py:22-34`.

**Owner and retained variants:** `sintergpu.codegen.point_plane` owns the symbolic residual/Jacobian expression; the two owned Caspar export adapters preserve AoSoA and scalar schedules. Preserve both ABI exports and independent NumPy analytic/finite-difference oracle; deleting that oracle would weaken validation.

**Required before deduplication:** Compare emitted residual and all6 Jacobian components for randomized/tail/indexed/duplicate-gather cases, exact convention, generated source/compiler drift and existing downstream fused/unfused tolerances.

### G19: TSDF fusion, occupancy counts and depth fusion

**Classification: related only.** SLAM integrates truncated normalized signed distance in sparse8x8x8 blocks, uses depth-dependent observation mass and separately accumulated surface color. Locus voxel_insert only hashes quantized cells and increments observation counts; it has no signed-distance surface model. VKL depth_fuse combines same-grid depth with consistency/confidence and provenance statuses; temporal fusion adds cross-frame evidence/cluster support. A weighted average appears in each but the represented quantities and rejection/retention policies differ.

**Source evidence:** `src/sintergpu/kernels/slang_rgbd_slam/tsdf.slang:113-130`; `references/donors/caspar-slang/src/caspar_slang/primitives.py:111-120`; `src/sintergpu/kernels/locus/geometry/voxel_insert.slang:5-14`; `src/sintergpu/references/vkl/depth_fuse.py:43-65`; `src/sintergpu/references/vkl/temporal_depth_fuse.py:93-129`.

**Owner and retained variants:** Keep mapping/tsdf, mapping/voxel_counts and depth/fusion separate; common tsdf_fuse scalar helper already serves distance and color. Preserve actual mass before confidence capping, truncation sign, color-weight isolation, coordinate/probe/capacity limits and overflow reporting.

**Required before deduplication:** Repeated capped-weight observations, plane/slanted surfaces, free-space without color, map bounds/overflow/probe exhaustion, conflicting temporal depths and provenance/confidence selection.

### G20: Point projection, sensor evidence, reprojection factors and Gaussian projection

**Classification: related only.** Pinhole pixel formulas recur but outputs differ. Caspar reprojection produces2D residuals and pose/landmark Jacobians; VKL projectDepthEvidence performs eligibility, ties-even pixel lookup and sensor support/conflict decisions; XSplat projects mean plus3D covariance to2D conic with dilation and differentiable covariance functions. They cannot substitute for each other. Shared scalar pinhole/rigid math must retain caller z guards, matrix conventions and derivative routes.

**Source evidence:** `references/donors/caspar-slang/src/caspar_slang/reprojection.py:204-212`; `src/sintergpu/kernels/caspar_slang/reprojection.slang:28-61`; `src/sintergpu/kernels/vkl/project_depth_evidence.slang:42-70`; `src/sintergpu/kernels/xsplat_vk/project.slang:15-33`; `src/sintergpu/kernels/xsplat_vk/gaussian_math.slang:9-50`.

**Owner and retained variants:** geometry scalar helpers own projection math; keep evidence, optimization residual and Gaussian renderer operation families separate. Preserve covariance/conic gradients, sensor conflict asymmetry, z epsilon and eligible/projected/sensor-valid distinctions.

**Required before deduplication:** Finite-difference pose/landmark and Gaussian VJP, covariance rotation/conic dilation, half-pixel evidence ties, behind-camera/huge projections, support/conflict threshold strictness.

### G21: Cholesky solves, Schur systems, pose graph PCG and Adam

**Classification: shared algorithm device helper.** Caspar6x6 and SLAM Schur up-to18-DOF use the same left-looking damped Cholesky loop pattern and zero invalid updates, but dimensions/storage stride and observation-count guards differ. VKL6x6 additionally symmetrizes H and rejects by undamped eigenvalue/condition_max before accepting a damped solve. Pose graph is matrix-free PCG with anchor node and cost rollback, while XSplat Adam updates first/second moments with bias correction; those are related optimization tasks, not alternative implementations of one solver.

**Source evidence:** `src/sintergpu/kernels/caspar_slang/gn.slang:69-103`; `src/sintergpu/kernels/slang_rgbd_slam/dense_ba.slang:135-155`; `src/sintergpu/references/vkl/tracking_solve.py:53-73`; `src/sintergpu/kernels/caspar_slang/pose_graph.slang:120-169`; `src/sintergpu/kernels/xsplat_vk/adam.slang:29-36`.

**Owner and retained variants:** optimization owns a parameterized small-SPD Cholesky helper preserving exact arithmetic loop order; health policy, Schur construction, PCG and Adam stay distinct. A damped invertible system may still fail VKL health policy. Do not make success statuses equivalent or replace PCG/Adam with Cholesky.

**Required before deduplication:** Dimension6/18, singular/indefinite/ill-conditioned inputs, damping boundaries, nonfinite propagation, exact zero failure update, condition gates, rollback, PCG anchor and Adam step/bias-correction sequences.

### G22: Photometric ISP and normalization

**Classification: same operation different contract.** VKL Slang ISP matches the forward Torch operation order: exposure,gain,vignette,color transform, nonnegative power-response floor and clamp. Slang explicitly supplies epsilon and flattened parameters and omits image-wide diagnostics/backward. Torch has a deliberate finite gradient at black and optional evidence outputs. Locus normalize is crop+nearest sampling+per-channel mean/std, so it is related preprocessing, not a replacement ISP.

**Source evidence:** `src/sintergpu/kernels/vkl/photometric_isp.slang:27-56`; `src/sintergpu/references/vkl/photometric_isp.py:94-125`; `src/sintergpu/references/vkl/photometric_isp.py:132-151`; `src/sintergpu/kernels/locus/semantics/normalize.slang:6-10`.

**Owner and retained variants:** image/photometric_isp owns forward policy, with Torch oracle and explicit backward/evidence capabilities; tensor/normalize remains separate. Preserve dtype-dependent epsilon, singleton linspace=-1, gradient path, diagnostic availability and crop sampler semantics.

**Required before deduplication:** All optional transforms/order, black with gamma<1, saturation,1x1 dimensions, per-batch/channel metadata, forward errors, Torch gradients and evidence results before any route replacement.

## D4/T0 reuse follow-up

The new experimental reprojection family reuses `vkl.geometry.depthGeometry` for camera XYZ/validity and `vkl.depth_fusion.temporalDepthFuse` for prior-depth clustering. Projection and temporal RGB warping share the same normalized-coordinate float32 sampling functions in `vkl/reprojection.slang`; no separate copies are maintained per public operation. This is shared implementation within a named profile, not another historical audit group.

G10 still keeps Locus registration separate: half-pixel bounds, rounding, initialization and confidence/support contracts differ. G20 still keeps sensor evidence and optimization/Gaussian projections separate. Direct pixel rounding cannot replace D4's normalized roundtrip (width4 pixel0.5 becomes0.50000006), and this planar float sampler cannot replace quantized YUYV decode/remap. Prior-only reference depth, unmasked RGB/diagnostics and epsilon-close raster support are pinned by independent CPU and frozen-donor tests. Both standalone compiler ABIs pass and each target vendor passes63cases/113states against the canonical donor profile; isolated-stage fixture timings are recorded; controlled no-regression and replacement/default selection remain pending.

## Recommended order and rejected alternatives

1. G18 is centralized through one maintained point-plane symbolic factory; both scalar and AoSoA exports and the independent CPU oracle remain. The other historical audit groups retain their stated qualification gates.
2. Establish shared camera/geometry helpers and named profiles for matrix/range, pixel coordinates, invalid data, normal orientation and tangent conventions.
3. Parameterize reduction/Cholesky helpers only where exact tree/loop order, layout, status and device assumptions survive. Keep VRDX as a pinned vendor unit.
4. Preserve fused variants and all independent references; compare transfer-inclusive and resident performance before switching downstream defaults. Announce profile/compatibility changes through versioned release notes.

- **Rejected: Delete/replace every same-named kernel with one implementation.** Normal orientation, tangent order, rounding, confidence, capacity and gradient contracts observably differ.
- **Rejected: Keep all imported kernels as independent permanent APIs with no shared ownership.** Leaves real duplicate algebra and untracked profile drift; central ownership/generation is needed.
- **Rejected: Treat snapshot and generated byte repetition as redundant kernels to delete.** Erases provenance/independent oracles or compile-ready composition; generation/hash checks address maintenance without deleting evidence.

## Review limits

- No test or benchmark was run for this semantic review; proposed dedup gates remain outstanding.
- VKL Torch functions are references and potential autograd routes, not proof that a corresponding canonical Slang GPU implementation exists for every operation.
- No claim that all inventory source files or all249 entries were audited.
- Exact_duplicate means exact symbolic residual/Jacobian algebra in G18, not identical resource/dispatch ABI.

## Geometry/depth port follow-up

The four forward profiles added for 0.1.2 retain these distinctions from the
original review; they do not collapse similarly named operations into aliases.

| New profile | Existing overlap | Disposition |
| --- | --- | --- |
| `vkl.geometry.depthGeometry` | Locus backprojection and SLAM normal generation | Same task with different layout, invalid-depth masking, right/down stencil and camera-facing normal contracts; retain a named profile |
| `vkl.geometry.worldTransform` | Locus point transformation and depthGeometry world output | Standalone transform always applies Rp+t, while depthGeometry remasks invalid depth; retain separate entry points |
| `vkl.depth_fusion.fuseDepth` | G19 depth/TSDF/occupancy fusion | Same-grid confidence-weighted depth and first-max provenance; distinct from signed-distance integration and voxel counts |
| `vkl.depth_fusion.temporalDepthFuse` | G19 temporal evidence | Asymmetric pairwise clusters, explicit trust and eleven forward outputs; retain specialized profile |

See [the numerical contracts](vkl-geometry-depth.md) and
[hash-bound lineage](../provenance/vkl-geometry-depth.json). New differential
tests compare all forward outputs with the frozen VKL implementations, including
logical/provenance values and invalid outputs. A borrowed-encoder fusion→geometry
test checks composition. These checks do not qualify cross-vendor behavior,
gradients, application results or performance, and do not replace the original
22-group review with a claim of an exhaustive audit.

Temporal best-observation selection is a further related-only temporal task: it
reduces equivalent observations by quality, with first-NaN/first-maximum semantics
and target fallback. It does not average depths, determine pairwise depth clusters
or integrate a TSDF. Keep `vkl.temporal_best.temporalBest` separate from the G19
fusion family; see [its contract](vkl-temporal-best.md) and
[hash-bound lineage](../provenance/vkl-temporal-best.json). Shared row-reduction
helpers would need to preserve these ordering and NaN rules before deduplication.

The [depth resampling profiles](vkl-depth-resampling.md) reuse the exact released
pyramid-level shader and its existing independent NumPy oracle. New composition
adds initialization and provenance labels; it does not duplicate the level
implementation. Nearest depth lifting shares a task name with image resize but
uses finite-positive masking and propagated labels, so it cannot substitute for
Locus float4 image nearest sampling or bilinear remap. RGB-guided densification
is a lower-median hole fill, distinct from bilateral filtering, same-grid weighted
fusion and temporal clustering; its initial direct mask persists across passes.
