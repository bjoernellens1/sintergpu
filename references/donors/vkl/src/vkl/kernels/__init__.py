from .reference.brief import brief_describe
from .reference.camera_registration import (
    camera_rectify,
    camera_rectify_from_grid,
    camera_rectify_grid,
    depth_camera_rays,
    depth_register,
)
from .reference.compact import select_compact
from .reference.contract import validate_frame_batch
from .reference.dense_depth_regularize import regularize_dense_depth
from .reference.dense_pose_depth import dense_pose_depth
from .reference.dense_track_reduce import dense_track_reduce
from .reference.depth_align import align_depth
from .reference.depth_convert import depth_convert
from .reference.depth_cost_volume import aggregate_depth_cost_volume
from .reference.depth_densify import depth_densify, depth_downsample, depth_upsample
from .reference.depth_evidence import depth_evidence
from .reference.depth_fuse import fuse_depth
from .reference.depth_pyramid import depth_pyramid
from .reference.direct_pose_depth import (
    direct_pose_depth_dso,
    direct_pose_depth_tensorized,
    dso_select_points,
)
from .reference.evidence import rgbd_evidence
from .reference.geometry import depth_geometry, world_transform
from .reference.keyframe_filter import keyframe_filter
from .reference.keypoint_evidence import keypoint_evidence
from .reference.keypoint_selectors import (
    fast_select,
    pose_observable_select,
    shi_tomasi_select,
    spatial_select_points,
    uniform_select,
)
from .reference.landmark_update import landmark_update
from .reference.landmark_validate import landmark_multiview_validate
from .reference.lk import lk_track
from .reference.local_match import binary_local_match
from .reference.mesh import (
    fit_local_planes,
    voxel_surface_mesh,
    voxelize_points,
    world_voxel_reduce,
)
from .reference.photometric_isp import photometric_isp
from .reference.point_render import point_splat_render
from .reference.project_depth_evidence import project_depth_evidence
from .reference.pyramid import resolution_pyramid
from .reference.reproject import depth_reproject, rasterize_reprojection, reproject_rgbd
from .reference.rescale import canonical_rescale
from .reference.residual import residual_evidence
from .reference.rigid_alignment import rigid_align_3d
from .reference.sparse_geometry import (
    project_landmarks,
    sparse_depth_validate,
    sparse_rgbd_observation,
)
from .reference.sparse_pose_reduce import sparse_pose_reduce
from .reference.spatial import spatial_bin_reduce
from .reference.temporal import temporal_restore_rgb, temporal_restore_rgbd
from .reference.temporal_best import temporal_best_observation
from .reference.temporal_compare import temporal_frame_compare
from .reference.temporal_depth_fuse import temporal_depth_fuse
from .reference.temporal_reproject import temporal_reproject_rgb, temporal_rgbd_reference
from .reference.tracking_health import (
    TRACK_REASON_CONDITION,
    TRACK_REASON_GEOMETRY,
    TRACK_REASON_PARTIALS,
    TRACK_REASON_PHOTOMETRIC,
    TRACK_REASON_RESIDUAL,
    TRACK_REASON_SOLVE,
    tracking_health,
)
from .reference.tracking_prepare import tracking_prepare
from .reference.tracking_reference import (
    apply_camera_local_tracking_update,
    select_tracking_reference,
)
from .reference.tracking_solve import deterministic_reduce_solve
from .reference.tracking_transaction import compose_tracking_solve
from .reference.triangulate import triangulate_depth
from .registry import REGISTRY, register
from .triton import select_best_inverse_depth

REGISTRY.setdefault("canonical_rescale", canonical_rescale)
REGISTRY.setdefault("select_compact", select_compact)
REGISTRY.setdefault("spatial_bin_reduce", spatial_bin_reduce)
REGISTRY.setdefault("residual_evidence", residual_evidence)
REGISTRY.setdefault("rigid_align_3d", rigid_align_3d)
REGISTRY.setdefault("depth_evidence", depth_evidence)
REGISTRY.setdefault("depth_convert", depth_convert)
REGISTRY.setdefault("dense_track_reduce", dense_track_reduce)
REGISTRY.setdefault("sparse_pose_reduce", sparse_pose_reduce)
REGISTRY.setdefault("deterministic_reduce_solve", deterministic_reduce_solve)
REGISTRY.setdefault("compose_tracking_solve", compose_tracking_solve)
REGISTRY.setdefault("dense_pose_depth", dense_pose_depth)
REGISTRY.setdefault("binary_local_match", binary_local_match)
REGISTRY.setdefault("landmark_multiview_validate", landmark_multiview_validate)
REGISTRY.setdefault("landmark_update", landmark_update)
REGISTRY.setdefault("lk_track", lk_track)
REGISTRY.setdefault("voxelize_points", voxelize_points)
REGISTRY.setdefault("world_voxel_reduce", world_voxel_reduce)
REGISTRY.setdefault("fit_local_planes", fit_local_planes)
REGISTRY.setdefault("voxel_surface_mesh", voxel_surface_mesh)
REGISTRY.setdefault("point_splat_render", point_splat_render)
REGISTRY.setdefault("brief_describe", brief_describe)
REGISTRY.setdefault("sparse_depth_validate", sparse_depth_validate)
REGISTRY.setdefault("sparse_rgbd_observation", sparse_rgbd_observation)
REGISTRY.setdefault("project_landmarks", project_landmarks)
REGISTRY.setdefault("aggregate_depth_cost_volume", aggregate_depth_cost_volume)
REGISTRY.setdefault("regularize_dense_depth", regularize_dense_depth)
REGISTRY.setdefault("align_depth", align_depth)
REGISTRY.setdefault("fuse_depth", fuse_depth)
REGISTRY.setdefault("temporal_depth_fuse", temporal_depth_fuse)
REGISTRY.setdefault("temporal_restore_rgbd", temporal_restore_rgbd)
REGISTRY.setdefault("temporal_restore_rgb", temporal_restore_rgb)
REGISTRY.setdefault("temporal_reproject_rgb", temporal_reproject_rgb)
REGISTRY.setdefault("temporal_rgbd_reference", temporal_rgbd_reference)
REGISTRY.setdefault("temporal_frame_compare", temporal_frame_compare)
REGISTRY.setdefault("temporal_best_observation", temporal_best_observation)
REGISTRY.setdefault("depth_pyramid", depth_pyramid)
REGISTRY.setdefault("depth_downsample", depth_downsample)
REGISTRY.setdefault("depth_upsample", depth_upsample)
REGISTRY.setdefault("depth_densify", depth_densify)
REGISTRY.setdefault("direct_pose_depth_dso", direct_pose_depth_dso)
REGISTRY.setdefault("direct_pose_depth_tensorized", direct_pose_depth_tensorized)
REGISTRY.setdefault("dso_select_points", dso_select_points)
REGISTRY.setdefault("depth_geometry", depth_geometry)
REGISTRY.setdefault("depth_reproject", depth_reproject)
REGISTRY.setdefault("rgbd_evidence", rgbd_evidence)
REGISTRY.setdefault("keyframe_filter", keyframe_filter)
REGISTRY.setdefault("keypoint_evidence", keypoint_evidence)
REGISTRY.setdefault("camera_rectify", camera_rectify)
REGISTRY.setdefault("camera_rectify_grid", camera_rectify_grid)
REGISTRY.setdefault("camera_rectify_from_grid", camera_rectify_from_grid)
REGISTRY.setdefault("depth_camera_rays", depth_camera_rays)
REGISTRY.setdefault("depth_register", depth_register)
REGISTRY.setdefault("uniform_select", uniform_select)
REGISTRY.setdefault("shi_tomasi_select", shi_tomasi_select)
REGISTRY.setdefault("fast_select", fast_select)
REGISTRY.setdefault("pose_observable_select", pose_observable_select)
REGISTRY.setdefault("spatial_select_points", spatial_select_points)
REGISTRY.setdefault("resolution_pyramid", resolution_pyramid)
REGISTRY.setdefault("reproject_rgbd", reproject_rgbd)
REGISTRY.setdefault("rasterize_reprojection", rasterize_reprojection)
REGISTRY.setdefault("world_transform", world_transform)
REGISTRY.setdefault("triangulate_depth", triangulate_depth)
REGISTRY.setdefault("photometric_isp", photometric_isp)
REGISTRY.setdefault("project_depth_evidence", project_depth_evidence)
REGISTRY.setdefault("tracking_prepare", tracking_prepare)
REGISTRY.setdefault("tracking_health", tracking_health)
REGISTRY.setdefault("select_tracking_reference", select_tracking_reference)
REGISTRY.setdefault("select_best_inverse_depth", select_best_inverse_depth)

__all__ = [
    "REGISTRY",
    "TRACK_REASON_CONDITION",
    "TRACK_REASON_GEOMETRY",
    "TRACK_REASON_PARTIALS",
    "TRACK_REASON_PHOTOMETRIC",
    "TRACK_REASON_RESIDUAL",
    "TRACK_REASON_SOLVE",
    "aggregate_depth_cost_volume",
    "align_depth",
    "apply_camera_local_tracking_update",
    "binary_local_match",
    "brief_describe",
    "camera_rectify",
    "camera_rectify_from_grid",
    "camera_rectify_grid",
    "canonical_rescale",
    "compose_tracking_solve",
    "dense_pose_depth",
    "dense_track_reduce",
    "depth_camera_rays",
    "depth_convert",
    "depth_densify",
    "depth_downsample",
    "depth_evidence",
    "depth_geometry",
    "depth_pyramid",
    "depth_register",
    "depth_reproject",
    "depth_upsample",
    "deterministic_reduce_solve",
    "direct_pose_depth_dso",
    "direct_pose_depth_tensorized",
    "dso_select_points",
    "fast_select",
    "fit_local_planes",
    "fuse_depth",
    "keyframe_filter",
    "keypoint_evidence",
    "landmark_multiview_validate",
    "landmark_update",
    "lk_track",
    "photometric_isp",
    "point_splat_render",
    "pose_observable_select",
    "project_depth_evidence",
    "project_landmarks",
    "rasterize_reprojection",
    "register",
    "regularize_dense_depth",
    "reproject_rgbd",
    "residual_evidence",
    "rigid_align_3d",
    "resolution_pyramid",
    "rgbd_evidence",
    "select_best_inverse_depth",
    "select_compact",
    "select_tracking_reference",
    "shi_tomasi_select",
    "sparse_depth_validate",
    "sparse_pose_reduce",
    "sparse_rgbd_observation",
    "spatial_bin_reduce",
    "spatial_select_points",
    "temporal_best_observation",
    "temporal_frame_compare",
    "temporal_reproject_rgb",
    "temporal_restore_rgb",
    "temporal_restore_rgbd",
    "temporal_rgbd_reference",
    "tracking_health",
    "tracking_prepare",
    "triangulate_depth",
    "uniform_select",
    "validate_frame_batch",
    "voxel_surface_mesh",
    "voxelize_points",
    "world_transform",
    "world_voxel_reduce",
]
