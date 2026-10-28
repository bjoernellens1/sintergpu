// SPDX-License-Identifier: Apache-2.0
// Candidate G18. Derived from the frozen Caspar DABSEG point_plane exports.
// See ../../candidate.json for upstream commits and source hashes.
// Explicit schedules preserve the two original residual FMA accumulation orders.
#ifndef SINTERGPU_CANDIDATE_POINT_PLANE
#define SINTERGPU_CANDIDATE_POINT_PLANE
public struct SinterPointPlaneFactor {
    float residual;
    float3 rotation;
    float3 translation;
};

// Rotation-first infinitesimal left action p + omega cross p + translation.
// No validity/robustification/frame transform is implied. Callers own those gates.
public SinterPointPlaneFactor sinterPointPlane<let AoSoASchedule : bool>(float3 p, float3 q, float3 n) {
    SinterPointPlaneFactor factor;
    float dz = fma(q.z, -1.0, p.z);
    float dx = fma(q.x, -1.0, p.x);
    float dy = fma(q.y, -1.0, p.y);
    if (AoSoASchedule)
        factor.residual = fma(n.y, dy, fma(n.x, dx, n.z * dz));
    else
        factor.residual = fma(n.x, dx, fma(n.y, dy, n.z * dz));
    factor.rotation = float3(fma(-1.0, p.z * n.y, p.y * n.z),
        fma(p.z, n.x, -1.0 * (p.x * n.z)),
        fma(-1.0, p.y * n.x, p.x * n.y));
    factor.translation = n;
    return factor;
}
#endif
