// SPDX-License-Identifier: Apache-2.0
// Experimental VKL-compatible per-pixel depth evidence; see docs/depth-evidence-map.md.
// Caller provides the right and bottom neighbor (zero, invalid at boundaries).
public struct DepthEvidenceMap {
    public float support;
    public float nearCount;
    public float farCount;
    public float depth;
    public float depthSquared;
    public float edge;
    public float confidence;
    public float stable;
    public float pairCount;
    public float pairSquaredDifference;
};
public DepthEvidenceMap depthEvidenceMap(float depth, bool valid, float confidence,
    float rightDepth, bool rightValid, float downDepth, bool downValid,
    float nearBound, float farBound)
{
    bool usable = valid && isfinite(depth);
    bool rp = usable && rightValid && isfinite(rightDepth);
    bool dp = usable && downValid && isfinite(downDepth);
    float d = usable ? depth : 0.0;
    float rd = rp ? rightDepth - d : 0.0;
    float dd = dp ? downDepth - d : 0.0;
    bool edge = (rp && abs(rd) > max(abs(d), 1e-6) * 0.25) ||
                (dp && abs(dd) > max(abs(d), 1e-6) * 0.25);
    DepthEvidenceMap r;
    r.support = usable ? 1.0 : 0.0;
    r.nearCount = (usable && d < nearBound) ? 1.0 : 0.0;
    r.farCount = (usable && d > farBound) ? 1.0 : 0.0;
    r.depth = d;
    r.depthSquared = d*d;
    r.edge = edge ? 1.0 : 0.0;
    r.confidence = usable ? confidence : 0.0;
    r.stable = (usable && !edge && confidence > 0.0) ? 1.0 : 0.0;
    r.pairCount = (rp ? 1.0 : 0.0) + (dp ? 1.0 : 0.0);
    r.pairSquaredDifference = rd*rd + dd*dd;
    return r;
}
