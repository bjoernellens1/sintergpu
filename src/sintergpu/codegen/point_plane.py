# SPDX-License-Identifier: Apache-2.0
"""Single symbolic expression with scalar and vector export adapters.

Pass the caller's symforce.symbolic module; importing this module has no optional
dependency or runtime initialization. The frozen donor generators remain intact.
Based on Caspar's Apache-2.0 point_plane definitions; see the family manifest.
"""


def point_plane_expression(sf, point, target, normal):
    """Return residual and closed-form left-action rotation-first Jacobian."""
    return normal.dot(point - target), sf.V6([*point.cross(normal), *normal])


def point_plane_scalars(sf, px, py, pz, qx, qy, qz, nx, ny, nz):
    """Seven-scalar adapter for inline/fused generated exports."""
    residual, jacobian = point_plane_expression(sf, sf.V3(px, py, pz),
        sf.V3(qx, qy, qz), sf.V3(nx, ny, nz))
    return (residual, *jacobian)
