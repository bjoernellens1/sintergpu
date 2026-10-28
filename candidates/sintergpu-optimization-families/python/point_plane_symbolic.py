# SPDX-License-Identifier: Apache-2.0
"""Single symbolic expression with scalar and vector export adapters.

Pass the caller's symforce.symbolic module; importing this module has no optional
dependency or runtime initialization. The frozen donor generators remain intact.
Based on Caspar's Apache-2.0 point_plane definitions; see the family manifest.
"""


def point_plane_expression(sf, point, target, normal):
    """Return residual and rotation-first left-action Jacobian at zero twist."""
    xi = sf.V6.symbolic('xi')
    moved = point + sf.V3(xi[:3]).cross(point) + sf.V3(xi[3:])
    expression = normal.dot(moved - target)
    zero = dict.fromkeys(xi, 0)
    return expression.subs(zero), sf.V6([expression.diff(v).subs(zero) for v in xi])


def point_plane_scalars(sf, px, py, pz, qx, qy, qz, nx, ny, nz):
    """Seven-scalar adapter for inline/fused generated exports."""
    residual, jacobian = point_plane_expression(sf, sf.V3(px, py, pz),
        sf.V3(qx, qy, qz), sf.V3(nx, ny, nz))
    return (residual, *jacobian)
