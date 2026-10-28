# SPDX-License-Identifier: Apache-2.0
"""Owned Caspar adapters sharing one point-plane symbolic expression.

Ported from caspar-slang cab53390aa111cd5526cb30c9aef484e73521186,
gn.py:33-61 and reprojection.py:74-200. Photometric/dense expressions and
export metadata stay unchanged. Frozen donors remain independent references.
SymForce/Caspar are optional generation dependencies, imported only on use.
"""
from functools import lru_cache
from sintergpu.codegen.point_plane import point_plane_expression, point_plane_scalars


@lru_cache(maxsize=1)
def residual_kernel():
    import symforce.symbolic as sf
    from symforce import typing as T
    from symforce.caspar import memory as mem
    from symforce.caspar.code_generation.library import CasparLibrary

    from caspar_slang.caspar_adapter import from_caspar_kernel

    lib = CasparLibrary()

    @lib.add_kernel
    def point_plane(
        points: T.Annotated[sf.V3, mem.ReadIndexed],
        targets: T.Annotated[sf.V3, mem.ReadSequential],
        normals: T.Annotated[sf.V3, mem.ReadSequential],
    ) -> T.Tuple[
        T.Annotated[sf.Scalar, mem.WriteSequential], T.Annotated[sf.V6, mem.WriteSequential]
    ]:
        return point_plane_expression(sf, points, targets, normals)

    return from_caspar_kernel(lib.kernels[0])


@lru_cache(maxsize=1)
def factor_kernels():
    """Canonical geometric/photometric factors for fused dense GPU solvers."""
    import symforce.symbolic as sf
    from symforce import typing as T
    from symforce.caspar import memory as mem
    from symforce.caspar.code_generation.library import CasparLibrary
    from caspar_slang.caspar_adapter import from_caspar_kernel

    R = T.Annotated[sf.Scalar, mem.ReadSequential]
    W = T.Annotated[sf.Scalar, mem.WriteSequential]
    lib = CasparLibrary(name="dense_factors")

    @lib.add_kernel
    def point_plane(
        px: R, py: R, pz: R, qx: R, qy: R, qz: R, nx: R, ny: R, nz: R
    ) -> T.Tuple[W, W, W, W, W, W, W]:
        return point_plane_scalars(sf, px, py, pz, qx, qy, qz, nx, ny, nz)

    @lib.add_kernel
    def photo_factor(
        x: R, y: R, z: R, fx: R, fy: R, gx: R, gy: R, targetI: R, sourceI: R
    ) -> T.Tuple[W, W, W, W, W, W, W]:
        point = sf.V3(x, y, z)
        xi = sf.V6.symbolic("xi")
        moved = point + sf.V3(xi[:3]).cross(point) + sf.V3(xi[3:])
        error = (
            targetI
            - sourceI
            + gx * fx * (moved[0] / moved[2] - x / z)
            + gy * fy * (moved[1] / moved[2] - y / z)
        )
        zero = {v: 0 for v in xi}
        return tuple([error.subs(zero)] + [error.diff(v).subs(zero) for v in xi])

    @lib.add_kernel
    def dense_factor(
        wx: R,
        wy: R,
        wz: R,
        x: R,
        y: R,
        z: R,
        tr00: R,
        tr01: R,
        tr02: R,
        tr10: R,
        tr11: R,
        tr12: R,
        tr20: R,
        tr21: R,
        tr22: R,
        sr00: R,
        sr01: R,
        sr02: R,
        sr10: R,
        sr11: R,
        sr12: R,
        sr20: R,
        sr21: R,
        sr22: R,
        rx: R,
        ry: R,
        rz: R,
        rho: R,
        qx: R,
        qy: R,
        qz: R,
        nx: R,
        ny: R,
        nz: R,
        targetIntensity: R,
        sourceIntensity: R,
        gx: R,
        gy: R,
        fx: R,
        fy: R,
    ) -> T.Tuple[W, W, W, W, W, W, W, W, W, W, W, W, W, W, W, W]:
        target_rotation = sf.M33([tr00, tr01, tr02, tr10, tr11, tr12, tr20, tr21, tr22])
        source_rotation = sf.M33([sr00, sr01, sr02, sr10, sr11, sr12, sr20, sr21, sr22])
        xi = sf.V6.symbolic("xi")
        drho = sf.Symbol("delta_rho")
        world_point = sf.V3(wx, wy, wz)
        change = sf.V3(xi[:3]).cross(world_point) + sf.V3(xi[3:])
        change += source_rotation * sf.V3(rx, ry, rz) * (1 / (rho + drho) - 1 / rho)
        moved = sf.V3(x, y, z) + target_rotation * change
        geometric = sf.V3(nx, ny, nz).dot(moved - sf.V3(qx, qy, qz))
        photometric = (
            targetIntensity
            - sourceIntensity
            + gx * fx * (moved[0] / moved[2] - x / z)
            + gy * fy * (moved[1] / moved[2] - y / z)
        )
        zero = {v: 0 for v in [*xi, drho]}
        values = [geometric.subs(zero), photometric.subs(zero)]
        values += [geometric.diff(v).subs(zero) for v in xi]
        values += [photometric.diff(v).subs(zero) for v in xi]
        values += [geometric.diff(drho).subs(zero), photometric.diff(drho).subs(zero)]
        return tuple(values)

    kernels = {kernel.name: from_caspar_kernel(kernel) for kernel in lib.kernels}
    for ir in kernels.values():
        ir.metadata.update(
            schedule="DABSEG/get_lines",
            register_count=str(len(ir.registers)),
            source_pose_sign="positive",
            target_pose_sign="negative",
        )
    kernels["dense_factor"].metadata.update(
        pose_basis="left world SE3 omega_xyz translation_xyz", depth_basis="source inverse z depth"
    )
    return kernels


@lru_cache(maxsize=1)
def dense_factor_source():
    """Self-contained dense factors; runtime supplies sampled fixed observations."""
    from caspar_slang.primitives import primitives_source, emit_scalar_function
    return (
        primitives_source()
        + "\n"
        + "\n".join(
            emit_scalar_function(ir, "caspar_" + name) for name, ir in factor_kernels().items()
        )
    )
