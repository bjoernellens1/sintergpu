"""Scalar SLAM math scheduled by upstream Caspar, exported as callable Slang.

Resource addressing and dispatch live in the consuming runtime. These functions
retain the actual SymForce -> DABSEG -> Caspar register scheduling provenance.
"""

import re
from functools import lru_cache

from .caspar_adapter import from_caspar_kernel
from .intrinsics import helper_source, lower_scalar_line
from .ir import AccessKind, KernelIR


def emit_scalar_function(kernel: KernelIR, name: str) -> str:
    """Export scalar sequential Caspar IR without constructing a compute shader."""
    kernel.validate()
    if kernel.metadata.get("source") != "symforce.caspar":
        raise ValueError("Scalar export requires canonical Caspar IR provenance")
    if not re.fullmatch(r"[A-Za-z_]\w*", name):
        raise ValueError("Invalid scalar function name")
    for buf in kernel.buffers:
        if (
            buf.components != 1
            or buf.index_name is not None
            or buf.scalar_type != "float"
            or buf.layout != ((0,),)
            or buf.access == AccessKind.READ_WRITE
        ):
            raise ValueError("Scalar export requires scalar sequential read/write buffers")
    body = []
    written = set()
    for original in kernel.code_lines:
        line = original
        for buf in kernel.buffers:
            address = re.escape(
                f"{buf.name}[(0u * params.{buf.name}_num_alloc + global_thread_idx * 1u + 0u)]"
            )
            if buf.access == AccessKind.READ:
                match = re.fullmatch(r"\s*([A-Za-z_]\w*) = " + address + r";\s*", line)
                if match:
                    line = f"{match[1]} = {buf.name};"
            elif buf.access == AccessKind.WRITE:
                match = re.fullmatch(address + r" = ([A-Za-z_]\w*);\s*", line)
                if match:
                    line = f"{buf.name} = {match[1]};"
                    written.add(buf.name)
        if any(token in line for token in ("[", "]", "params.", "global_thread_idx")):
            raise ValueError(f"Unlowered resource addressing in scalar function: {line}")
        body.append(lower_scalar_line(line))
    if written != {b.name for b in kernel.buffers if b.access == AccessKind.WRITE}:
        raise ValueError("Scalar function must initialize every output")
    args = ", ".join(
        ("out " if b.access == AccessKind.WRITE else "") + "float " + b.name for b in kernel.buffers
    )
    registers = (
        "float " + ", ".join(r + " = 0.0f" for r in kernel.registers) + ";"
        if kernel.registers
        else ""
    )
    return (
        f"// SymForce/Caspar DABSEG kernel: {kernel.name}\nvoid {name}({args})\n{{\n    {registers}\n"
        + "\n".join("    " + line for line in body)
        + "\n}\n"
    )


@lru_cache(maxsize=1)
def primitive_kernels() -> dict[str, KernelIR]:
    import symforce.symbolic as sf
    from symforce import typing as T
    from symforce.caspar import memory as mem
    from symforce.caspar.code_generation.library import CasparLibrary

    R = T.Annotated[sf.Scalar, mem.ReadSequential]
    W = T.Annotated[sf.Scalar, mem.WriteSequential]
    lib = CasparLibrary(name="slam_primitives")

    @lib.add_kernel
    def backproject(u: R, v: R, z: R, fx: R, fy: R, cx: R, cy: R) -> T.Tuple[W, W, W]:
        return (u - cx) * z / fx, (v - cy) * z / fy, z

    @lib.add_kernel
    def transform(
        x: R,
        y: R,
        z: R,
        r00: R,
        r01: R,
        r02: R,
        r10: R,
        r11: R,
        r12: R,
        r20: R,
        r21: R,
        r22: R,
        tx: R,
        ty: R,
        tz: R,
    ) -> T.Tuple[W, W, W]:
        return (
            r00 * x + r01 * y + r02 * z + tx,
            r10 * x + r11 * y + r12 * z + ty,
            r20 * x + r21 * y + r22 * z + tz,
        )

    @lib.add_kernel
    def project(x: R, y: R, z: R, fx: R, fy: R, cx: R, cy: R) -> T.Tuple[W, W]:
        return fx * x / z + cx, fy * y / z + cy

    @lib.add_kernel
    def tsdf_fuse(
        oldDistance: R, oldWeight: R, observation: R, observationWeight: R, maxWeight: R
    ) -> T.Tuple[W, W]:
        # The denominator uses the actual observation mass before confidence capping.
        total = oldWeight + observationWeight
        safe = sf.Max(total, sf.Scalar(1e-12))
        return (oldDistance * oldWeight + observation * observationWeight) / safe, sf.Min(
            total, maxWeight
        )

    @lib.add_kernel
    def se3_coefficients(thetaSquared: R) -> T.Tuple[W, W, W]:
        # Caspar evaluates both sides of a select. Keep the trigonometric side
        # finite at zero; the Taylor side preserves float32 precision near zero.
        safe_squared = sf.Max(thetaSquared, sf.Scalar(1e-2))
        theta = sf.sqrt(safe_squared)
        small = thetaSquared < sf.Scalar(1e-2)
        a = sf.Piecewise(
            (1 - thetaSquared / 6 + thetaSquared**2 / 120, small), (sf.sin(theta) / theta, True)
        )
        b = sf.Piecewise(
            (sf.Scalar(0.5) - thetaSquared / 24 + thetaSquared**2 / 720, small),
            ((1 - sf.cos(theta)) / safe_squared, True),
        )
        c = sf.Piecewise(
            (sf.Scalar(1) / 6 - thetaSquared / 120 + thetaSquared**2 / 5040, small),
            ((theta - sf.sin(theta)) / (safe_squared * theta), True),
        )
        return a, b, c

    @lib.add_kernel
    def twist_point(
        wx: R,
        wy: R,
        wz: R,
        vx: R,
        vy: R,
        vz: R,
        a: R,
        b: R,
        c: R,
        px: R,
        py: R,
        pz: R,
        translationFlag: R,
    ) -> T.Tuple[W, W, W]:
        omega = sf.V3(wx, wy, wz)
        point = sf.V3(px, py, pz)
        velocity = sf.V3(vx, vy, vz)
        rotated = point + a * omega.cross(point) + b * omega.cross(omega.cross(point))
        translated = velocity + b * omega.cross(velocity) + c * omega.cross(omega.cross(velocity))
        result = rotated + translationFlag * translated
        return result[0], result[1], result[2]

    kernels = {kernel.name: from_caspar_kernel(kernel) for kernel in lib.kernels}
    for ir in kernels.values():
        ir.metadata.update(
            primitive=ir.name,
            schedule="DABSEG/get_lines",
            register_count=str(len(ir.registers)),
            export="scalar sequential function",
        )
    return kernels


@lru_cache(maxsize=1)
def primitives_source() -> str:
    return (
        helper_source()
        + "\n\n"
        + "\n".join(
            emit_scalar_function(ir, "caspar_" + name) for name, ir in primitive_kernels().items()
        )
    )
