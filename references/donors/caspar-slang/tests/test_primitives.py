import importlib.util
import os
import re
import shutil
import subprocess

import pytest

from caspar_slang.ir import AccessKind, Buffer, KernelIR


def test_scalar_function_rejects_non_caspar_and_resource_addressing():
    from caspar_slang.primitives import emit_scalar_function

    ir = KernelIR("bad", [Buffer("x", AccessKind.READ, layout=((0,),))], [], ["x[42] = 1;"])
    with pytest.raises(ValueError, match="Caspar"):
        emit_scalar_function(ir, "bad")
    ir.metadata["source"] = "symforce.caspar"
    with pytest.raises(ValueError, match="resource|scalar"):
        emit_scalar_function(ir, "bad")
    ir.buffers[0] = Buffer("x", AccessKind.READ, components=3)
    with pytest.raises(ValueError, match="scalar"):
        emit_scalar_function(ir, "bad")


@pytest.mark.skipif(importlib.util.find_spec("symforce") is None, reason="requires SymForce")
def test_primitives_have_canonical_provenance_and_scalar_interfaces():
    from caspar_slang.primitives import primitive_kernels, primitives_source

    kernels = primitive_kernels()
    assert list(kernels) == [
        "backproject",
        "transform",
        "project",
        "tsdf_fuse",
        "se3_coefficients",
        "twist_point",
    ]
    source = primitives_source()
    assert source.count("float caspar_copysign(") == 1
    assert "StructuredBuffer" not in source
    assert "params." not in source
    for name, ir in kernels.items():
        assert ir.metadata["source"] == "symforce.caspar"
        match = re.search(r"void caspar_" + name + r"\((.*?)\)", source, re.S)
        assert match
        expected = ", ".join(
            ("out " if b.access == AccessKind.WRITE else "") + "float " + b.name for b in ir.buffers
        )
        assert " ".join(match[1].split()) == expected
        assert all(b.components == 1 and b.index_name is None for b in ir.buffers)


@pytest.mark.skipif(
    shutil.which("slangc") is None or importlib.util.find_spec("symforce") is None,
    reason="requires slangc and SymForce",
)
def test_primitives_spirv(tmp_path):
    from caspar_slang.compiler import compile_slang
    from caspar_slang.primitives import primitives_source

    source = tmp_path / "primitives.slang"
    source.write_text(
        primitives_source()
        + """\nRWStructuredBuffer<float> output;
[shader("compute")][numthreads(1,1,1)]
void computeMain(uint3 id:SV_DispatchThreadID) {
 float a,b,c,x,y,z,u,v,d,w;
 caspar_backproject(2,3,1,10,10,0,0,x,y,z);
 caspar_transform(x,y,z,1,0,0,0,1,0,0,0,1,0,0,0,x,y,z);
 caspar_project(x,y,z,10,10,0,0,u,v);
 caspar_tsdf_fuse(0,0,0.5,1,20,d,w);
 caspar_se3_coefficients(0,a,b,c);
 caspar_twist_point(0,0,0,0,0,0,a,b,c,x,y,z,1,x,y,z);
 output[0]=x+y+z+u+v+d+w;
}"""
    )
    output = tmp_path / "primitives.spv"
    compile_slang(source, output)
    assert output.read_bytes()[:4] == b"\x03\x02\x23\x07"
    if shutil.which("spirv-val"):
        subprocess.run(["spirv-val", str(output)], check=True, capture_output=True, text=True)


@pytest.mark.skipif(
    os.environ.get("CASPAR_SLANG_GPU_TESTS") != "1", reason="set CASPAR_SLANG_GPU_TESTS=1"
)
def test_gpu_primitives_match_independent_numpy():
    import numpy as np

    from caspar_slang.primitives import primitive_kernels, primitives_source
    from caspar_slang.runtime import VulkanRuntime

    runtime = VulkanRuntime()

    def dispatch(name, inputs, expected, atol=2e-5):
        inputs = np.asarray(inputs, dtype=np.float32)
        expected = np.asarray(expected)
        ir = primitive_kernels()[name]
        nin = sum(b.access == AccessKind.READ for b in ir.buffers)
        nout = sum(b.access == AccessKind.WRITE for b in ir.buffers)
        assert inputs.shape[1] == nin
        args = ", ".join(
            [f"inputData[i*{nin}+{j}]" for j in range(nin)] + [f"o{j}" for j in range(nout)]
        )
        decl = "; ".join(f"float o{j}" for j in range(nout)) + ";"
        stores = "\n".join(f"outputData[i*{nout}+{j}]=o{j};" for j in range(nout))
        source = (
            primitives_source()
            + f"""
StructuredBuffer<float> inputData;
RWStructuredBuffer<float> outputData;
[shader("compute")][numthreads(1,1,1)]
void computeMain(uint3 id:SV_DispatchThreadID) {{
 uint i=id.x; {decl}
 caspar_{name}({args});
 {stores}
}}
"""
        )
        out = runtime.buffer(np.zeros(len(inputs) * nout, np.float32))
        runtime.compile(source).dispatch(
            thread_count=[len(inputs), 1, 1],
            vars={"inputData": runtime.buffer(inputs), "outputData": out},
        )
        actual = out.to_numpy().view(np.float32).reshape(len(inputs), nout)
        assert np.isfinite(actual).all()
        np.testing.assert_allclose(actual, expected, atol=atol, rtol=2e-5)
        return actual

    rng = np.random.default_rng(443)
    xyz = rng.normal(size=(17, 3))
    xyz[:, 2] = rng.uniform(0.3, 4, 17)
    uv = rng.uniform(0, 300, size=(17, 2))
    intrinsics = np.array([310, 290, 160, 120])
    z = xyz[:, 2]
    inputs = np.column_stack((uv, z, np.tile(intrinsics, (17, 1))))
    points = np.column_stack(((uv[:, 0] - 160) * z / 310, (uv[:, 1] - 120) * z / 290, z))
    dispatch("backproject", inputs, points)
    dispatch(
        "project",
        np.column_stack((xyz, np.tile(intrinsics, (17, 1)))),
        np.column_stack((310 * xyz[:, 0] / z + 160, 290 * xyz[:, 1] / z + 120)),
    )
    angle = 0.37
    R = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
    translation = np.array([0.5, -0.3, 0.2])
    dispatch(
        "transform",
        np.column_stack((xyz, np.tile(np.r_[R.ravel(), translation], (17, 1)))),
        xyz @ R.T + translation,
    )
    fuse = np.array(
        [[0, 0, 0.3, 1, 20], [-0.5, 2, 0.8, 4, 3], [1, 0, 0, 0, 20], [0.2, 20, -0.2, 1, 20]]
    )
    total = fuse[:, 1] + fuse[:, 3]
    expected = np.column_stack(
        (
            (fuse[:, 0] * fuse[:, 1] + fuse[:, 2] * fuse[:, 3]) / np.maximum(total, 1e-12),
            np.minimum(total, fuse[:, 4]),
        )
    )
    dispatch("tsdf_fuse", fuse, expected)
    # Independent high precision series near zero, trigonometric coefficients elsewhere.
    theta_squared = np.array([0, 1e-12, 1e-6, 0.0099, 0.01, 0.03, 0.5, 3.0])
    coeff = []
    for t2 in theta_squared:
        if t2 < 1e-2:
            coeff.append(
                [
                    1 - t2 / 6 + t2 * t2 / 120 - t2**3 / 5040,
                    0.5 - t2 / 24 + t2 * t2 / 720 - t2**3 / 40320,
                    1 / 6 - t2 / 120 + t2 * t2 / 5040 - t2**3 / 362880,
                ]
            )
        else:
            t = np.sqrt(t2)
            coeff.append([np.sin(t) / t, (1 - np.cos(t)) / t2, (t - np.sin(t)) / (t2 * t)])
    coefficients = dispatch("se3_coefficients", theta_squared[:, None], coeff)
    omega = np.column_stack((np.sqrt(theta_squared), np.zeros((len(theta_squared), 2))))
    velocity = rng.normal(size=omega.shape)
    point = rng.normal(size=omega.shape)
    a, b, c = np.asarray(coeff).T
    expected = (
        point
        + a[:, None] * np.cross(omega, point)
        + b[:, None] * np.cross(omega, np.cross(omega, point))
    )
    translated = (
        velocity
        + b[:, None] * np.cross(omega, velocity)
        + c[:, None] * np.cross(omega, np.cross(omega, velocity))
    )
    flags = np.arange(len(omega)) % 2
    dispatch(
        "twist_point",
        np.column_stack((omega, velocity, coefficients, point, flags)),
        expected + flags[:, None] * translated,
    )
    # Exercise confidence capping over repeated fusion inside one GPU invocation.
    shader = (
        primitives_source()
        + """
RWStructuredBuffer<float> outputData;
[shader("compute")][numthreads(1,1,1)]
void computeMain(uint3 id:SV_DispatchThreadID) {
 float d=1,w=0;
 for(uint j=0;j<15;j++){ float nd,nw; caspar_tsdf_fuse(d,w,float(j%3)*0.2-0.2,1,5,nd,nw); d=nd;w=nw; }
 outputData[0]=d;outputData[1]=w;
}
"""
    )
    out = runtime.buffer(np.zeros(2, np.float32))
    runtime.compile(shader).dispatch(thread_count=[1, 1, 1], vars={"outputData": out})
    d, w = 1.0, 0.0
    for j in range(15):
        d = (d * w + (j % 3) * 0.2 - 0.2) / (w + 1)
        w = min(w + 1, 5)
    np.testing.assert_allclose(out.to_numpy().view(np.float32).reshape(-1), [d, w], atol=2e-6)
