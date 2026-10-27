from __future__ import annotations

import re

_SIMPLE_REPLACEMENTS = {
    "sinf(": "sin(",
    "cosf(": "cos(",
    "tanf(": "tan(",
    "asinf(": "asin(",
    "acosf(": "acos(",
    "atanf(": "atan(",
    "atan2f(": "atan2(",
    "sqrtf(": "sqrt(",
    "rsqrtf(": "rsqrt(",
    "powf(": "pow(",
    "fabsf(": "abs(",
    "fminf(": "min(",
    "fmaxf(": "max(",
    "fmaf(": "fma(",
    "fma(": "fma(",
    "__frcp_rn(": "rcp(",
    "copysignf(": "caspar_copysign(",
    "copysign(": "caspar_copysign(",
    "ldexpf(": "caspar_ldexp(",
    "ldexp(": "caspar_ldexp(",
    "hypotf(": "caspar_hypot(",
    "hypot(": "caspar_hypot(",
    "rhypotf(": "caspar_rhypot(",
    "rhypot(": "caspar_rhypot(",
    "norm3df(": "caspar_norm3(",
    "norm3d(": "caspar_norm3(",
    "rnorm3df(": "caspar_rnorm3(",
    "rnorm3d(": "caspar_rnorm3(",
    "norm4df(": "caspar_norm4(",
    "norm4d(": "caspar_norm4(",
    "rnorm4df(": "caspar_rnorm4(",
    "rnorm4d(": "caspar_rnorm4(",
}

_SINCOS = re.compile(
    r"^\s*sincosf?\(\s*([^,]+)\s*,\s*&([A-Za-z_]\w*)\s*,\s*&([A-Za-z_]\w*)\s*\)\s*;\s*$"
)


def lower_scalar_line(line: str) -> str:
    """Lower a CUDA-flavored scalar line emitted by Caspar to Slang/HLSL syntax."""
    m = _SINCOS.match(line)
    if m:
        x, s, c = m.groups()
        return f"sincos({x.strip()}, {s}, {c});"

    out = line
    for src, dst in _SIMPLE_REPLACEMENTS.items():
        out = out.replace(src, dst)

    # CUDA emits a few C-style float constants and ordinary arithmetic that Slang accepts.
    return out


def helper_source() -> str:
    return r"""
float caspar_copysign(float mag, float sign_value)
{
    uint m = asuint(abs(mag));
    uint s = asuint(sign_value) & 0x80000000u;
    return asfloat(m | s);
}

float caspar_ldexp(float x, int exponent)
{
    return x * exp2((float)exponent);
}

float caspar_hypot(float a, float b)
{
    return length(float2(a, b));
}

float caspar_rhypot(float a, float b)
{
    return rsqrt(a * a + b * b);
}

float caspar_norm3(float a, float b, float c)
{
    return length(float3(a, b, c));
}

float caspar_rnorm3(float a, float b, float c)
{
    return rsqrt(a * a + b * b + c * c);
}

float caspar_norm4(float a, float b, float c, float d)
{
    return length(float4(a, b, c, d));
}

float caspar_rnorm4(float a, float b, float c, float d)
{
    return rsqrt(a * a + b * b + c * c + d * d);
}
""".strip()
