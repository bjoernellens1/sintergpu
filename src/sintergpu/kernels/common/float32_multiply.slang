// SPDX-License-Identifier: Apache-2.0
#pragma once
// Keep the native float32 product rounded before a later add/subtract. Slang
// 2026.12 does not emit this decoration from the precise FP option. This helper
// controls contraction, not native denormal mode, NaN payload or sqrt accuracy.
float sintergpuF32MulNoContract(float a,float b)
{
    __target_switch
    {
    case spirv:
        return spirv_asm
        {
            %product:$$float = OpFMul $a $b;
            OpDecorate %product NoContraction;
            result:$$float = OpCopyObject %product;
        };
    default:
        precise float product=a*b;
        return product;
    }
}
float3 sintergpuF32ScaleNoContract(float3 value,float scale)
{
    return float3(sintergpuF32MulNoContract(value.x,scale),
                  sintergpuF32MulNoContract(value.y,scale),
                  sintergpuF32MulNoContract(value.z,scale));
}
