// SPDX-License-Identifier: Apache-2.0
#pragma once
// VKL's exact distinct ordered 31x31 comparison pattern. Independent of resources.
struct VklBriefPair { int64_t ax; int64_t ay; int64_t bx; int64_t by; };
VklBriefPair vklBriefPair(uint bit)
{
    uint64_t code=(uint64_t(bit)*uint64_t(104729)+uint64_t(13007))%uint64_t(922560);
    uint64_t first=code/uint64_t(960), second=code%uint64_t(960);
    second+=uint64_t(second>=first);
    VklBriefPair p;
    p.ax=int64_t(first%31)-15; p.ay=int64_t(first/31)-15;
    p.bx=int64_t(second%31)-15; p.by=int64_t(second/31)-15;
    return p;
}
