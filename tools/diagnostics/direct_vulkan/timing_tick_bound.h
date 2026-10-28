// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cmath>
#include <cstdint>
#include <stdexcept>
inline long double sinterTickInterval(uint64_t a,uint64_t b,uint32_t bits,long double period,uint64_t hostBound){
 if(!bits||bits>64||!std::isfinite(period)||period<=0||!hostBound)throw std::invalid_argument("timestamp capability");
 const auto wrap=std::ldexp(period,int(bits));if((long double)hostBound>=wrap)throw std::runtime_error("timestamp wrap ambiguity");
 uint64_t mask=bits==64?~uint64_t(0):(uint64_t(1)<<bits)-1;
 if((a&~mask)||(b&~mask))throw std::runtime_error("timestamp invalid bits");
 uint64_t delta=(b-a)&mask;long double ns=(long double)delta*period;
 if(!delta||ns>(long double)hostBound)throw std::runtime_error("timestamp interval bound");
 return ns;
}
