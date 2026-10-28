// SPDX-License-Identifier: Apache-2.0
#include "timing_tick_bound.h"
#include <cassert>
int main(){assert(sinterTickInterval(254,2,8,1,5)==4);assert(sinterTickInterval(1,10,64,1,20)==9);try{sinterTickInterval(1,2,8,1,256);assert(false);}catch(const std::runtime_error&){}try{sinterTickInterval(1,10,8,1,5);assert(false);}catch(const std::runtime_error&){}try{sinterTickInterval(1,1,8,1,5);assert(false);}catch(const std::runtime_error&){}try{sinterTickInterval(257,2,8,1,5);assert(false);}catch(const std::runtime_error&){} }
