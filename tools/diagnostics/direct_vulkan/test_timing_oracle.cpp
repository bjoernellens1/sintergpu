// SPDX-License-Identifier: Apache-2.0
#include "timing_oracle.h"
#include <cassert>
#include <fstream>
#include <iterator>
#include <limits>
int main(){std::vector<uint8_t> a(74268,0);std::vector<uint8_t> b=a;sinterValidatePointPlaneFixture(a,b);b[0]=1;try{sinterValidatePointPlaneFixture(a,b);assert(false);}catch(const std::runtime_error&){}b=a;size_t outputOffset=8192+16512+16512;b[outputOffset+1025*4]=1;try{sinterValidatePointPlaneFixture(a,b);assert(false);}catch(const std::runtime_error&){}b=a;float bad=1;std::memcpy(b.data()+outputOffset,&bad,4);try{sinterValidatePointPlaneFixture(a,b);assert(false);}catch(const std::runtime_error&){}
 const char* root="diagnostics/direct-point-plane-4070-1e4af42/retained-receipt.json.artifacts/";
 auto read=[&](const char* name){std::ifstream f(std::string(root)+name,std::ios::binary);if(!f)throw std::runtime_error("fixture missing");return std::vector<uint8_t>(std::istreambuf_iterator<char>(f),{});};
 auto actual=read("input.bin");auto donor=read("donor.bin");auto candidate=read("candidate.bin");sinterValidatePointPlaneFixture(actual,donor);sinterValidatePointPlaneFixture(actual,candidate);assert(donor==candidate);
 auto corrupt=donor;float nan=std::numeric_limits<float>::quiet_NaN();
 std::memcpy(corrupt.data()+outputOffset,&nan,4);try{sinterValidatePointPlaneFixture(actual,corrupt);assert(false);}catch(const std::runtime_error&){}
 corrupt=donor;float badJac=1000000;std::memcpy(corrupt.data()+outputOffset+4156+4*(4128+1),&badJac,4);
 try{sinterValidatePointPlaneFixture(actual,corrupt);assert(false);}catch(const std::runtime_error&){}
 auto invalid=actual;uint32_t badIndex=512;std::memcpy(invalid.data()+74268-4100,&badIndex,4);auto invalidOut=donor;std::memcpy(invalidOut.data()+74268-4100,&badIndex,4);
 try{sinterValidatePointPlaneFixture(invalid,invalidOut);assert(false);}catch(const std::runtime_error&){}
}
