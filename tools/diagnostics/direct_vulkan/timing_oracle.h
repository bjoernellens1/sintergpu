// SPDX-License-Identifier: Apache-2.0
// Independent scalar CPU check for the one fixed point-plane fixture.
#pragma once
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>
inline void sinterValidatePointPlaneFixture(const std::vector<uint8_t>& input,const std::vector<uint8_t>& output){
 constexpr std::array<size_t,6> sizes={8192,16512,16512,4156,24796,4100};
 constexpr size_t total=8192+16512+16512+4156+24796+4100;
 if(input.size()!=total||output.size()!=total)throw std::invalid_argument("fixture size");
 size_t off[6]{};for(size_t i=1;i<6;++i)off[i]=off[i-1]+sizes[i-1];
 for(size_t i:{size_t(0),size_t(1),size_t(2),size_t(5)})if(std::memcmp(input.data()+off[i],output.data()+off[i],sizes[i]))throw std::runtime_error("input changed");
 auto f=[&](const std::vector<uint8_t>& a,size_t byte)->double{float v;std::memcpy(&v,a.data()+byte,4);return v;};
 auto exact=[&](size_t i,size_t first,size_t end){if(std::memcmp(input.data()+off[i]+4*first,output.data()+off[i]+4*first,4*(end-first)))throw std::runtime_error("tail changed");};
 exact(3,1025,1039);exact(4,4100,4128);exact(4,6178,6199);
 auto close=[](double got,double expected){return std::isfinite(got)&&std::isfinite(expected)&&std::abs(got-expected)<=2e-6+3e-6*std::abs(expected);};
 for(size_t i=0;i<1025;++i){uint32_t index;std::memcpy(&index,input.data()+off[5]+4*i,4);if(index>=512)throw std::runtime_error("index range");double p[3],q[3],n[3];for(size_t k=0;k<3;++k){p[k]=f(input,off[0]+4*(4*index+k));q[k]=f(input,off[1]+4*(4*i+k));n[k]=f(input,off[2]+4*(4*i+k));}
  double residual=0;for(size_t k=0;k<3;++k)residual+=n[k]*(p[k]-q[k]);
  double jac[6]={p[1]*n[2]-p[2]*n[1],p[2]*n[0]-p[0]*n[2],p[0]*n[1]-p[1]*n[0],n[0],n[1],n[2]};
  if(!close(f(output,off[3]+4*i),residual))throw std::runtime_error("residual mismatch");
  for(size_t k=0;k<6;++k){size_t at=k<4?4*i+k:4128+2*i+(k-4);if(!close(f(output,off[4]+4*at),jac[k]))throw std::runtime_error("jacobian mismatch");}
 }
}
