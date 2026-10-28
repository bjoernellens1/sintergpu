// SPDX-License-Identifier: Apache-2.0
// Test-only driver: execute the retained, unmodified donor CPU projection.
#include "xsplatvk/reference.hpp"
#include <iomanip>
#include <iostream>
int main() {
  xsplatvk::Camera c{};
  for (auto& v:c.world_to_camera) std::cin>>v;
  std::cin>>c.fx>>c.fy>>c.cx>>c.cy>>c.width>>c.height;
  unsigned n;std::cin>>n;
  std::cout<<std::setprecision(9);
  for(unsigned i=0;i<n;++i) {
    xsplatvk::Gaussian g{};
    std::cin>>g.mean.x>>g.mean.y>>g.mean.z>>g.log_scale.x>>g.log_scale.y>>g.log_scale.z;
    for(auto& v:g.rotation)std::cin>>v;
    if(!std::cin)return 2;
    const auto p=xsplatvk::reference::project(g,c);
    std::cout<<p.mean_px.x<<' '<<p.mean_px.y<<' '<<p.conic.xx<<' '<<p.conic.xy<<' '<<p.conic.yy<<' '<<p.depth<<' '<<p.visible<<'\n';
  }
}
