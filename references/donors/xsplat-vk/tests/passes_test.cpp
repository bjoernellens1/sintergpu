#include "xsplatvk/passes.hpp"

#include <cstdlib>
#include <iostream>

using namespace xsplatvk;

static void expect(bool ok, const char *msg) {
  if (!ok) {
    std::cerr << msg << '\n';
    std::exit(1);
  }
}

int main() {
  expect(shader_name(Pass::ProjectionForward, 32, false) == "projection_forward.sg32.spv", "projection shader name");
  expect(shader_name(Pass::RasterBackward, 64, true) == "rasterize_backward.sg64.emuf32.spv", "backward emulation shader name");
  expect(shader_name(Pass::RadixSpine, 32, false) == "radix_sort/spine.spv", "radix shader name");
  expect(supported_subgroup(16) == 16, "subgroup exact");
  expect(pass_info(Pass::ProjectionBackward).binding_mask == 0x1fbf, "projection backward bindings");
  std::cout << "pass manifest tests passed\n";
}
