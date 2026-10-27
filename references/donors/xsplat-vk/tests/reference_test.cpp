#include "xsplatvk/reference.hpp"

#include <cmath>
#include <cstdlib>
#include <iostream>

using namespace xsplatvk;

namespace {

void expect_near(float a, float b, float eps, const char *what) {
  if (std::abs(a - b) > eps) {
    std::cerr << what << ": " << a << " != " << b << '\n';
    std::exit(1);
  }
}

float loss(CompositeState s, CompositeGrad g) {
  return s.color.x * g.d_color.x + s.color.y * g.d_color.y + s.color.z * g.d_color.z +
         s.transmittance * g.d_transmittance;
}

void test_composite_backward() {
  CompositeState in{{0.1F, 0.2F, 0.3F}, 0.72F};
  Vec3 rgb{0.8F, 0.4F, 0.2F};
  float alpha = 0.31F;
  CompositeGrad d_out{{0.7F, -0.2F, 0.5F}, 0.13F};
  CompositeGrad d_in{};
  const SampleGrad d = reference::composite_backward(in, rgb, alpha, d_out, d_in);

  constexpr float e = 1e-3F;
  const auto finite = [&](float da, Vec3 drgb) {
    const float lp = loss(reference::composite(in, {rgb.x + drgb.x, rgb.y + drgb.y, rgb.z + drgb.z}, alpha + da), d_out);
    const float lm = loss(reference::composite(in, {rgb.x - drgb.x, rgb.y - drgb.y, rgb.z - drgb.z}, alpha - da), d_out);
    return (lp - lm) / (2.F * e);
  };
  expect_near(d.d_alpha, finite(e, {}), 2e-4F, "d_alpha");
  expect_near(d.d_rgb.x, finite(0.F, {e, 0, 0}), 2e-4F, "d_rgb.x");
  expect_near(d.d_rgb.y, finite(0.F, {0, e, 0}), 2e-4F, "d_rgb.y");
  expect_near(d.d_rgb.z, finite(0.F, {0, 0, e}), 2e-4F, "d_rgb.z");
}

void test_projection() {
  Camera c{};
  c.world_to_camera = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
  c.fx = c.fy = 500.F;
  c.cx = 320.F;
  c.cy = 240.F;
  c.width = 640;
  c.height = 480;

  Gaussian g{};
  g.mean = {0.F, 0.F, 2.F};
  g.log_scale = {-3.F, -3.F, -3.F};
  const auto p = reference::project(g, c);
  expect_near(p.mean_px.x, 320.F, 1e-5F, "mean x");
  expect_near(p.mean_px.y, 240.F, 1e-5F, "mean y");
  if (!p.visible || p.covariance.xx <= 0 || p.covariance.yy <= 0) {
    std::cerr << "projection should be visible and positive definite\n";
    std::exit(1);
  }
}

} // namespace

int main() {
  test_composite_backward();
  test_projection();
  std::cout << "reference tests passed\n";
}
