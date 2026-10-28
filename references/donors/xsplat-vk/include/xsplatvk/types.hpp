#pragma once

#include <array>
#include <cstdint>

namespace xsplatvk {

struct Vec2 {
  float x{};
  float y{};
};

struct Vec3 {
  float x{};
  float y{};
  float z{};
};

struct Mat2Sym {
  float xx{};
  float xy{};
  float yy{};
};

// CPU/reference representation only. Production GPU storage is SoA.
struct Gaussian {
  Vec3 mean{};
  Vec3 log_scale{};
  std::array<float, 4> rotation{0.F, 0.F, 0.F, 1.F};
  float opacity_logit{};
  Vec3 sh0{};
};

struct Camera {
  std::array<float, 16> world_to_camera{}; // row-major 4x4
  float fx{};
  float fy{};
  float cx{};
  float cy{};
  std::uint32_t width{};
  std::uint32_t height{};
};

struct ProjectedGaussian {
  Vec2 mean_px{};
  Mat2Sym covariance{};
  Mat2Sym conic{};
  float depth{};
  bool visible{};
};

struct CompositeState {
  Vec3 color{};
  float transmittance{1.F};
};

struct CompositeGrad {
  Vec3 d_color{};
  float d_transmittance{};
};

struct SampleGrad {
  Vec3 d_rgb{};
  float d_alpha{};
};

} // namespace xsplatvk
