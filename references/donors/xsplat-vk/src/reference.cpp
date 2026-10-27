#include "xsplatvk/reference.hpp"

#include <algorithm>
#include <array>
#include <cmath>

namespace xsplatvk::reference {
namespace {

struct Mat3 {
  float v[3][3]{};
};

Vec3 transform_point(const std::array<float, 16> &m, Vec3 p) {
  return {
      m[0] * p.x + m[1] * p.y + m[2] * p.z + m[3],
      m[4] * p.x + m[5] * p.y + m[6] * p.z + m[7],
      m[8] * p.x + m[9] * p.y + m[10] * p.z + m[11],
  };
}

Mat3 mul(const Mat3 &a, const Mat3 &b) {
  Mat3 out{};
  for (int r = 0; r < 3; ++r)
    for (int c = 0; c < 3; ++c)
      for (int k = 0; k < 3; ++k)
        out.v[r][c] += a.v[r][k] * b.v[k][c];
  return out;
}

Mat3 transpose(const Mat3 &m) {
  Mat3 out{};
  for (int r = 0; r < 3; ++r)
    for (int c = 0; c < 3; ++c)
      out.v[r][c] = m.v[c][r];
  return out;
}

Mat3 rotation(std::array<float, 4> q) {
  const float n = std::sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
  if (n > 0.F)
    for (float &v : q)
      v /= n;
  const auto [x, y, z, w] = q;
  return {{{1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)},
           {2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)},
           {2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)}}};
}

Mat3 covariance3d(const Gaussian &g) {
  Mat3 s{};
  s.v[0][0] = std::exp(g.log_scale.x);
  s.v[1][1] = std::exp(g.log_scale.y);
  s.v[2][2] = std::exp(g.log_scale.z);
  const Mat3 l = mul(rotation(g.rotation), s);
  return mul(l, transpose(l));
}

Mat3 view_rotation(const Camera &c) {
  return {{{c.world_to_camera[0], c.world_to_camera[1], c.world_to_camera[2]},
           {c.world_to_camera[4], c.world_to_camera[5], c.world_to_camera[6]},
           {c.world_to_camera[8], c.world_to_camera[9], c.world_to_camera[10]}}};
}

Mat2Sym inverse(Mat2Sym m) {
  const float det = m.xx * m.yy - m.xy * m.xy;
  if (det <= 1e-12F)
    return {};
  return {m.yy / det, -m.xy / det, m.xx / det};
}

} // namespace

float sigmoid(float x) { return 1.F / (1.F + std::exp(-x)); }

float gaussian_alpha(Vec2 d, Mat2Sym c, float opacity) {
  const float q = c.xx * d.x * d.x + 2.F * c.xy * d.x * d.y + c.yy * d.y * d.y;
  return std::min(0.999F, opacity * std::exp(-0.5F * q));
}

CompositeState composite(CompositeState in, Vec3 rgb, float alpha) {
  const float w = in.transmittance * alpha;
  in.color.x += w * rgb.x;
  in.color.y += w * rgb.y;
  in.color.z += w * rgb.z;
  in.transmittance *= 1.F - alpha;
  return in;
}

SampleGrad composite_backward(CompositeState in, Vec3 rgb, float alpha, CompositeGrad out,
                              CompositeGrad &d_in) {
  const float dot_rgb = out.d_color.x * rgb.x + out.d_color.y * rgb.y + out.d_color.z * rgb.z;
  SampleGrad sample{
      {out.d_color.x * in.transmittance * alpha, out.d_color.y * in.transmittance * alpha,
       out.d_color.z * in.transmittance * alpha},
      in.transmittance * (dot_rgb - out.d_transmittance),
  };
  d_in.d_color = out.d_color;
  d_in.d_transmittance = alpha * dot_rgb + (1.F - alpha) * out.d_transmittance;
  return sample;
}

ProjectedGaussian project(const Gaussian &g, const Camera &camera, float dilation) {
  const Vec3 p = transform_point(camera.world_to_camera, g.mean);
  if (p.z <= 1e-4F || camera.width == 0 || camera.height == 0)
    return {};

  const Mat3 cov = covariance3d(g);
  const Mat3 w = view_rotation(camera);
  Mat3 j{};
  const float iz = 1.F / p.z;
  const float iz2 = iz * iz;
  j.v[0][0] = camera.fx * iz;
  j.v[0][2] = -camera.fx * p.x * iz2;
  j.v[1][1] = camera.fy * iz;
  j.v[1][2] = -camera.fy * p.y * iz2;

  const Mat3 t = mul(j, w);
  const Mat3 cov2 = mul(mul(t, cov), transpose(t));
  Mat2Sym c{cov2.v[0][0] + dilation, cov2.v[0][1], cov2.v[1][1] + dilation};
  const Vec2 px{camera.fx * p.x * iz + camera.cx, camera.fy * p.y * iz + camera.cy};
  const bool visible = px.x >= 0.F && px.x < static_cast<float>(camera.width) && px.y >= 0.F &&
                       px.y < static_cast<float>(camera.height);
  return {px, c, inverse(c), p.z, visible};
}

} // namespace xsplatvk::reference
