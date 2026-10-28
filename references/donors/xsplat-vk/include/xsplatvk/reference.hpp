#pragma once

#include "xsplatvk/types.hpp"

namespace xsplatvk::reference {

float sigmoid(float x);
float gaussian_alpha(Vec2 delta, Mat2Sym conic, float opacity);
CompositeState composite(CompositeState in, Vec3 rgb, float alpha);
SampleGrad composite_backward(CompositeState in, Vec3 rgb, float alpha, CompositeGrad out_grad,
                              CompositeGrad &in_grad);
ProjectedGaussian project(const Gaussian &g, const Camera &camera, float covariance_dilation = 0.3F);

} // namespace xsplatvk::reference
