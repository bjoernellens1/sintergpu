# Projection pixel error model

This model validates the imported shader's camera transform and perspective pixel
coordinates on the declared finite FP32 fixture domain. It does not change a
shader, certify covariance math, or establish camera/rendering accuracy.

A single global relative error against float64 is inappropriate when the
camera-space depth or the final pixel coordinate is a cancellation. For example,
our unchanged1,025-element fixture contains an off-screen point whose transformed
depth is approximately0.010317. The unmodified donor C++ path and shader have
different legal FP32 evaluation paths. The original failed reports are retained;
no samples from those attempts are accepted.

The derivation is fixed from prepared FP32 input values and allowed arithmetic,
not fitted to observed GPU errors. `benchmarks/xsplat_projection_bounds.py` is
hashed alongside every raw record. The original17 fixtures and their inputs are
retained, including the near-plane/off-screen cases.

## Camera transform

For each component, the exact transform is a four-term dot product of
`(mean.x,mean.y,mean.z,1)` with one row of the uploaded FP32 matrix. Four
multiplications and three additions require at most seven rounded operations.
Contraction uses fewer rounding steps and is also enclosed. Let:

- `S = sum(abs(matrix[i,j] * homogeneous_mean[j]))`;
- `u = 2^-23`, conservatively allowing one-ULP relative error rather than assuming
  round-to-nearest's half-ULP bound;
- `gamma7 = 7*u/(1-7*u)`.

The absolute dot-product error is bounded by `gamma7*S`, independently of the
result's magnitude or cancellation. Add a seven-operation absolute underflow
allowance, `7*smallest_normal/(1-7*u)`, for possible intermediate flushing. The
model rejects subnormal source inputs and potential intermediate overflow.
It includes the corresponding FP64 arithmetic allowance for computing the center
and `S` itself. Constants, radii and interval endpoints are rounded outward.
This is an operation-count derivation for this camera dot, not a generic
conditioning multiplier for arbitrary shader functions.

## Perspective arithmetic

Interval arithmetic propagates the center bounds through reciprocal, products and
principal-point addition. Core add/multiply use a conservative one-ULP rounding
envelope, while division uses the Vulkan2.5-ULP bound for normal-range positive
denominators. Float64 interval operations are rounded outward as well. The model
unions both multiplication associations and equivalent division forms:

```text
(focal * x) * (1 / z) + principal
focal * (x * (1 / z)) + principal
focal * (x / z) + principal
(focal * x) / z + principal
```

The operation envelopes include the exact intermediate result, so permitted
multiply/add contraction is covered too. No reciprocal accuracy is inferred from
one vendor's measurements. These assumptions follow the [Vulkan precision and
operation rules](https://docs.vulkan.org/spec/latest/appendices/spirvenv.html);
relaxed-precision shaders, nonfinite inputs, overflow and unsupported denominator
ranges are outside this model. The actual SlangPy pipeline compiler dump is retained
as SPIR-V binary and assembly. It has only32-bit float types, no RelaxedPrecision
or fast-math decorations, and the camera/pixel chain uses the covered operations.
Every collection dumps its actual program and requires the exact reviewed binary
hash before projection dispatch; changed compiler output remains pending. Compiler
dumps and checks occur outside all timing regions.

If the center interval crosses the shader's depth rejection threshold, or pixel
intervals do not establish a unique viewport classification, qualification is
pending. Neither visibility flag is accepted merely because a boundary is
ambiguous. Definite behind-camera outputs must retain exact zero pixel values.
The actual visibility flags and complete tail records are always exact.

## Additional accuracy checks and evidence

All pixel coordinates must fit the input-derived envelope. Visible pixel
coordinates additionally retain the original benchmark's2e-6 absolute/relative
comparison to the independent float64 oracle rounded to FP32. Conic and depth,
Adam state and fixed-probe gradient checks retain their2e-6 bounds. No existing
centered projection, gradient, ordering or capacity test is weakened.

CPU verification includes60-digit decimal perspective calculations under severe
cancellation,24 independent FP32 camera addition orders and two perspective paths
for every measured extent, all four extents against the retained unmodified C++
reference, rejection of coordinates outside the envelope and rejection of both
flags for ambiguous classifications. The small127-element C++/oracle case also
keeps its original blanket2e-6 check.

Raw reports retain enclosure widths, observed deviation from the float64 oracle
rounded to FP32, visible-coordinate deviation and unresolved counts separately.
A wide interval conveys weak coordinate accuracy evidence; a passing record is
not permission to use off-screen projection coordinates as metric ground truth.
