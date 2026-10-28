// SPDX-License-Identifier: Apache-2.0
// Resource-independent scalar residual normal-equation terms, no reductions.
// Row-major H, then b, then squared residual. Caller owns weights and tree.
public float sinterNormalTerm<let Dimension : int>(uint term, float r, float J[Dimension]) {
    if (term < Dimension * Dimension) return J[term / Dimension] * J[term % Dimension];
    if (term < Dimension * Dimension + Dimension) return J[term - Dimension * Dimension] * r;
    return r * r;
}
