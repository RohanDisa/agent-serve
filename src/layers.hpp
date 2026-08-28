#pragma once

#include "common.hpp"

namespace tinf {

void rmsnorm(float* out, const float* x, const float* weight, int n, float eps);
void silu_mul(float* gate, const float* up, int n);  // gate = silu(gate) * up
void add_inplace(float* a, const float* b, int n);
void softmax(float* x, int n);

}  // namespace tinf
