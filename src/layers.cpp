#include "layers.hpp"
#include <algorithm>
#include <cmath>

namespace tinf {

void rmsnorm(float* out, const float* x, const float* weight, int n, float eps) {
    double ss = 0;
    for (int i = 0; i < n; ++i) ss += (double)x[i] * x[i];
    float inv = 1.0f / std::sqrt((float)(ss / n) + eps);
    for (int i = 0; i < n; ++i) out[i] = x[i] * inv * weight[i];
}

void silu_mul(float* gate, const float* up, int n) {
    for (int i = 0; i < n; ++i) {
        float v = gate[i];
        gate[i] = (v / (1.0f + std::exp(-v))) * up[i];
    }
}

void add_inplace(float* a, const float* b, int n) {
    for (int i = 0; i < n; ++i) a[i] += b[i];
}

void softmax(float* x, int n) {
    float m = x[0];
    for (int i = 1; i < n; ++i) m = std::max(m, x[i]);
    float s = 0;
    for (int i = 0; i < n; ++i) {
        x[i] = std::exp(x[i] - m);
        s += x[i];
    }
    float inv = 1.0f / s;
    for (int i = 0; i < n; ++i) x[i] *= inv;
}

}  // namespace tinf
