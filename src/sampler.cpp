#include "sampler.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>

namespace tinf {

int sample_greedy(const float* logits, int n) {
    int b = 0;
    for (int i = 1; i < n; ++i)
        if (logits[i] > logits[b]) b = i;
    return b;
}

int sample_logits(const float* logits, int n, float temperature, float top_p, std::mt19937& rng) {
    if (temperature <= 0.0f) return sample_greedy(logits, n);
    std::vector<int> idx(n);
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(), [&](int a, int b) { return logits[a] > logits[b]; });

    std::vector<float> p(n);
    float m = logits[idx[0]];
    float s = 0;
    for (int i = 0; i < n; ++i) {
        p[i] = std::exp((logits[idx[i]] - m) / temperature);
        s += p[i];
    }
    for (int i = 0; i < n; ++i) p[i] /= s;

    int last = n;
    if (top_p < 1.0f) {
        float acc = 0;
        last = 0;
        for (int i = 0; i < n; ++i) {
            acc += p[i];
            last = i + 1;
            if (acc >= top_p) break;
        }
        float z = 0;
        for (int i = 0; i < last; ++i) z += p[i];
        for (int i = 0; i < last; ++i) p[i] /= z;
    }

    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    float r = dist(rng);
    float acc = 0;
    for (int i = 0; i < last; ++i) {
        acc += p[i];
        if (r <= acc) return idx[i];
    }
    return idx[last - 1];
}

}  // namespace tinf
