#pragma once

#include "common.hpp"
#include <random>

namespace tinf {

int sample_greedy(const float* logits, int n);

// temperature <= 0 is greedy. top_p in (0,1]; 1 disables nucleus.
int sample_logits(const float* logits, int n, float temperature, float top_p, std::mt19937& rng);

}  // namespace tinf
