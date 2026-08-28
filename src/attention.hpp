#pragma once

#include "common.hpp"
#include "kvcache.hpp"

namespace tinf {

enum class RopeType { None, Standard, Llama3 };

struct RopeParams {
    RopeType type = RopeType::Standard;
    float theta = 10000.0f;
    int n_rot = 0;  // usually head_dim
    float scale = 1.0f;
    float low_freq_f = 1.0f;
    float high_freq_f = 4.0f;
    float orig_ctx = 8192.0f;
};

void rope_freqs(std::vector<float>& freq, const RopeParams& rp);

// Llama / GPT-NeoX style: rotate pairs (x[i], x[i + n_rot/2]).
void apply_rope(float* x, int n_heads, int head_dim, int pos, const float* freq, Timings* t);

void apply_rope_many(float* x, int seq, int n_heads, int head_dim, int pos0, const float* freq, Timings* t);

void attn_decode(float* out, const float* q, const KVCache& cache, int layer, int n_heads, int n_kv_heads,
                 int head_dim, int n_pos, float* scores, Timings* t);
void attn_prefill(float* out, const float* q, const float* k, const float* v, int seq, int n_heads,
                  int n_kv_heads, int head_dim, float* scores, Timings* t);

}  // namespace tinf
