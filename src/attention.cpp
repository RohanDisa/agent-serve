#include "attention.hpp"
#include "layers.hpp"
#include "kvcache.hpp"
#include "quant.hpp"
#include <cmath>

namespace tinf {

void rope_freqs(std::vector<float>& freq, const RopeParams& rp) {
    int d2 = rp.n_rot / 2;
    freq.resize(d2);
    for (int i = 0; i < d2; ++i) {
        float f = 1.0f / std::pow(rp.theta, (float)(2 * i) / (float)rp.n_rot);
        if (rp.type == RopeType::Llama3) {
            float wavelen = 2.0f * 3.14159265358979323846f / f;
            float lo_wl = rp.orig_ctx / rp.low_freq_f;
            float hi_wl = rp.orig_ctx / rp.high_freq_f;
            if (wavelen < hi_wl) {
                // keep
            } else if (wavelen > lo_wl) {
                f /= rp.scale;
            } else {
                float smooth = (rp.orig_ctx / wavelen - rp.low_freq_f) / (rp.high_freq_f - rp.low_freq_f);
                f = (1.0f - smooth) * f / rp.scale + smooth * f;
            }
        }
        freq[i] = f;
    }
}

static void rope_one(float* x, int head_dim, int pos, const float* freq) {
    int d2 = head_dim / 2;
    for (int i = 0; i < d2; ++i) {
        float c = std::cos((float)pos * freq[i]);
        float s = std::sin((float)pos * freq[i]);
        float x0 = x[i];
        float x1 = x[i + d2];
        x[i] = x0 * c - x1 * s;
        x[i + d2] = x0 * s + x1 * c;
    }
}

void apply_rope(float* x, int n_heads, int head_dim, int pos, const float* freq, Timings* t) {
    ScopeTime st(t ? &t->rope : nullptr);
    for (int h = 0; h < n_heads; ++h) rope_one(x + h * head_dim, head_dim, pos, freq);
}

void apply_rope_many(float* x, int seq, int n_heads, int head_dim, int pos0, const float* freq, Timings* t) {
    ScopeTime st(t ? &t->rope : nullptr);
    for (int i = 0; i < seq; ++i)
        for (int h = 0; h < n_heads; ++h)
            rope_one(x + (i * n_heads + h) * head_dim, head_dim, pos0 + i, freq);
}

void attn_decode(float* out, const float* q, const KVCache& cache, int layer, int n_heads, int n_kv_heads,
                 int head_dim, int n_pos, float* scores, Timings* t) {
    const int n_rep = n_heads / n_kv_heads;
    const float scale = 1.0f / std::sqrt((float)head_dim);
    for (int h = 0; h < n_heads; ++h) {
        const int hv = h / n_rep;
        const float* qh = q + h * head_dim;
        {
            ScopeTime st(t ? &t->attn : nullptr);
            for (int p = 0; p < n_pos; ++p)
                scores[p] = scale * dot_f32(qh, cache.k_at(layer, hv, p), head_dim, true);
        }
        {
            ScopeTime st(t ? &t->softmax : nullptr);
            softmax(scores, n_pos);
        }
        ScopeTime st(t ? &t->attn : nullptr);
        float* oh = out + h * head_dim;
        std::memset(oh, 0, (size_t)head_dim * sizeof(float));
        for (int p = 0; p < n_pos; ++p) {
            const float* vp = cache.v_at(layer, hv, p);
            const float a = scores[p];
            for (int d = 0; d < head_dim; ++d) oh[d] += a * vp[d];
        }
    }
}

void attn_prefill(float* out, const float* q, const float* k, const float* v, int seq, int n_heads,
                  int n_kv_heads, int head_dim, float* scores, Timings* t) {
    const int n_rep = n_heads / n_kv_heads;
    const float scale = 1.0f / std::sqrt((float)head_dim);
    for (int h = 0; h < n_heads; ++h) {
        const int hv = h / n_rep;
        for (int i = 0; i < seq; ++i) {
            const float* qi = q + (i * n_heads + h) * head_dim;
            {
                ScopeTime st(t ? &t->attn : nullptr);
                for (int j = 0; j <= i; ++j) {
                    const float* kj = k + (j * n_kv_heads + hv) * head_dim;
                    scores[j] = scale * dot_f32(qi, kj, head_dim, true);
                }
            }
            {
                ScopeTime st(t ? &t->softmax : nullptr);
                softmax(scores, i + 1);
            }
            ScopeTime st(t ? &t->attn : nullptr);
            float* oi = out + (i * n_heads + h) * head_dim;
            std::memset(oi, 0, (size_t)head_dim * sizeof(float));
            for (int j = 0; j <= i; ++j) {
                const float* vj = v + (j * n_kv_heads + hv) * head_dim;
                const float a = scores[j];
                for (int d = 0; d < head_dim; ++d) oi[d] += a * vj[d];
            }
        }
    }
}

}  // namespace tinf
