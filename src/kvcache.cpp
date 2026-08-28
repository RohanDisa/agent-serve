#include "kvcache.hpp"

namespace tinf {

void KVCache::init(int n_layers, int n_kv_heads, int head_dim, int max_seq, KVLayout layout) {
    n_layers_ = n_layers;
    n_kv_heads_ = n_kv_heads;
    head_dim_ = head_dim;
    max_seq_ = max_seq;
    layout_ = layout;
    layer_stride_ = (size_t)n_kv_heads * max_seq * head_dim;
    k_.reset(n_layers * layer_stride_);
    v_.reset(n_layers * layer_stride_);
}

void KVCache::clear() {
    if (k_.p) std::memset(k_.p, 0, k_.n * sizeof(float));
    if (v_.p) std::memset(v_.p, 0, v_.n * sizeof(float));
}

size_t KVCache::offset(int layer, int kv_head, int pos) const {
    // SeqMajor: [layer][pos][kv_head][dim]
    // HeadMajor: [layer][kv_head][pos][dim]  — each head's timeline is a contiguous slab
    if (layout_ == KVLayout::HeadMajor)
        return (size_t)layer * layer_stride_ +
               ((size_t)kv_head * max_seq_ + pos) * head_dim_;
    return (size_t)layer * layer_stride_ +
           ((size_t)pos * n_kv_heads_ + kv_head) * head_dim_;
}

void KVCache::store(int layer, int pos, const float* k, const float* v) {
    for (int h = 0; h < n_kv_heads_; ++h) {
        std::memcpy(k_.p + offset(layer, h, pos), k + h * head_dim_, (size_t)head_dim_ * sizeof(float));
        std::memcpy(v_.p + offset(layer, h, pos), v + h * head_dim_, (size_t)head_dim_ * sizeof(float));
    }
}

const float* KVCache::k_at(int layer, int kv_head, int pos) const {
    return k_.p + offset(layer, kv_head, pos);
}
const float* KVCache::v_at(int layer, int kv_head, int pos) const {
    return v_.p + offset(layer, kv_head, pos);
}

}  // namespace tinf
