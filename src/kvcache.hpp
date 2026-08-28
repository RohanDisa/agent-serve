#pragma once

#include "common.hpp"

namespace tinf {

class KVCache {
public:
    void init(int n_layers, int n_kv_heads, int head_dim, int max_seq, KVLayout layout);
    void clear();

    // Write k,v for one layer at position pos. k,v are [n_kv_heads, head_dim].
    void store(int layer, int pos, const float* k, const float* v);

    // Pointer to K of (layer, head, pos) — head_dim contiguous values.
    const float* k_at(int layer, int kv_head, int pos) const;
    const float* v_at(int layer, int kv_head, int pos) const;

    KVLayout layout() const { return layout_; }
    int max_seq() const { return max_seq_; }

private:
    int n_layers_ = 0, n_kv_heads_ = 0, head_dim_ = 0, max_seq_ = 0;
    KVLayout layout_ = KVLayout::SeqMajor;
    AlignedBuf k_;
    AlignedBuf v_;
    size_t layer_stride_ = 0;

    size_t offset(int layer, int kv_head, int pos) const;
};

}  // namespace tinf
