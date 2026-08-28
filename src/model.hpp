#pragma once

#include "gguf.hpp"
#include "kvcache.hpp"
#include "attention.hpp"

namespace tinf {

struct LayerWeights {
    const Tensor* attn_norm = nullptr;
    const Tensor* wq = nullptr;
    const Tensor* wk = nullptr;
    const Tensor* wv = nullptr;
    const Tensor* wo = nullptr;
    const Tensor* bq = nullptr;
    const Tensor* bk = nullptr;
    const Tensor* bv = nullptr;
    const Tensor* ffn_norm = nullptr;
    const Tensor* gate = nullptr;
    const Tensor* up = nullptr;
    const Tensor* down = nullptr;
};

class Model {
public:
    Model(const std::string& path, const RunConfig& cfg);

    // Prefill: consume tokens[0..n), write KV for those positions, write logits of last token.
    void prefill(const int* tokens, int n, float* logits);
    // Decode: append one token at n_pos_, write logits.
    void decode(int token, float* logits);

    void reset();

    int n_vocab() const { return n_vocab_; }
    int n_embd() const { return n_embd_; }
    int n_layer() const { return n_layer_; }
    int n_pos() const { return n_pos_; }
    int bos() const { return bos_; }
    int eos() const { return eos_; }
    size_t weight_bytes() const { return gguf_.weight_bytes(); }
    const GGUF& gguf() const { return gguf_; }
    const RunConfig& cfg() const { return cfg_; }
    RunConfig& cfg() { return cfg_; }
    const Timings& timings() const { return timings_; }
    Timings& timings() { return timings_; }
    ThreadPool& pool() { return pool_; }
    KVCache& kv() { return kv_; }

    std::string describe() const;

private:
    void apply_bias(float* x, const Tensor* b, int n);
    void layer_decode(int layer, float* x);
    void layer_prefill(int layer, float* x, int seq);

    GGUF gguf_;
    RunConfig cfg_;
    ThreadPool pool_;
    Timings timings_;
    RopeParams rope_;
    std::vector<float> rope_freq_;
    KVCache kv_;
    std::vector<LayerWeights> layers_;
    const Tensor* tok_embd_ = nullptr;
    const Tensor* output_norm_ = nullptr;
    const Tensor* output_ = nullptr;

    int n_vocab_ = 0;
    int n_embd_ = 0;
    int n_ff_ = 0;
    int n_layer_ = 0;
    int n_head_ = 0;
    int n_kv_head_ = 0;
    int head_dim_ = 0;
    int bos_ = 1;
    int eos_ = 2;
    float rms_eps_ = 1e-5f;
    int n_pos_ = 0;

    AlignedBuf x_;      // [max_seq * n_embd]  (decode uses first n_embd)
    AlignedBuf xb_;
    AlignedBuf q_;
    AlignedBuf k_;
    AlignedBuf v_;
    AlignedBuf hb_;     // [max_seq * n_ff] or [n_ff]
    AlignedBuf hb2_;
    AlignedBuf att_;    // scores [max_seq]
    AlignedBuf logits_;
};

}  // namespace tinf
