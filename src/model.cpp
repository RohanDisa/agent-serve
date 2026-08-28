#include "model.hpp"
#include "layers.hpp"
#include "matmul.hpp"
#include "quant.hpp"
#include <sstream>

namespace tinf {

Model::Model(const std::string& path, const RunConfig& cfg)
    : gguf_(path), cfg_(cfg), pool_(cfg.n_threads, cfg.pin_threads) {
    const std::string a = gguf_.arch();
    auto key = [&](const char* f) { return gguf_.kv_arch(f); };

    n_vocab_ = (int)gguf_.u64(key("vocab_size"), 0);
    if (!n_vocab_) n_vocab_ = (int)gguf_.arr_str("tokenizer.ggml.tokens").size();
    n_embd_ = (int)gguf_.u64(key("embedding_length"));
    n_ff_ = (int)gguf_.u64(key("feed_forward_length"));
    n_layer_ = (int)gguf_.u64(key("block_count"));
    n_head_ = (int)gguf_.u64(key("attention.head_count"));
    n_kv_head_ = (int)gguf_.u64(key("attention.head_count_kv"), n_head_);
    int rope_dim = (int)gguf_.u64(key("rope.dimension_count"), 0);
    head_dim_ = n_head_ ? n_embd_ / n_head_ : 0;
    if (rope_dim) head_dim_ = rope_dim;  // usually equals n_embd/n_head
    if (n_head_) head_dim_ = n_embd_ / n_head_;
    rms_eps_ = gguf_.f32(key("attention.layer_norm_rms_epsilon"), 1e-5f);
    rope_.theta = gguf_.f32(key("rope.freq_base"), 10000.0f);
    rope_.n_rot = head_dim_;
    std::string rtype = gguf_.str(key("rope.scaling.type"), "");
    if (rtype == "llama3") {
        rope_.type = RopeType::Llama3;
        rope_.scale = gguf_.f32(key("rope.scaling.factor"), 32.0f);
        rope_.low_freq_f = gguf_.f32(key("rope.scaling.low_freq_factor"), 1.0f);
        rope_.high_freq_f = gguf_.f32(key("rope.scaling.high_freq_factor"), 4.0f);
        rope_.orig_ctx = (float)gguf_.u64(key("rope.scaling.original_context_length"), 8192);
    } else {
        rope_.type = RopeType::Standard;
    }
    rope_freqs(rope_freq_, rope_);

    bos_ = (int)gguf_.u64("tokenizer.ggml.bos_token_id", 1);
    eos_ = (int)gguf_.u64("tokenizer.ggml.eos_token_id", 2);

    tok_embd_ = &gguf_.must("token_embd.weight");
    output_norm_ = &gguf_.must("output_norm.weight");
    output_ = gguf_.find("output.weight");
    if (!output_) output_ = tok_embd_;
    if (!n_vocab_) n_vocab_ = (int)tok_embd_->rows();

    layers_.resize(n_layer_);
    for (int i = 0; i < n_layer_; ++i) {
        auto& L = layers_[i];
        std::string p = "blk." + std::to_string(i) + ".";
        L.attn_norm = &gguf_.must(p + "attn_norm.weight");
        L.wq = &gguf_.must(p + "attn_q.weight");
        L.wk = &gguf_.must(p + "attn_k.weight");
        L.wv = &gguf_.must(p + "attn_v.weight");
        L.wo = &gguf_.must(p + "attn_output.weight");
        L.bq = gguf_.find(p + "attn_q.bias");
        L.bk = gguf_.find(p + "attn_k.bias");
        L.bv = gguf_.find(p + "attn_v.bias");
        L.ffn_norm = &gguf_.must(p + "ffn_norm.weight");
        L.gate = &gguf_.must(p + "ffn_gate.weight");
        L.up = &gguf_.must(p + "ffn_up.weight");
        L.down = &gguf_.must(p + "ffn_down.weight");
    }

    int max_seq = cfg_.max_seq;
    kv_.init(n_layer_, n_kv_head_, head_dim_, max_seq, cfg_.kv_layout);
    x_.reset((size_t)max_seq * n_embd_);
    xb_.reset((size_t)max_seq * n_embd_);
    q_.reset((size_t)max_seq * n_head_ * head_dim_);
    k_.reset((size_t)max_seq * n_kv_head_ * head_dim_);
    v_.reset((size_t)max_seq * n_kv_head_ * head_dim_);
    hb_.reset((size_t)max_seq * n_ff_);
    hb2_.reset((size_t)max_seq * n_ff_);
    att_.reset(max_seq);
    logits_.reset(n_vocab_);
}

void Model::reset() {
    n_pos_ = 0;
    kv_.clear();
}

void Model::apply_bias(float* x, const Tensor* b, int n) {
    if (!b) return;
    const float* bp = (const float*)b->data;
    for (int i = 0; i < n; ++i) x[i] += bp[i];
}

std::string Model::describe() const {
    std::ostringstream o;
    o << "arch=" << gguf_.arch()
      << " layers=" << n_layer_
      << " dim=" << n_embd_
      << " heads=" << n_head_ << "/" << n_kv_head_
      << " ff=" << n_ff_
      << " vocab=" << n_vocab_
      << " weights=" << (weight_bytes() / 1e6) << " MB"
      << " stage=" << cfg_.stage
      << " threads=" << cfg_.n_threads
      << " simd=" << (cfg_.use_simd ? "on" : "off")
      << " block=" << (cfg_.use_blocking ? "on" : "off")
      << " kv=" << (cfg_.kv_layout == KVLayout::HeadMajor ? "head" : "seq");
    return o.str();
}

void Model::layer_decode(int li, float* x) {
    const LayerWeights& L = layers_[li];
    const Exec ex{&cfg_, &pool_, &timings_};
    const int D = n_embd_;
    const int H = n_head_ * head_dim_;
    const int KV = n_kv_head_ * head_dim_;

    {
        ScopeTime st(&timings_.rmsnorm);
        rmsnorm(xb_.p, x, (const float*)L.attn_norm->data, D, rms_eps_);
    }
    matmul(q_.p, xb_.p, *L.wq, 1, ex);
    matmul(k_.p, xb_.p, *L.wk, 1, ex);
    matmul(v_.p, xb_.p, *L.wv, 1, ex);
    apply_bias(q_.p, L.bq, H);
    apply_bias(k_.p, L.bk, KV);
    apply_bias(v_.p, L.bv, KV);
    apply_rope(q_.p, n_head_, head_dim_, n_pos_, rope_freq_.data(), &timings_);
    apply_rope(k_.p, n_kv_head_, head_dim_, n_pos_, rope_freq_.data(), &timings_);
    kv_.store(li, n_pos_, k_.p, v_.p);

    attn_decode(xb_.p, q_.p, kv_, li, n_head_, n_kv_head_, head_dim_, n_pos_ + 1, att_.p, &timings_);
    matmul(q_.p, xb_.p, *L.wo, 1, ex);  // reuse q_ as attn out
    add_inplace(x, q_.p, D);

    {
        ScopeTime st(&timings_.rmsnorm);
        rmsnorm(xb_.p, x, (const float*)L.ffn_norm->data, D, rms_eps_);
    }
    matmul(hb_.p, xb_.p, *L.gate, 1, ex);
    matmul(hb2_.p, xb_.p, *L.up, 1, ex);
    silu_mul(hb_.p, hb2_.p, n_ff_);
    matmul(xb_.p, hb_.p, *L.down, 1, ex);
    add_inplace(x, xb_.p, D);
}

void Model::layer_prefill(int li, float* x, int seq) {
    const LayerWeights& L = layers_[li];
    const Exec ex{&cfg_, &pool_, &timings_};
    const int D = n_embd_;
    const int H = n_head_ * head_dim_;
    const int KV = n_kv_head_ * head_dim_;

    for (int i = 0; i < seq; ++i) {
        ScopeTime st(&timings_.rmsnorm);
        rmsnorm(xb_.p + i * D, x + i * D, (const float*)L.attn_norm->data, D, rms_eps_);
    }
    matmul(q_.p, xb_.p, *L.wq, seq, ex);
    matmul(k_.p, xb_.p, *L.wk, seq, ex);
    matmul(v_.p, xb_.p, *L.wv, seq, ex);
    for (int i = 0; i < seq; ++i) {
        apply_bias(q_.p + i * H, L.bq, H);
        apply_bias(k_.p + i * KV, L.bk, KV);
        apply_bias(v_.p + i * KV, L.bv, KV);
    }
    apply_rope_many(q_.p, seq, n_head_, head_dim_, n_pos_, rope_freq_.data(), &timings_);
    apply_rope_many(k_.p, seq, n_kv_head_, head_dim_, n_pos_, rope_freq_.data(), &timings_);
    for (int i = 0; i < seq; ++i)
        kv_.store(li, n_pos_ + i, k_.p + i * KV, v_.p + i * KV);

    attn_prefill(xb_.p, q_.p, k_.p, v_.p, seq, n_head_, n_kv_head_, head_dim_, att_.p, &timings_);
    matmul(q_.p, xb_.p, *L.wo, seq, ex);
    for (int i = 0; i < seq; ++i) add_inplace(x + i * D, q_.p + i * D, D);

    for (int i = 0; i < seq; ++i) {
        ScopeTime st(&timings_.rmsnorm);
        rmsnorm(xb_.p + i * D, x + i * D, (const float*)L.ffn_norm->data, D, rms_eps_);
    }
    matmul(hb_.p, xb_.p, *L.gate, seq, ex);
    matmul(hb2_.p, xb_.p, *L.up, seq, ex);
    silu_mul(hb_.p, hb2_.p, seq * n_ff_);
    matmul(xb_.p, hb_.p, *L.down, seq, ex);
    for (int i = 0; i < seq; ++i) add_inplace(x + i * D, xb_.p + i * D, D);
}

void Model::prefill(const int* tokens, int n, float* logits) {
    if (n <= 0) return;
    if (n_pos_ + n > cfg_.max_seq) throw std::runtime_error("prefill exceeds max_seq");
    for (int i = 0; i < n; ++i) dequant_row(x_.p + i * n_embd_, *tok_embd_, tokens[i]);
    for (int li = 0; li < n_layer_; ++li) layer_prefill(li, x_.p, n);
    {
        ScopeTime st(&timings_.rmsnorm);
        rmsnorm(xb_.p, x_.p + (n - 1) * n_embd_, (const float*)output_norm_->data, n_embd_, rms_eps_);
    }
    Exec ex{&cfg_, &pool_, &timings_};
    matmul(logits ? logits : logits_.p, xb_.p, *output_, 1, ex);
    n_pos_ += n;
}

void Model::decode(int token, float* logits) {
    if (n_pos_ + 1 > cfg_.max_seq) throw std::runtime_error("decode exceeds max_seq");
    dequant_row(x_.p, *tok_embd_, token);
    for (int li = 0; li < n_layer_; ++li) layer_decode(li, x_.p);
    {
        ScopeTime st(&timings_.rmsnorm);
        rmsnorm(xb_.p, x_.p, (const float*)output_norm_->data, n_embd_, rms_eps_);
    }
    Exec ex{&cfg_, &pool_, &timings_};
    matmul(logits ? logits : logits_.p, xb_.p, *output_, 1, ex);
    n_pos_ += 1;
}

}  // namespace tinf
