#include "gguf.hpp"
#include "model.hpp"
#include "sampler.hpp"
#include <cstring>
#include <cstdio>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

using namespace tinf;

static std::vector<uint8_t> as_bytes(const std::vector<float>& v) {
    std::vector<uint8_t> b(v.size() * 4);
    std::memcpy(b.data(), v.data(), b.size());
    return b;
}

static Tensor desc(const std::string& name, int ndim, int64_t a, int64_t b = 1) {
    Tensor t;
    t.name = name;
    t.dtype = DType::F32;
    t.ndim = ndim;
    t.shape[0] = a;
    t.shape[1] = b;
    return t;
}

static std::string write_tiny(const std::string& path, uint32_t seed) {
    const int n_vocab = 64, n_embd = 32, n_ff = 64, n_layer = 2, n_head = 4, n_kv = 2;
    const int head_dim = n_embd / n_head;  // 8
    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(0.0f, 0.02f);
    auto rnd = [&](int n) {
        std::vector<float> v(n);
        for (float& x : v) x = dist(rng);
        return v;
    };

    std::unordered_map<std::string, std::string> ms;
    std::unordered_map<std::string, uint64_t> mu;
    std::unordered_map<std::string, float> mf;
    ms["general.architecture"] = "llama";
    ms["tokenizer.ggml.model"] = "llama";
    mu["llama.vocab_size"] = n_vocab;
    mu["llama.embedding_length"] = n_embd;
    mu["llama.feed_forward_length"] = n_ff;
    mu["llama.block_count"] = n_layer;
    mu["llama.attention.head_count"] = n_head;
    mu["llama.attention.head_count_kv"] = n_kv;
    mu["llama.context_length"] = 128;
    mu["llama.rope.dimension_count"] = head_dim;
    mf["llama.rope.freq_base"] = 10000.0f;
    mf["llama.attention.layer_norm_rms_epsilon"] = 1e-5f;
    mu["tokenizer.ggml.bos_token_id"] = 1;
    mu["tokenizer.ggml.eos_token_id"] = 2;

    std::vector<std::string> toks(n_vocab);
    for (int i = 0; i < n_vocab; ++i) toks[i] = "t" + std::to_string(i);

    std::vector<Tensor> td;
    std::vector<std::vector<uint8_t>> pay;
    auto add = [&](const Tensor& t, const std::vector<float>& v) {
        td.push_back(t);
        pay.push_back(as_bytes(v));
    };

    add(desc("token_embd.weight", 2, n_embd, n_vocab), rnd(n_embd * n_vocab));
    add(desc("output_norm.weight", 1, n_embd), rnd(n_embd));
    add(desc("output.weight", 2, n_embd, n_vocab), rnd(n_embd * n_vocab));
    for (int i = 0; i < n_layer; ++i) {
        std::string p = "blk." + std::to_string(i) + ".";
        add(desc(p + "attn_norm.weight", 1, n_embd), rnd(n_embd));
        add(desc(p + "attn_q.weight", 2, n_embd, n_head * head_dim), rnd(n_embd * n_head * head_dim));
        add(desc(p + "attn_k.weight", 2, n_embd, n_kv * head_dim), rnd(n_embd * n_kv * head_dim));
        add(desc(p + "attn_v.weight", 2, n_embd, n_kv * head_dim), rnd(n_embd * n_kv * head_dim));
        add(desc(p + "attn_output.weight", 2, n_head * head_dim, n_embd), rnd(n_head * head_dim * n_embd));
        add(desc(p + "ffn_norm.weight", 1, n_embd), rnd(n_embd));
        add(desc(p + "ffn_gate.weight", 2, n_embd, n_ff), rnd(n_embd * n_ff));
        add(desc(p + "ffn_up.weight", 2, n_embd, n_ff), rnd(n_embd * n_ff));
        add(desc(p + "ffn_down.weight", 2, n_ff, n_embd), rnd(n_ff * n_embd));
    }
    write_gguf(path, ms, mu, mf, toks, td, pay);
    return path;
}

static std::vector<int> gen(const std::string& path, const RunConfig& cfg, const std::vector<int>& prompt,
                            int n_predict) {
    Model m(path, cfg);
    std::vector<float> logits(m.n_vocab());
    m.prefill(prompt.data(), (int)prompt.size(), logits.data());
    std::vector<int> out;
    for (int i = 0; i < n_predict; ++i) {
        int id = sample_greedy(logits.data(), m.n_vocab());
        out.push_back(id);
        m.decode(id, logits.data());
    }
    return out;
}

// Twenty fixed prompts (token ids) — the correctness gate corpus.
static std::vector<std::vector<int>> prompts20() {
    std::vector<std::vector<int>> p;
    for (int i = 0; i < 20; ++i) {
        std::vector<int> seq = {1, 3 + (i % 20), 5 + (i * 3) % 30, 7};
        p.push_back(seq);
    }
    return p;
}

int main() {
    const std::string path = "tiny_ci.gguf";
    write_tiny(path, 12345);
    const auto prompts = prompts20();
    int hw = 2;

    RunConfig v0 = stage_preset("v0", hw);
    v0.max_seq = 64;
    std::vector<std::vector<int>> gold;
    gold.reserve(prompts.size());
    for (const auto& pr : prompts) gold.push_back(gen(path, v0, pr, 8));

    const char* stages[] = {"v1", "v2", "v4"};
    for (const char* st : stages) {
        RunConfig c = stage_preset(st, hw);
        c.max_seq = 64;
        for (int i = 0; i < (int)prompts.size(); ++i) {
            auto got = gen(path, c, prompts[i], 8);
            if (got != gold[i]) {
                std::fprintf(stderr, "token mismatch stage %s prompt %d\n", st, i);
                return 1;
            }
        }
        std::printf("stage %s matches v0 on 20 prompts\n", st);
    }
    std::printf("test_correctness ok\n");
    return 0;
}
