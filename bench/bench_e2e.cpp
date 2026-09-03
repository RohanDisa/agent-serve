#include "model.hpp"
#include "sampler.hpp"
#include "tokenizer.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

using namespace tinf;

static double percentile(std::vector<double> v, double p) {
    if (v.empty()) return 0;
    std::sort(v.begin(), v.end());
    double idx = p * (v.size() - 1);
    size_t i = (size_t)idx;
    double f = idx - i;
    if (i + 1 >= v.size()) return v.back();
    return v[i] * (1 - f) + v[i + 1] * f;
}

static double median(std::vector<double> v) {
    return percentile(std::move(v), 0.5);
}

struct RunOut {
    double prefill_s = 0;
    double decode_s = 0;
    int n_in = 0;
    int n_out = 0;
    std::vector<double> tok_ms;
    Timings ops;
};

static RunOut once(Model& m, const std::vector<int>& prompt, int n_predict) {
    RunOut r;
    m.reset();
    m.timings().clear();
    std::vector<float> logits(m.n_vocab());
    double t0 = now_s();
    m.prefill(prompt.data(), (int)prompt.size(), logits.data());
    r.prefill_s = now_s() - t0;
    r.n_in = (int)prompt.size();
    std::mt19937 rng(42);
    t0 = now_s();
    for (int i = 0; i < n_predict; ++i) {
        double u = now_s();
        int id = sample_greedy(logits.data(), m.n_vocab());
        m.decode(id, logits.data());
        r.tok_ms.push_back((now_s() - u) * 1e3);
    }
    r.decode_s = now_s() - t0;
    r.n_out = n_predict;
    r.ops = m.timings();
    return r;
}

int main(int argc, char** argv) {
    std::string model_path, prompt = "The capital of France is", stage = "v5";
    int n_predict = 64, n_in = 0, threads = -1;
    int warmup = 3, runs = 10, max_seq = 4096;
    std::string out_path;
    bool pin = false;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&] { return std::string(argv[++i]); };
        if (a == "--model") model_path = next();
        else if (a == "--prompt") prompt = next();
        else if (a == "--stage") stage = next();
        else if (a == "--n-predict") n_predict = std::stoi(next());
        else if (a == "--n-in") n_in = std::stoi(next());
        else if (a == "--threads") threads = std::stoi(next());
        else if (a == "--warmup") warmup = std::stoi(next());
        else if (a == "--runs") runs = std::stoi(next());
        else if (a == "--max-seq") max_seq = std::stoi(next());
        else if (a == "--out") out_path = next();
        else if (a == "--pin") pin = true;
    }
    if (model_path.empty()) {
        std::fprintf(stderr, "usage: bench_e2e --model PATH [--stage v0..v5] [--n-predict N] [--out file.json]\n");
        return 1;
    }

    int hw = (int)std::thread::hardware_concurrency();
    RunConfig cfg = stage_preset(stage, hw > 0 ? hw : 4);
    if (threads > 0) cfg.n_threads = threads;
    cfg.max_seq = max_seq;
    cfg.temperature = 0;
    cfg.seed = 42;
    cfg.pin_threads = pin;

    Model model(model_path, cfg);
    Tokenizer tok(model.gguf());
    std::vector<int> ids = tok.encode(prompt, true);
    if (n_in > 0) {
        ids.resize(n_in, ids.empty() ? model.bos() : ids.back());
        if (ids.empty()) ids.push_back(model.bos());
    }
    std::fprintf(stderr, "%s\n", model.describe().c_str());
    std::fprintf(stderr, "prompt_tokens=%d n_predict=%d warmup=%d runs=%d\n", (int)ids.size(), n_predict,
                 warmup, runs);

    for (int w = 0; w < warmup; ++w) once(model, ids, std::min(8, n_predict));

    std::vector<double> prefill_tps, decode_tps, ttft;
    std::vector<double> p50s, p95s, p99s;
    Timings last;
    size_t wbytes = model.weight_bytes();
    for (int r = 0; r < runs; ++r) {
        RunOut o = once(model, ids, n_predict);
        last = o.ops;
        prefill_tps.push_back(o.n_in / o.prefill_s);
        decode_tps.push_back(o.n_out / o.decode_s);
        ttft.push_back(o.prefill_s);
        p50s.push_back(percentile(o.tok_ms, 0.50));
        p95s.push_back(percentile(o.tok_ms, 0.95));
        p99s.push_back(percentile(o.tok_ms, 0.99));
        std::fprintf(stderr, "  run %d  prefill %.1f tok/s  decode %.1f tok/s  ttft %.3fs\n", r,
                     prefill_tps.back(), decode_tps.back(), ttft.back());
    }

    double d_med = median(decode_tps);
    double p_med = median(prefill_tps);
    // arithmetic intensity (approx): decode GEMV is 2 FLOP / byte_of_weight
    double decode_gb_s = (double)wbytes * d_med / 1e9;
    double flop_per_tok = 2.0 * (double)wbytes;  // F32-equivalent; overestimate for quant
    // better: for each weight byte we still do ~2 FLOPs per F32 element. Report GB/s from bytes read.
    double tot = last.total();
    auto pct = [&](double x) { return tot > 0 ? 100.0 * x / tot : 0; };

    char buf[4096];
    std::snprintf(
        buf, sizeof(buf),
        "{\n"
        "  \"bench\": \"e2e\",\n"
        "  \"stage\": \"%s\",\n"
        "  \"threads\": %d,\n"
        "  \"simd\": %s,\n"
        "  \"blocked\": %s,\n"
        "  \"kv_layout\": \"%s\",\n"
        "  \"n_in\": %d,\n"
        "  \"n_predict\": %d,\n"
        "  \"weight_bytes\": %zu,\n"
        "  \"prefill_tok_s_median\": %.4f,\n"
        "  \"prefill_tok_s_q1\": %.4f,\n"
        "  \"prefill_tok_s_q3\": %.4f,\n"
        "  \"decode_tok_s_median\": %.4f,\n"
        "  \"decode_tok_s_q1\": %.4f,\n"
        "  \"decode_tok_s_q3\": %.4f,\n"
        "  \"ttft_s_median\": %.6f,\n"
        "  \"decode_tok_ms_p50\": %.4f,\n"
        "  \"decode_tok_ms_p95\": %.4f,\n"
        "  \"decode_tok_ms_p99\": %.4f,\n"
        "  \"decode_gb_s\": %.4f,\n"
        "  \"ops_pct\": {\"matmul\": %.2f, \"attn\": %.2f, \"rmsnorm\": %.2f, \"rope\": %.2f, "
        "\"softmax\": %.2f, \"sample\": %.2f}\n"
        "}\n",
        stage.c_str(), cfg.n_threads, cfg.use_simd ? "true" : "false", cfg.use_blocking ? "true" : "false",
        cfg.kv_layout == KVLayout::HeadMajor ? "head" : "seq", (int)ids.size(), n_predict, wbytes, p_med,
        percentile(prefill_tps, 0.25), percentile(prefill_tps, 0.75), d_med, percentile(decode_tps, 0.25),
        percentile(decode_tps, 0.75), median(ttft), median(p50s), median(p95s), median(p99s), decode_gb_s,
        pct(last.matmul), pct(last.attn), pct(last.rmsnorm), pct(last.rope), pct(last.softmax),
        pct(last.sample));

    std::fputs(buf, stdout);
    if (!out_path.empty()) {
        std::ofstream f(out_path);
        f << buf;
    }
    (void)flop_per_tok;
    return 0;
}
