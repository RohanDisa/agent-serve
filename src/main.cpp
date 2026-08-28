#include "model.hpp"
#include "sampler.hpp"
#include "tokenizer.hpp"
#include <iostream>
#include <sstream>

using namespace tinf;

static void usage() {
    std::fprintf(stderr,
                 "tinferno — from-scratch Llama inference (measurement first)\n"
                 "\n"
                 "usage: tinferno --model PATH [options]\n"
                 "  --model PATH         GGUF file (F32 / F16 / Q8_0 / Q4_0)\n"
                 "  --prompt TEXT        prompt string\n"
                 "  --tokens 1,2,3       skip tokenizer; comma-separated ids\n"
                 "  --n-predict N        tokens to generate (default 64)\n"
                 "  --stage v0..v5       implementation stage preset (default v2)\n"
                 "  --threads N          override thread count\n"
                 "  --temp T             temperature; 0 = greedy (default 0)\n"
                 "  --top-p P            nucleus sampling (default 0.9)\n"
                 "  --seed N             rng seed (default 42)\n"
                 "  --max-seq N          KV cache length (default 4096)\n"
                 "  --kv-layout seq|head override KV layout\n"
                 "  --pin                pin threads to CPUs 0..N-1\n"
                 "  --profile            print operator breakdown\n"
                 "  --no-bos             do not prepend BOS\n");
}

static std::vector<int> parse_tokens(const std::string& s) {
    std::vector<int> v;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, ',')) {
        if (!item.empty()) v.push_back(std::stoi(item));
    }
    return v;
}

int main(int argc, char** argv) {
    std::string model_path, prompt, token_str, stage = "v2";
    int n_predict = 64, threads = -1, seed = 42, max_seq = 4096;
    float temp = 0, topp = 0.9f;
    bool pin = false, profile = false, add_bos = true;
    std::string kv_layout;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) die("missing value for " + a);
            return argv[++i];
        };
        if (a == "--model") model_path = next();
        else if (a == "--prompt") prompt = next();
        else if (a == "--tokens") token_str = next();
        else if (a == "--n-predict") n_predict = std::stoi(next());
        else if (a == "--stage") stage = next();
        else if (a == "--threads") threads = std::stoi(next());
        else if (a == "--temp") temp = std::stof(next());
        else if (a == "--top-p") topp = std::stof(next());
        else if (a == "--seed") seed = std::stoi(next());
        else if (a == "--max-seq") max_seq = std::stoi(next());
        else if (a == "--kv-layout") kv_layout = next();
        else if (a == "--pin") pin = true;
        else if (a == "--profile") profile = true;
        else if (a == "--no-bos") add_bos = false;
        else if (a == "-h" || a == "--help") {
            usage();
            return 0;
        } else {
            die("unknown flag " + a);
        }
    }
    if (model_path.empty()) {
        usage();
        return 1;
    }

    int hw = (int)std::thread::hardware_concurrency();
    if (hw <= 0) hw = 4;
    RunConfig cfg = stage_preset(stage, hw);
    if (threads > 0) cfg.n_threads = threads;
    cfg.n_predict = n_predict;
    cfg.temperature = temp;
    cfg.top_p = topp;
    cfg.seed = seed;
    cfg.max_seq = max_seq;
    cfg.pin_threads = pin;
    cfg.profile = profile;
    if (kv_layout == "head") cfg.kv_layout = KVLayout::HeadMajor;
    if (kv_layout == "seq") cfg.kv_layout = KVLayout::SeqMajor;

    try {
        Model model(model_path, cfg);
        Tokenizer tok(model.gguf());
        std::fprintf(stderr, "%s\n", model.describe().c_str());

        std::vector<int> prompt_ids;
        if (!token_str.empty()) prompt_ids = parse_tokens(token_str);
        else prompt_ids = tok.encode(prompt, add_bos);
        if (prompt_ids.empty()) die("empty prompt");

        std::vector<float> logits(model.n_vocab());
        model.timings().clear();
        double t0 = now_s();
        model.prefill(prompt_ids.data(), (int)prompt_ids.size(), logits.data());
        double t_prefill = now_s() - t0;

        std::mt19937 rng(seed);
        std::vector<int> gen;
        t0 = now_s();
        for (int i = 0; i < n_predict; ++i) {
            double ts = now_s();
            int id = sample_logits(logits.data(), model.n_vocab(), temp, topp, rng);
            model.timings().sample += now_s() - ts;
            if (id == model.eos() && i > 0) break;
            gen.push_back(id);
            std::string piece = tok.decode_one(id);
            std::fputs(piece.c_str(), stdout);
            std::fflush(stdout);
            model.decode(id, logits.data());
        }
        double t_decode = now_s() - t0;
        std::fputc('\n', stdout);

        int n_in = (int)prompt_ids.size();
        int n_out = (int)gen.size();
        std::fprintf(stderr, "\nprefill %d tok in %.3fs  (%.1f tok/s)  TTFT %.3fs\n", n_in, t_prefill,
                     n_in / t_prefill, t_prefill);
        std::fprintf(stderr, "decode  %d tok in %.3fs  (%.1f tok/s)\n", n_out, t_decode,
                     n_out / std::max(t_decode, 1e-9));
        if (profile) {
            double tot = model.timings().total();
            auto pct = [&](double x) { return tot > 0 ? 100.0 * x / tot : 0; };
            const Timings& t = model.timings();
            std::fprintf(stderr,
                         "ops  matmul %.1f%%  attn %.1f%%  rmsnorm %.1f%%  rope %.1f%%  softmax %.1f%%  "
                         "sample %.1f%%\n",
                         pct(t.matmul), pct(t.attn), pct(t.rmsnorm), pct(t.rope), pct(t.softmax),
                         pct(t.sample));
        }
    } catch (const std::exception& e) {
        die(e.what());
    }
    return 0;
}
