#include "matmul.hpp"
#include "quant.hpp"
#include <algorithm>
#include <cstdio>
#include <string>
#include <vector>

using namespace tinf;

int main(int argc, char** argv) {
    int M = 256, N = 2048, K = 2048;
    int threads = (int)std::thread::hardware_concurrency();
    bool simd = true, blocked = true;
    int warmup = 3, runs = 10;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&] { return std::atoi(argv[++i]); };
        if (a == "--m") M = next();
        else if (a == "--n") N = next();
        else if (a == "--k") K = next();
        else if (a == "--threads") threads = next();
        else if (a == "--no-simd") simd = false;
        else if (a == "--no-block") blocked = false;
        else if (a == "--runs") runs = next();
    }
    if (threads < 1) threads = 1;

    std::vector<float> A((size_t)M * K, 0.01f), B((size_t)N * K, 0.02f), C((size_t)M * N, 0);
    ThreadPool pool(threads, false);
    for (int w = 0; w < warmup; ++w) gemm_f32_bench(C.data(), A.data(), B.data(), M, N, K, simd, blocked, &pool);

    std::vector<double> gflops;
    for (int r = 0; r < runs; ++r) {
        double t0 = now_s();
        gemm_f32_bench(C.data(), A.data(), B.data(), M, N, K, simd, blocked, &pool);
        double t = now_s() - t0;
        double flops = 2.0 * M * N * (double)K;
        gflops.push_back(flops / t / 1e9);
        std::fprintf(stderr, "  run %d  %.2f GFLOP/s  (%.3fs)\n", r, gflops.back(), t);
    }
    std::sort(gflops.begin(), gflops.end());
    double med = gflops[gflops.size() / 2];
    double q1 = gflops[gflops.size() / 4];
    double q3 = gflops[(3 * gflops.size()) / 4];
    double sink = C[0] + C[(size_t)M * N / 2] + C.back();
    std::printf(
        "{\n  \"bench\": \"gemm_f32\",\n  \"M\": %d, \"N\": %d, \"K\": %d,\n"
        "  \"threads\": %d, \"simd\": %s, \"blocked\": %s,\n"
        "  \"gflop_s_median\": %.4f, \"gflop_s_q1\": %.4f, \"gflop_s_q3\": %.4f,\n"
        "  \"sink\": %.6f\n}\n",
        M, N, K, threads, simd ? "true" : "false", blocked ? "true" : "false", med, q1, q3, sink);
    std::fprintf(stderr, "median %.2f GFLOP/s  IQR [%.2f, %.2f]\n", med, q1, q3);
    return 0;
}
