#include "layers.hpp"
#include "quant.hpp"
#include "matmul.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

using namespace tinf;

static int g_fail = 0;

static void expect(bool cond, const char* msg) {
    if (!cond) {
        std::fprintf(stderr, "FAIL: %s\n", msg);
        ++g_fail;
    }
}

static void expect_near(float a, float b, float eps, const char* msg) {
    if (std::fabs(a - b) > eps) {
        std::fprintf(stderr, "FAIL: %s  got %.6g want %.6g\n", msg, a, b);
        ++g_fail;
    }
}

int main() {
    {
        float x[4] = {1, 2, 3, 4};
        float w[4] = {1, 1, 1, 1};
        float o[4];
        rmsnorm(o, x, w, 4, 1e-5f);
        double ss = 1 + 4 + 9 + 16;
        float inv = 1.0f / std::sqrt((float)(ss / 4) + 1e-5f);
        expect_near(o[0], 1 * inv, 1e-5f, "rmsnorm0");
        expect_near(o[3], 4 * inv, 1e-5f, "rmsnorm3");
    }
    {
        float x[3] = {1, 2, 3};
        softmax(x, 3);
        float s = x[0] + x[1] + x[2];
        expect_near(s, 1.0f, 1e-5f, "softmax sum");
        expect(x[2] > x[1] && x[1] > x[0], "softmax order");
    }
    {
        float g[2] = {0, 1};
        float u[2] = {2, 3};
        silu_mul(g, u, 2);
        expect_near(g[0], 0.0f, 1e-6f, "silu(0)*2");
        float silu1 = 1.0f / (1.0f + std::exp(-1.0f));
        expect_near(g[1], silu1 * 3, 1e-5f, "silu(1)*3");
    }
    {
        uint16_t h = f32_to_f16(1.5f);
        expect_near(f16_to_f32(h), 1.5f, 1e-3f, "f16 roundtrip 1.5");
    }
    {
        const int n = 64;
        std::vector<float> x(n), y(n);
        for (int i = 0; i < n; ++i) {
            x[i] = std::sin(0.1f * i);
            y[i] = std::cos(0.07f * i);
        }
        float d0 = dot_f32(x.data(), y.data(), n, false);
        float d1 = dot_f32(x.data(), y.data(), n, true);
        expect_near(d0, d1, 1e-4f, "dot simd vs scalar");
    }
    {
        const int n = 64;
        std::vector<float> x(n);
        for (int i = 0; i < n; ++i) x[i] = std::sin(0.3f * i);
        std::vector<block_q8_0> q(n / QK8_0);
        quant_q8_0_row(q.data(), x.data(), n);
        std::vector<float> y(n);
        dequant_q8_0_row(y.data(), q.data(), n);
        float err = 0;
        for (int i = 0; i < n; ++i) err = std::max(err, std::fabs(x[i] - y[i]));
        expect(err < 0.05f, "q8_0 roundtrip err");
        float ds = dot_q8_0_f32(x.data(), q.data(), n, false);
        float dq = dot_q8_0_f32(x.data(), q.data(), n, true);
        expect_near(ds, dq, 1e-3f, "q8_0 dot simd");
    }
    {
        const int n = 64;
        std::vector<float> x(n);
        for (int i = 0; i < n; ++i) x[i] = std::sin(0.3f * i);
        std::vector<block_q4_0> q(n / QK4_0);
        quant_q4_0_row(q.data(), x.data(), n);
        std::vector<float> y(n);
        dequant_q4_0_row(y.data(), q.data(), n);
        float err = 0;
        for (int i = 0; i < n; ++i) err = std::max(err, std::fabs(x[i] - y[i]));
        expect(err < 0.6f, "q4_0 roundtrip err");
        float ds = dot_q4_0_f32(x.data(), q.data(), n, false);
        float dq = dot_q4_0_f32(x.data(), q.data(), n, true);
        expect_near(ds, dq, 2e-2f, "q4_0 dot simd");
    }
    {
        const int M = 4, N = 5, K = 8;
        std::vector<float> A(M * K), W(N * K), C(M * N), ref(M * N);
        for (int i = 0; i < M * K; ++i) A[i] = 0.01f * (i + 1);
        for (int i = 0; i < N * K; ++i) W[i] = 0.02f * (i + 3);
        for (int m = 0; m < M; ++m)
            for (int n = 0; n < N; ++n) {
                float s = 0;
                for (int k = 0; k < K; ++k) s += A[m * K + k] * W[n * K + k];
                ref[m * N + n] = s;
            }
        ThreadPool pool(1, false);
        gemm_f32_bench(C.data(), A.data(), W.data(), M, N, K, false, false, &pool);
        for (int i = 0; i < M * N; ++i) expect_near(C[i], ref[i], 1e-5f, "gemm scalar");
        gemm_f32_bench(C.data(), A.data(), W.data(), M, N, K, true, false, &pool);
        for (int i = 0; i < M * N; ++i) expect_near(C[i], ref[i], 1e-4f, "gemm simd");
        gemm_f32_bench(C.data(), A.data(), W.data(), M, N, K, true, true, &pool);
        for (int i = 0; i < M * N; ++i) expect_near(C[i], ref[i], 1e-4f, "gemm blocked");
    }

    if (g_fail) {
        std::fprintf(stderr, "%d failures\n", g_fail);
        return 1;
    }
    std::printf("test_ops ok\n");
    return 0;
}
