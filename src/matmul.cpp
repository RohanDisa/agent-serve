#include "matmul.hpp"

namespace tinf {

static float dot_weight_row(const float* x, const Tensor& W, int64_t row, int K, bool simd) {
    size_t rb = row_nbytes(W.dtype, W.cols());
    const uint8_t* src = W.data + row * rb;
    switch (W.dtype) {
        case DType::F32: return dot_f32(x, (const float*)src, K, simd);
        case DType::F16: return dot_f16_f32(x, (const uint16_t*)src, K, simd);
        case DType::Q8_0: return dot_q8_0_f32(x, (const block_q8_0*)src, K, simd);
        case DType::Q4_0: return dot_q4_0_f32(x, (const block_q4_0*)src, K, simd);
        default: {
            thread_local std::vector<float> tmp;
            if ((int)tmp.size() < K) tmp.resize(K);
            dequant_row(tmp.data(), W, row);
            return dot_f32(x, tmp.data(), K, simd);
        }
    }
}

static void gemm_naive(float* C, const float* A, const Tensor& W, int M, int N, int K, bool simd,
                       ThreadPool* pool) {
    auto body = [&](int n0, int n1) {
        for (int n = n0; n < n1; ++n) {
            for (int m = 0; m < M; ++m)
                C[m * N + n] = dot_weight_row(A + m * K, W, n, K, simd);
        }
    };
    if (pool) pool->parallel(N, body);
    else body(0, N);
}

// Prefill: reuse a dequantized / streamed K-panel of W across the M token rows.
static void gemm_blocked(float* C, const float* A, const Tensor& W, int M, int N, int K, bool simd,
                         ThreadPool* pool) {
    const int BK = 256;
    const int BN = 32;
    auto body = [&](int n0, int n1) {
        std::vector<float> panel((size_t)BN * BK);
        std::vector<float> rows((size_t)BN * K);
        for (int nbase = n0; nbase < n1; nbase += BN) {
            int bn = std::min(BN, n1 - nbase);
            for (int n = 0; n < bn; ++n)
                dequant_row(rows.data() + n * K, W, nbase + n);
            for (int k0 = 0; k0 < K; k0 += BK) {
                int bk = std::min(BK, K - k0);
                for (int n = 0; n < bn; ++n)
                    std::memcpy(panel.data() + n * BK, rows.data() + n * K + k0, (size_t)bk * sizeof(float));
                for (int m = 0; m < M; ++m) {
                    for (int n = 0; n < bn; ++n) {
                        float s = dot_f32(A + m * K + k0, panel.data() + n * BK, bk, simd);
                        if (k0 == 0) C[m * N + nbase + n] = s;
                        else C[m * N + nbase + n] += s;
                    }
                }
            }
        }
    };
    if (pool) pool->parallel(N, body);
    else body(0, N);
}

void matmul(float* C, const float* A, const Tensor& W, int M, const Exec& ex) {
    ScopeTime st(ex.t ? &ex.t->matmul : nullptr);
    const int N = (int)W.rows();
    const int K = (int)W.cols();
    const bool simd = ex.cfg && ex.cfg->use_simd;
    const bool blocked = ex.cfg && ex.cfg->use_blocking && M > 1;
    ThreadPool* pool = ex.pool;
    if (blocked) gemm_blocked(C, A, W, M, N, K, simd, pool);
    else gemm_naive(C, A, W, M, N, K, simd, pool);
}

void gemm_f32_bench(float* C, const float* A, const float* B, int M, int N, int K, bool simd, bool blocked,
                    ThreadPool* pool) {
    Tensor W;
    W.dtype = DType::F32;
    W.ndim = 2;
    W.shape[0] = K;
    W.shape[1] = N;
    W.data = (const uint8_t*)B;
    RunConfig cfg;
    cfg.use_simd = simd;
    cfg.use_blocking = blocked;
    Exec ex;
    ex.cfg = &cfg;
    ex.pool = pool;
    matmul(C, A, W, M, ex);
}

}  // namespace tinf
