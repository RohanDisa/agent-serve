#pragma once

#include "quant.hpp"

namespace tinf {

// C[m, n] = A[m, k] @ W[n, k]^T   i.e. C[m, n] = dot(A[m,:], W[n,:])
// W is a GGUF weight tensor (row-major, ggml ne0 = K).
void matmul(float* C, const float* A, const Tensor& W, int M, const Exec& ex);

// Isolated F32 GEMM for the FLOPS microbenchmark: C[M,N] = A[M,K] @ B[N,K]^T
void gemm_f32_bench(float* C, const float* A, const float* B, int M, int N, int K, bool simd, bool blocked,
                    ThreadPool* pool);

}  // namespace tinf
