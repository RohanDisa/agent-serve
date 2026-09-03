#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef _WIN32
#include <windows.h>
#include <malloc.h>
#endif

#if defined(__AVX2__) || defined(_M_AVX2)
#include <immintrin.h>
#define TINF_AVX2 1
#endif

#ifndef STREAM_ARRAY_SIZE
#define STREAM_ARRAY_SIZE 100000000
#endif

#ifndef NTIMES
#define NTIMES 10
#endif

#ifndef NSTREAM
#define NSTREAM 3  // warmup
#endif

// McCalpin STREAM triad: a[i] = b[i] + q * c[i]
// Bytes counted: 3 * N * sizeof(double). OpenMP over the whole array.

static double* a;
static double* b;
static double* c;
static const double q = 3.0;

static void* xmalloc(size_t n) {
#ifdef _WIN32
    void* p = _aligned_malloc(n, 64);
#else
    void* p = std::aligned_alloc(64, (n + 63) / 64 * 64);
#endif
    if (!p) {
        std::fprintf(stderr, "alloc failed (%zu bytes)\n", n);
        std::exit(1);
    }
    return p;
}

static void pin_physical(int logical_even) {
#ifdef _WIN32
    DWORD_PTR mask = (DWORD_PTR)1 << (logical_even % 64);
    SetThreadAffinityMask(GetCurrentThread(), mask);
#endif
}

static void bind_omp_to_physical_cores(int n_threads) {
#ifdef _OPENMP
#pragma omp parallel num_threads(n_threads)
    {
        int tid = omp_get_thread_num();
        pin_physical(tid * 2);  // 4600H: SMT pairs are (0,1), (2,3), ...
    }
#else
    pin_physical(0);
    (void)n_threads;
#endif
}

static double triad(int n_threads) {
    const int n = STREAM_ARRAY_SIZE;
#ifdef _OPENMP
    omp_set_num_threads(n_threads);
#endif
    auto t0 = std::chrono::steady_clock::now();
#ifdef _OPENMP
#pragma omp parallel num_threads(n_threads)
    {
        int tid = omp_get_thread_num();
        pin_physical(tid * 2);
        int nthr = omp_get_num_threads();
        int chunk = ((n + nthr - 1) / nthr + 7) / 8 * 8;
        int s = tid * chunk;
        int e = s + chunk;
        if (e > n) e = n;
        if (s < e) {
#if defined(TINF_AVX2)
            const __m256d vq = _mm256_set1_pd(q);
            int i = s;
            for (; i + 8 <= e; i += 8) {
                __m256d b0 = _mm256_load_pd(b + i);
                __m256d c0 = _mm256_load_pd(c + i);
                __m256d b1 = _mm256_load_pd(b + i + 4);
                __m256d c1 = _mm256_load_pd(c + i + 4);
                _mm256_store_pd(a + i, _mm256_fmadd_pd(vq, c0, b0));
                _mm256_store_pd(a + i + 4, _mm256_fmadd_pd(vq, c1, b1));
            }
            for (; i < e; ++i) a[i] = b[i] + q * c[i];
#else
            for (int i = s; i < e; ++i) a[i] = b[i] + q * c[i];
#endif
        }
    }
#else
    (void)n_threads;
    for (int i = 0; i < n; ++i) a[i] = b[i] + q * c[i];
#endif
    auto t1 = std::chrono::steady_clock::now();
    return std::chrono::duration<double>(t1 - t0).count();
}

int main(int argc, char** argv) {
    int n_threads = 6;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--threads" && i + 1 < argc) n_threads = std::atoi(argv[++i]);
    }
    if (n_threads < 1) n_threads = 1;

#ifdef _OPENMP
    omp_set_dynamic(0);
    omp_set_num_threads(n_threads);
#endif

#ifdef _WIN32
    SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);
#endif

    const size_t n = (size_t)STREAM_ARRAY_SIZE;
    a = (double*)xmalloc(n * sizeof(double));
    b = (double*)xmalloc(n * sizeof(double));
    c = (double*)xmalloc(n * sizeof(double));

#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(n_threads)
    for (int i = 0; i < STREAM_ARRAY_SIZE; ++i) {
        a[i] = 1.0;
        b[i] = 2.0;
        c[i] = 0.0;
    }
#else
    for (size_t i = 0; i < n; ++i) {
        a[i] = 1.0;
        b[i] = 2.0;
        c[i] = 0.0;
    }
#endif

    bind_omp_to_physical_cores(n_threads);

#ifdef _MSC_VER
    char compiler[80];
    std::snprintf(compiler, sizeof(compiler), "MSVC %d", _MSC_FULL_VER);
    const char* flags = "/O2 /openmp /arch:AVX2 /DSTREAM_ARRAY_SIZE=100000000";
#else
    const char* compiler = "g++/clang";
    const char* flags = "-O3 -march=native -fopenmp -DSTREAM_ARRAY_SIZE=100000000";
#endif

    std::fprintf(stderr,
                 "STREAM triad  N=%d  threads=%d  (%.1f MiB/array)  OpenMP=%s\n"
                 "compiler %s  flags %s\n",
                 STREAM_ARRAY_SIZE, n_threads, (n * sizeof(double)) / (1024.0 * 1024.0),
#ifdef _OPENMP
                 "on",
#else
                 "OFF",
#endif
                 compiler, flags);

    for (int w = 0; w < NSTREAM; ++w) triad(n_threads);

    std::vector<double> gbps;
    gbps.reserve(NTIMES);
    for (int r = 0; r < NTIMES; ++r) {
        double t = triad(n_threads);
        double g = (3.0 * (double)n * sizeof(double) / 1e9) / t;
        gbps.push_back(g);
        std::fprintf(stderr, "  run %d  %.2f GB/s  (%.4fs)\n", r, g, t);
    }
    std::sort(gbps.begin(), gbps.end());
    double med = gbps[gbps.size() / 2];
    double q1 = gbps[gbps.size() / 4];
    double q3 = gbps[(3 * gbps.size()) / 4];
    volatile double sink = a[0] + a[n / 2] + a[n - 1];

    std::printf(
        "{\n"
        "  \"bench\": \"stream_triad\",\n"
        "  \"array_size\": %d,\n"
        "  \"threads\": %d,\n"
        "  \"compiler\": \"%s\",\n"
        "  \"flags\": \"%s\",\n"
        "  \"openmp\": %s,\n"
        "  \"gb_s_median\": %.4f,\n"
        "  \"gb_s_q1\": %.4f,\n"
        "  \"gb_s_q3\": %.4f,\n"
        "  \"sink\": %.6f\n"
        "}\n",
        STREAM_ARRAY_SIZE, n_threads, compiler, flags,
#ifdef _OPENMP
        "true",
#else
        "false",
#endif
        med, q1, q3, (double)sink);

    std::fprintf(stderr, "median %.2f GB/s  IQR [%.2f, %.2f]\n", med, q1, q3);
#ifdef _WIN32
    _aligned_free(a);
    _aligned_free(b);
    _aligned_free(c);
#else
    std::free(a);
    std::free(b);
    std::free(c);
#endif
    return 0;
}
