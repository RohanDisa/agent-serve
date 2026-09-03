#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <malloc.h>
#else
#include <sched.h>
#include <unistd.h>
#endif

#ifndef STREAM_ARRAY_SIZE
#define STREAM_ARRAY_SIZE 100000000
#endif
#ifndef NTIMES
#define NTIMES 10
#endif
#ifndef NWARM
#define NWARM 3
#endif

static double *a, *b, *c;
static const double scalar = 3.0;

static void *xmalloc(size_t n) {
#ifdef _WIN32
    void *p = _aligned_malloc(n, 64);
#else
    void *p = NULL;
    if (posix_memalign(&p, 64, n) != 0) p = NULL;
#endif
    if (!p) {
        fprintf(stderr, "alloc failed (%zu bytes)\n", n);
        exit(1);
    }
    return p;
}

static void pin_physical(int tid) {
    /* 4600H Windows/WSL logical map: SMT siblings are consecutive, physical cores even. */
    int cpu = tid * 2;
#ifdef _WIN32
    DWORD_PTR mask = ((DWORD_PTR)1) << (cpu % 64);
    SetThreadAffinityMask(GetCurrentThread(), mask);
#else
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpu, &set);
    sched_setaffinity(0, sizeof(set), &set);
#endif
}

static void bind_threads(int n_threads) {
#ifdef _OPENMP
#pragma omp parallel num_threads(n_threads)
    {
        pin_physical(omp_get_thread_num());
    }
#else
    pin_physical(0);
    (void)n_threads;
#endif
}

static double now_s(void) {
#ifdef _WIN32
    static LARGE_INTEGER freq;
    static int init;
    LARGE_INTEGER t;
    if (!init) {
        QueryPerformanceFrequency(&freq);
        init = 1;
    }
    QueryPerformanceCounter(&t);
    return (double)t.QuadPart / (double)freq.QuadPart;
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
#endif
}

/* Bytes moved per element, McCalpin convention (no extra write-allocate term). */
static double kernel_triad(int n) {
    int i;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
    for (i = 0; i < n; i++) a[i] = b[i] + scalar * c[i];
#else
    for (i = 0; i < n; i++) a[i] = b[i] + scalar * c[i];
#endif
    return 3.0 * (double)n * sizeof(double);
}

static double kernel_copy(int n) {
    int i;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
    for (i = 0; i < n; i++) a[i] = b[i];
#else
    for (i = 0; i < n; i++) a[i] = b[i];
#endif
    return 2.0 * (double)n * sizeof(double);
}

static double kernel_scale(int n) {
    int i;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
    for (i = 0; i < n; i++) a[i] = scalar * b[i];
#else
    for (i = 0; i < n; i++) a[i] = scalar * b[i];
#endif
    return 2.0 * (double)n * sizeof(double);
}

/* Sequential read-only reduction. Decode is read-dominated; this is the matching roof. */
static double kernel_read(int n) {
    int i;
    double sum = 0.0;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) reduction(+ : sum)
    for (i = 0; i < n; i++) sum += a[i];
#else
    for (i = 0; i < n; i++) sum += a[i];
#endif
    /* keep the reduction live */
    a[0] = sum * 1e-300 + a[0];
    return 1.0 * (double)n * sizeof(double);
}

typedef double (*kern_fn)(int);

static void run_kernel(const char *name, kern_fn fn, int n, int n_threads, double *med, double *q1, double *q3,
                       double *sink) {
    double gbps[NTIMES];
    int r, w;
#ifdef _OPENMP
    omp_set_dynamic(0);
    omp_set_num_threads(n_threads);
#endif
    bind_threads(n_threads);
    for (w = 0; w < NWARM; w++) fn(n);
    for (r = 0; r < NTIMES; r++) {
        double t0 = now_s();
        double bytes = fn(n);
        double t1 = now_s();
        gbps[r] = (bytes / 1e9) / (t1 - t0);
        fprintf(stderr, "  %-6s t=%d run %d  %.2f GB/s\n", name, n_threads, r, gbps[r]);
    }
    /* insertion sort */
    for (r = 1; r < NTIMES; r++) {
        double v = gbps[r];
        int j = r;
        while (j > 0 && gbps[j - 1] > v) {
            gbps[j] = gbps[j - 1];
            j--;
        }
        gbps[j] = v;
    }
    *med = gbps[NTIMES / 2];
    *q1 = gbps[NTIMES / 4];
    *q3 = gbps[(3 * NTIMES) / 4];
    *sink = a[0] + a[n / 2] + a[n - 1];
}

int main(int argc, char **argv) {
    int n_threads = 6;
    const char *kernel = "all";
    int i;
    const int n = STREAM_ARRAY_SIZE;

    for (i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--threads") == 0 && i + 1 < argc) n_threads = atoi(argv[++i]);
        else if (strcmp(argv[i], "--kernel") == 0 && i + 1 < argc) kernel = argv[++i];
    }
    if (n_threads < 1) n_threads = 1;

#ifdef _WIN32
    SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);
#endif

#ifdef _OPENMP
    omp_set_dynamic(0);
    omp_set_num_threads(n_threads);
#endif

    a = (double *)xmalloc((size_t)n * sizeof(double));
    b = (double *)xmalloc((size_t)n * sizeof(double));
    c = (double *)xmalloc((size_t)n * sizeof(double));
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
    for (i = 0; i < n; i++) {
        a[i] = 1.0;
        b[i] = 2.0;
        c[i] = 0.0;
    }
#else
    for (i = 0; i < n; i++) {
        a[i] = 1.0;
        b[i] = 2.0;
        c[i] = 0.0;
    }
#endif

#ifdef _MSC_VER
    {
        char compiler[80];
        snprintf(compiler, sizeof(compiler), "MSVC %d", _MSC_FULL_VER);
        fprintf(stderr, "STREAM N=%d threads=%d OpenMP=%s\ncompiler %s\nflags /O2 /openmp /arch:AVX2 /DSTREAM_ARRAY_SIZE=%d\n",
                n, n_threads,
#ifdef _OPENMP
                "on",
#else
                "OFF",
#endif
                compiler, STREAM_ARRAY_SIZE);
    }
#else
    fprintf(stderr,
            "STREAM N=%d threads=%d OpenMP=%s\ncompiler gcc\nflags -O3 -march=native -fopenmp -DSTREAM_ARRAY_SIZE=%d\n",
            n, n_threads,
#ifdef _OPENMP
            "on",
#else
            "OFF",
#endif
            STREAM_ARRAY_SIZE);
#endif
    fprintf(stderr, "pinning even logical CPUs (physical cores), tid*2\n");

    printf("{\n  \"array_size\": %d,\n  \"threads\": %d,\n  \"kernels\": {\n", n, n_threads);

    {
        const char *names[] = {"triad", "copy", "scale", "read"};
        kern_fn fns[] = {kernel_triad, kernel_copy, kernel_scale, kernel_read};
        int k, first = 1;
        for (k = 0; k < 4; k++) {
            double med, q1, q3, sink;
            if (strcmp(kernel, "all") != 0 && strcmp(kernel, names[k]) != 0) continue;
            run_kernel(names[k], fns[k], n, n_threads, &med, &q1, &q3, &sink);
            if (!first) printf(",\n");
            first = 0;
            printf("    \"%s\": {\"gb_s_median\": %.4f, \"gb_s_q1\": %.4f, \"gb_s_q3\": %.4f, \"sink\": %.6g}", names[k],
                   med, q1, q3, sink);
        }
        printf("\n  }\n}\n");
    }

#ifdef _WIN32
    _aligned_free(a);
    _aligned_free(b);
    _aligned_free(c);
#else
    free(a);
    free(b);
    free(c);
#endif
    return 0;
}
