#pragma once

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <malloc.h>
#else
#include <fcntl.h>
#include <pthread.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace tinf {

[[noreturn]] inline void die(const std::string& msg) {
    std::fprintf(stderr, "tinferno: %s\n", msg.c_str());
    std::exit(1);
}

inline void* aligned_malloc(size_t n, size_t align = 64) {
    if (n == 0) n = align;
    size_t bytes = (n + align - 1) / align * align;
#ifdef _WIN32
    void* p = _aligned_malloc(bytes, align);
#else
    void* p = std::aligned_alloc(align, bytes);
#endif
    if (!p) throw std::bad_alloc();
    return p;
}

inline void aligned_free(void* p) {
#ifdef _WIN32
    _aligned_free(p);
#else
    std::free(p);
#endif
}

struct AlignedBuf {
    float* p = nullptr;
    size_t n = 0;
    AlignedBuf() = default;
    explicit AlignedBuf(size_t count) { reset(count); }
    ~AlignedBuf() { reset(0); }
    AlignedBuf(const AlignedBuf&) = delete;
    AlignedBuf& operator=(const AlignedBuf&) = delete;
    AlignedBuf(AlignedBuf&& o) noexcept : p(o.p), n(o.n) { o.p = nullptr; o.n = 0; }
    void reset(size_t count) {
        if (p) aligned_free(p);
        p = nullptr;
        n = count;
        if (count) {
            p = static_cast<float*>(aligned_malloc(count * sizeof(float)));
            std::memset(p, 0, count * sizeof(float));
        }
    }
};

inline double now_s() {
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double>(clock::now().time_since_epoch()).count();
}

inline void pin_thread(int cpu) {
#ifdef _WIN32
    DWORD_PTR mask = (DWORD_PTR)1 << (cpu % 64);
    SetThreadAffinityMask(GetCurrentThread(), mask);
#else
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpu, &set);
    pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
#endif
}

enum class DType : int { F32 = 0, F16 = 1, Q4_0 = 2, Q4_1 = 3, Q8_0 = 8, Q6_K = 14 };

enum class KVLayout { SeqMajor, HeadMajor };

struct RunConfig {
    int n_threads = 1;
    bool use_simd = false;
    bool use_blocking = false;
    KVLayout kv_layout = KVLayout::SeqMajor;
    int n_predict = 64;
    float temperature = 0.0f;
    float top_p = 0.9f;
    int seed = 42;
    int max_seq = 4096;
    bool pin_threads = false;
    bool profile = false;
    std::string stage = "v2";
};

struct Timings {
    double matmul = 0;
    double attn = 0;
    double rmsnorm = 0;
    double rope = 0;
    double softmax = 0;
    double sample = 0;
    void clear() { *this = Timings{}; }
    double total() const { return matmul + attn + rmsnorm + rope + softmax + sample; }
};

struct ScopeTime {
    double* acc;
    double t0;
    explicit ScopeTime(double* a) : acc(a), t0(now_s()) {}
    ~ScopeTime() { if (acc) *acc += now_s() - t0; }
};

class ThreadPool {
public:
    explicit ThreadPool(int n_threads, bool pin = false)
        : n_(std::max(1, n_threads)) {
        if (n_ == 1) {
            if (pin) pin_thread(0);
            return;
        }
        workers_.reserve(n_ - 1);
        for (int i = 1; i < n_; ++i) {
            workers_.emplace_back([this, i, pin] {
                if (pin) pin_thread(i);
                worker_loop(i);
            });
        }
        if (pin) pin_thread(0);
    }

    ~ThreadPool() {
        {
            std::lock_guard<std::mutex> lk(mu_);
            stop_ = true;
        }
        cv_.notify_all();
        for (auto& t : workers_) t.join();
    }

    int n_threads() const { return n_; }

    void parallel(int n, const std::function<void(int, int)>& fn) {
        if (n <= 0) return;
        if (n_ == 1 || n <= 1) {
            fn(0, n);
            return;
        }
        fn_ = &fn;
        n_items_ = n;
        done_ = 0;
        {
            std::lock_guard<std::mutex> lk(mu_);
            ++work_id_;
        }
        cv_.notify_all();
        do_chunk(0);
        std::unique_lock<std::mutex> lk(mu_);
        cv_done_.wait(lk, [&] { return done_ >= n_ - 1; });
    }

private:
    void do_chunk(int tid) {
        int chunk = (n_items_ + n_ - 1) / n_;
        int s = tid * chunk;
        int e = std::min(n_items_, s + chunk);
        if (s < e) (*fn_)(s, e);
    }

    void worker_loop(int tid) {
        uint64_t seen = 0;
        while (true) {
            std::unique_lock<std::mutex> lk(mu_);
            cv_.wait(lk, [&] { return stop_ || work_id_ != seen; });
            if (stop_) return;
            seen = work_id_;
            lk.unlock();
            do_chunk(tid);
            std::lock_guard<std::mutex> lk2(mu_);
            if (++done_ >= n_ - 1) cv_done_.notify_one();
        }
    }

    int n_;
    std::vector<std::thread> workers_;
    std::mutex mu_;
    std::condition_variable cv_;
    std::condition_variable cv_done_;
    bool stop_ = false;
    uint64_t work_id_ = 0;
    int n_items_ = 0;
    int done_ = 0;
    const std::function<void(int, int)>* fn_ = nullptr;
};

struct Exec {
    const RunConfig* cfg = nullptr;
    ThreadPool* pool = nullptr;
    Timings* t = nullptr;
};

class MappedFile {
public:
    MappedFile() = default;
    explicit MappedFile(const std::string& path) { open(path); }
    ~MappedFile() { close(); }
    MappedFile(const MappedFile&) = delete;
    MappedFile& operator=(const MappedFile&) = delete;

    void open(const std::string& path) {
        close();
#ifdef _WIN32
        int wlen = MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, nullptr, 0);
        std::wstring w(wlen, 0);
        MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, &w[0], wlen);
        file_ = CreateFileW(w.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr,
                            OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
        if (file_ == INVALID_HANDLE_VALUE) throw std::runtime_error("cannot open " + path);
        LARGE_INTEGER sz;
        if (!GetFileSizeEx(file_, &sz)) throw std::runtime_error("GetFileSizeEx failed");
        size_ = (size_t)sz.QuadPart;
        map_ = CreateFileMappingW(file_, nullptr, PAGE_READONLY, sz.HighPart, sz.LowPart, nullptr);
        if (!map_) throw std::runtime_error("CreateFileMapping failed");
        data_ = MapViewOfFile(map_, FILE_MAP_READ, 0, 0, 0);
        if (!data_) throw std::runtime_error("MapViewOfFile failed");
#else
        fd_ = ::open(path.c_str(), O_RDONLY);
        if (fd_ < 0) throw std::runtime_error("cannot open " + path);
        struct stat st {};
        if (fstat(fd_, &st) != 0) throw std::runtime_error("fstat failed");
        size_ = (size_t)st.st_size;
        data_ = mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd_, 0);
        if (data_ == MAP_FAILED) throw std::runtime_error("mmap failed");
#endif
    }

    void close() {
#ifdef _WIN32
        if (data_) UnmapViewOfFile(data_);
        if (map_) CloseHandle(map_);
        if (file_ && file_ != INVALID_HANDLE_VALUE) CloseHandle(file_);
        file_ = INVALID_HANDLE_VALUE;
        map_ = nullptr;
#else
        if (data_ && data_ != MAP_FAILED) munmap(data_, size_);
        if (fd_ >= 0) ::close(fd_);
        fd_ = -1;
#endif
        data_ = nullptr;
        size_ = 0;
    }

    const uint8_t* data() const { return static_cast<const uint8_t*>(data_); }
    size_t size() const { return size_; }

private:
    void* data_ = nullptr;
    size_t size_ = 0;
#ifdef _WIN32
    HANDLE file_ = INVALID_HANDLE_VALUE;
    HANDLE map_ = nullptr;
#else
    int fd_ = -1;
#endif
};

inline RunConfig stage_preset(const std::string& stage, int hw_threads) {
    RunConfig c;
    c.stage = stage;
    int nt = std::max(1, hw_threads);
    if (stage == "v0") {
        c.n_threads = 1;
        c.use_simd = false;
        c.use_blocking = false;
        c.kv_layout = KVLayout::SeqMajor;
    } else if (stage == "v1") {
        c.n_threads = nt;
        c.use_simd = false;
        c.use_blocking = false;
        c.kv_layout = KVLayout::SeqMajor;
    } else if (stage == "v2" || stage == "v3") {
        c.n_threads = nt;
        c.use_simd = true;
        c.use_blocking = false;
        c.kv_layout = KVLayout::SeqMajor;
    } else if (stage == "v4") {
        c.n_threads = nt;
        c.use_simd = true;
        c.use_blocking = true;
        c.kv_layout = KVLayout::SeqMajor;
    } else if (stage == "v5") {
        c.n_threads = nt;
        c.use_simd = true;
        c.use_blocking = true;
        c.kv_layout = KVLayout::HeadMajor;
    } else {
        throw std::runtime_error("unknown stage " + stage + " (use v0..v5)");
    }
    return c;
}

}  // namespace tinf
