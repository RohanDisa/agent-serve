#include "quant.hpp"
#include <algorithm>
#include <cmath>

#if defined(__AVX2__) || defined(_M_AVX2)
#include <immintrin.h>
#define TINF_AVX2 1
#endif
#if defined(__ARM_NEON) || defined(__aarch64__)
#include <arm_neon.h>
#define TINF_NEON 1
#endif

namespace tinf {

float f16_to_f32(uint16_t h) {
#if defined(TINF_AVX2)
    return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(h)));
#else
    uint32_t sign = (uint32_t)(h >> 15) << 31;
    uint32_t exp = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    uint32_t bits;
    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {
            exp = 127 - 14;
            while ((mant & 0x400) == 0) {
                mant <<= 1;
                --exp;
            }
            mant &= 0x3FF;
            bits = sign | (exp << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7F800000u | (mant << 13);
    } else {
        bits = sign | ((exp + (127 - 15)) << 23) | (mant << 13);
    }
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
#endif
}

uint16_t f32_to_f16(float f) {
#if defined(TINF_AVX2)
    return (uint16_t)_mm_cvtsi128_si32(_mm_cvtps_ph(_mm_set_ss(f), 0));
#else
    uint32_t bits;
    std::memcpy(&bits, &f, 4);
    uint32_t sign = (bits >> 16) & 0x8000;
    int32_t exp = (int32_t)((bits >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = bits & 0x7FFFFF;
    if (exp <= 0) {
        if (exp < -10) return (uint16_t)sign;
        mant = (mant | 0x800000) >> (1 - exp);
        return (uint16_t)(sign | (mant >> 13));
    }
    if (exp >= 31) return (uint16_t)(sign | 0x7C00);
    return (uint16_t)(sign | (exp << 10) | (mant >> 13));
#endif
}

void dequant_q8_0_row(float* out, const block_q8_0* x, int n) {
    int nb = n / QK8_0;
    for (int i = 0; i < nb; ++i) {
        float d = f16_to_f32(x[i].d);
        for (int j = 0; j < QK8_0; ++j) out[i * QK8_0 + j] = d * (float)x[i].qs[j];
    }
}

void dequant_q4_0_row(float* out, const block_q4_0* x, int n) {
    int nb = n / QK4_0;
    for (int i = 0; i < nb; ++i) {
        float d = f16_to_f32(x[i].d);
        for (int j = 0; j < QK4_0 / 2; ++j) {
            int x0 = (x[i].qs[j] & 0x0F) - 8;
            int x1 = (x[i].qs[j] >> 4) - 8;
            out[i * QK4_0 + j] = d * (float)x0;
            out[i * QK4_0 + j + 16] = d * (float)x1;
        }
    }
}

void dequant_q4_1_row(float* out, const block_q4_1* x, int n) {
    int nb = n / QK4_1;
    for (int i = 0; i < nb; ++i) {
        float d = f16_to_f32(x[i].d);
        float m = f16_to_f32(x[i].m);
        for (int j = 0; j < QK4_1 / 2; ++j) {
            int x0 = x[i].qs[j] & 0x0F;
            int x1 = x[i].qs[j] >> 4;
            out[i * QK4_1 + j] = x0 * d + m;
            out[i * QK4_1 + j + 16] = x1 * d + m;
        }
    }
}

void dequant_q6_K_row(float* out, const block_q6_K* x, int n) {
    int nb = n / QK_K;
    for (int i = 0; i < nb; ++i) {
        const float d = f16_to_f32(x[i].d);
        const uint8_t* ql = x[i].ql;
        const uint8_t* qh = x[i].qh;
        const int8_t* sc = x[i].scales;
        float* y = out + i * QK_K;
        for (int n0 = 0; n0 < QK_K; n0 += 128) {
            for (int l = 0; l < 32; ++l) {
                int is = l / 16;
                int8_t q1 = (int8_t)((ql[l] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                int8_t q2 = (int8_t)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                int8_t q3 = (int8_t)((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                int8_t q4 = (int8_t)((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                y[l] = d * (float)sc[is] * q1;
                y[l + 32] = d * (float)sc[is + 2] * q2;
                y[l + 64] = d * (float)sc[is + 4] * q3;
                y[l + 96] = d * (float)sc[is + 6] * q4;
            }
            y += 128;
            ql += 64;
            qh += 32;
            sc += 8;
        }
    }
}

void dequant_f16_row(float* out, const uint16_t* x, int n) {
    for (int i = 0; i < n; ++i) out[i] = f16_to_f32(x[i]);
}

void dequant_row(float* out, const Tensor& t, int64_t row) {
    int n = (int)t.cols();
    size_t rb = row_nbytes(t.dtype, t.cols());
    const uint8_t* src = t.data + row * rb;
    switch (t.dtype) {
        case DType::F32: std::memcpy(out, src, (size_t)n * 4); break;
        case DType::F16: dequant_f16_row(out, (const uint16_t*)src, n); break;
        case DType::Q8_0: dequant_q8_0_row(out, (const block_q8_0*)src, n); break;
        case DType::Q4_0: dequant_q4_0_row(out, (const block_q4_0*)src, n); break;
        case DType::Q4_1: dequant_q4_1_row(out, (const block_q4_1*)src, n); break;
        case DType::Q6_K: dequant_q6_K_row(out, (const block_q6_K*)src, n); break;
    }
}

void quant_q8_0_row(block_q8_0* out, const float* x, int n) {
    int nb = n / QK8_0;
    for (int i = 0; i < nb; ++i) {
        float amax = 0;
        for (int j = 0; j < QK8_0; ++j) amax = std::max(amax, std::fabs(x[i * QK8_0 + j]));
        float d = amax / 127.0f;
        float id = d > 0 ? 1.0f / d : 0;
        out[i].d = f32_to_f16(d);
        for (int j = 0; j < QK8_0; ++j) {
            float v = x[i * QK8_0 + j] * id;
            int q = (int)std::round(v);
            if (q > 127) q = 127;
            if (q < -127) q = -127;
            out[i].qs[j] = (int8_t)q;
        }
    }
}

void quant_q4_0_row(block_q4_0* out, const float* x, int n) {
    int nb = n / QK4_0;
    for (int i = 0; i < nb; ++i) {
        float amax = 0;
        for (int j = 0; j < QK4_0; ++j) amax = std::max(amax, std::fabs(x[i * QK4_0 + j]));
        float d = amax / 8.0f;
        float id = d > 0 ? 1.0f / d : 0;
        out[i].d = f32_to_f16(d);
        for (int j = 0; j < QK4_0 / 2; ++j) {
            int x0 = (int)std::round(x[i * QK4_0 + j] * id);
            int x1 = (int)std::round(x[i * QK4_0 + j + 16] * id);
            if (x0 < -8) x0 = -8;
            if (x0 > 7) x0 = 7;
            if (x1 < -8) x1 = -8;
            if (x1 > 7) x1 = 7;
            out[i].qs[j] = (uint8_t)((x0 + 8) | ((x1 + 8) << 4));
        }
    }
}

#if defined(TINF_AVX2)
static float hsum8(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    lo = _mm_add_ps(lo, hi);
    __m128 sh = _mm_movehdup_ps(lo);
    lo = _mm_add_ps(lo, sh);
    sh = _mm_movehl_ps(sh, lo);
    lo = _mm_add_ss(lo, sh);
    return _mm_cvtss_f32(lo);
}
#endif

float dot_f32(const float* a, const float* b, int n, bool simd) {
#if defined(TINF_AVX2)
    if (simd) {
        __m256 acc = _mm256_setzero_ps();
        int i = 0;
        for (; i + 32 <= n; i += 32) {
            acc = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), acc);
            acc = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 8), _mm256_loadu_ps(b + i + 8), acc);
            acc = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 16), _mm256_loadu_ps(b + i + 16), acc);
            acc = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 24), _mm256_loadu_ps(b + i + 24), acc);
        }
        for (; i + 8 <= n; i += 8)
            acc = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), acc);
        float s = hsum8(acc);
        for (; i < n; ++i) s += a[i] * b[i];
        return s;
    }
#elif defined(TINF_NEON)
    if (simd) {
        float32x4_t acc = vdupq_n_f32(0);
        int i = 0;
        for (; i + 16 <= n; i += 16) {
#if defined(__aarch64__)
            acc = vfmaq_f32(acc, vld1q_f32(a + i), vld1q_f32(b + i));
            acc = vfmaq_f32(acc, vld1q_f32(a + i + 4), vld1q_f32(b + i + 4));
            acc = vfmaq_f32(acc, vld1q_f32(a + i + 8), vld1q_f32(b + i + 8));
            acc = vfmaq_f32(acc, vld1q_f32(a + i + 12), vld1q_f32(b + i + 12));
#else
            acc = vmlaq_f32(acc, vld1q_f32(a + i), vld1q_f32(b + i));
            acc = vmlaq_f32(acc, vld1q_f32(a + i + 4), vld1q_f32(b + i + 4));
            acc = vmlaq_f32(acc, vld1q_f32(a + i + 8), vld1q_f32(b + i + 8));
            acc = vmlaq_f32(acc, vld1q_f32(a + i + 12), vld1q_f32(b + i + 12));
#endif
        }
        float buf[4];
        vst1q_f32(buf, acc);
        float s = buf[0] + buf[1] + buf[2] + buf[3];
        for (; i < n; ++i) s += a[i] * b[i];
        return s;
    }
#endif
    (void)simd;
    double s = 0;
    for (int i = 0; i < n; ++i) s += (double)a[i] * b[i];
    return (float)s;
}

float dot_f16_f32(const float* x, const uint16_t* w, int n, bool simd) {
#if defined(TINF_AVX2)
    if (simd) {
        __m256 acc = _mm256_setzero_ps();
        int i = 0;
        for (; i + 8 <= n; i += 8) {
            __m128i h = _mm_loadu_si128((const __m128i*)(w + i));
            __m256 wf = _mm256_cvtph_ps(h);
            acc = _mm256_fmadd_ps(_mm256_loadu_ps(x + i), wf, acc);
        }
        float s = hsum8(acc);
        for (; i < n; ++i) s += x[i] * f16_to_f32(w[i]);
        return s;
    }
#endif
    float s = 0;
    for (int i = 0; i < n; ++i) s += x[i] * f16_to_f32(w[i]);
    (void)simd;
    return s;
}

float dot_q8_0_f32(const float* x, const block_q8_0* w, int n, bool simd) {
    int nb = n / QK8_0;
    float sum = 0;
#if defined(TINF_AVX2)
    if (simd) {
        for (int i = 0; i < nb; ++i) {
            const float d = f16_to_f32(w[i].d);
            const float* xi = x + i * QK8_0;
            const int8_t* q = w[i].qs;
            __m256 acc = _mm256_setzero_ps();
            for (int j = 0; j < 32; j += 8) {
                __m128i b8 = _mm_loadl_epi64((const __m128i*)(q + j));
                __m256i b32 = _mm256_cvtepi8_epi32(b8);
                __m256 bf = _mm256_cvtepi32_ps(b32);
                acc = _mm256_fmadd_ps(_mm256_loadu_ps(xi + j), bf, acc);
            }
            sum += d * hsum8(acc);
        }
        return sum;
    }
#elif defined(TINF_NEON)
    if (simd) {
        for (int i = 0; i < nb; ++i) {
            const float d = f16_to_f32(w[i].d);
            const float* xi = x + i * QK8_0;
            const int8_t* q = w[i].qs;
            float32x4_t acc = vdupq_n_f32(0);
            for (int j = 0; j < 32; j += 8) {
                int8x8_t q8 = vld1_s8(q + j);
                int16x8_t q16 = vmovl_s8(q8);
                int32x4_t lo = vmovl_s16(vget_low_s16(q16));
                int32x4_t hi = vmovl_s16(vget_high_s16(q16));
#if defined(__aarch64__)
                acc = vfmaq_f32(acc, vld1q_f32(xi + j), vcvtq_f32_s32(lo));
                acc = vfmaq_f32(acc, vld1q_f32(xi + j + 4), vcvtq_f32_s32(hi));
#else
                acc = vmlaq_f32(acc, vld1q_f32(xi + j), vcvtq_f32_s32(lo));
                acc = vmlaq_f32(acc, vld1q_f32(xi + j + 4), vcvtq_f32_s32(hi));
#endif
            }
            float buf[4];
            vst1q_f32(buf, acc);
            sum += d * (buf[0] + buf[1] + buf[2] + buf[3]);
        }
        return sum;
    }
#endif
    for (int i = 0; i < nb; ++i) {
        float d = f16_to_f32(w[i].d);
        float s = 0;
        for (int j = 0; j < QK8_0; ++j) s += x[i * QK8_0 + j] * (float)w[i].qs[j];
        sum += d * s;
    }
    (void)simd;
    return sum;
}

float dot_q4_0_f32(const float* x, const block_q4_0* w, int n, bool simd) {
    int nb = n / QK4_0;
    float sum = 0;
#if defined(TINF_AVX2)
    if (simd) {
        const __m256i m4 = _mm256_set1_epi8(0x0F);
        const __m256i off = _mm256_set1_epi8(8);
        for (int i = 0; i < nb; ++i) {
            float d = f16_to_f32(w[i].d);
            const float* xi = x + i * QK4_0;
            __m128i qraw = _mm_loadu_si128((const __m128i*)w[i].qs);
            __m256i q8 = _mm256_set_m128i(_mm_srli_epi16(qraw, 4), qraw);
            q8 = _mm256_and_si256(q8, m4);
            q8 = _mm256_sub_epi8(q8, off);
            __m256 acc = _mm256_setzero_ps();
            int8_t tmp[32];
            _mm256_storeu_si256((__m256i*)tmp, q8);
            for (int j = 0; j < 32; j += 8) {
                __m128i b8 = _mm_loadl_epi64((const __m128i*)(tmp + j));
                __m256 bf = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(b8));
                acc = _mm256_fmadd_ps(_mm256_loadu_ps(xi + j), bf, acc);
            }
            sum += d * hsum8(acc);
        }
        return sum;
    }
#endif
    (void)simd;
    for (int i = 0; i < nb; ++i) {
        float d = f16_to_f32(w[i].d);
        float s = 0;
        for (int j = 0; j < 16; ++j) {
            int x0 = (w[i].qs[j] & 0x0F) - 8;
            int x1 = (w[i].qs[j] >> 4) - 8;
            s += x[i * QK4_0 + j] * (float)x0;
            s += x[i * QK4_0 + j + 16] * (float)x1;
        }
        sum += d * s;
    }
    return sum;
}

}  // namespace tinf
