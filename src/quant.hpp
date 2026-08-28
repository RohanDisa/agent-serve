#pragma once

#include "tensor.hpp"

namespace tinf {

constexpr int QK8_0 = 32;
constexpr int QK4_0 = 32;
constexpr int QK4_1 = 32;
constexpr int QK_K = 256;

#pragma pack(push, 1)
struct block_q8_0 {
    uint16_t d;
    int8_t qs[QK8_0];
};
struct block_q4_0 {
    uint16_t d;
    uint8_t qs[QK4_0 / 2];
};
struct block_q4_1 {
    uint16_t d;
    uint16_t m;
    uint8_t qs[QK4_1 / 2];
};
struct block_q6_K {
    uint8_t ql[QK_K / 2];
    uint8_t qh[QK_K / 4];
    int8_t scales[QK_K / 16];
    uint16_t d;
};
#pragma pack(pop)
static_assert(sizeof(block_q8_0) == 34, "q8_0 block");
static_assert(sizeof(block_q4_0) == 18, "q4_0 block");
static_assert(sizeof(block_q4_1) == 20, "q4_1 block");
static_assert(sizeof(block_q6_K) == 210, "q6_K block");

float f16_to_f32(uint16_t h);
uint16_t f32_to_f16(float f);

void dequant_row(float* out, const Tensor& t, int64_t row);
void dequant_q8_0_row(float* out, const block_q8_0* x, int n);
void dequant_q4_0_row(float* out, const block_q4_0* x, int n);
void dequant_q4_1_row(float* out, const block_q4_1* x, int n);
void dequant_q6_K_row(float* out, const block_q6_K* x, int n);
void dequant_f16_row(float* out, const uint16_t* x, int n);

void quant_q8_0_row(block_q8_0* out, const float* x, int n);
void quant_q4_0_row(block_q4_0* out, const float* x, int n);

float dot_q8_0_f32(const float* x, const block_q8_0* w, int n, bool simd);
float dot_q4_0_f32(const float* x, const block_q4_0* w, int n, bool simd);
float dot_f16_f32(const float* x, const uint16_t* w, int n, bool simd);
float dot_f32(const float* a, const float* b, int n, bool simd);

}  // namespace tinf
