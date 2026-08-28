#include "tensor.hpp"
#include "quant.hpp"

namespace tinf {

const char* dtype_name(DType d) {
    switch (d) {
        case DType::F32: return "F32";
        case DType::F16: return "F16";
        case DType::Q4_0: return "Q4_0";
        case DType::Q4_1: return "Q4_1";
        case DType::Q8_0: return "Q8_0";
        case DType::Q6_K: return "Q6_K";
    }
    return "?";
}

DType dtype_from_ggml(int ggml_type) {
    switch (ggml_type) {
        case 0: return DType::F32;
        case 1: return DType::F16;
        case 2: return DType::Q4_0;
        case 3: return DType::Q4_1;
        case 8: return DType::Q8_0;
        case 14: return DType::Q6_K;
        default: throw std::runtime_error("unsupported ggml type " + std::to_string(ggml_type));
    }
}

int ggml_from_dtype(DType d) {
    switch (d) {
        case DType::F32: return 0;
        case DType::F16: return 1;
        case DType::Q4_0: return 2;
        case DType::Q4_1: return 3;
        case DType::Q8_0: return 8;
        case DType::Q6_K: return 14;
    }
    return 0;
}

int blck_size(DType d) {
    switch (d) {
        case DType::F32:
        case DType::F16: return 1;
        case DType::Q4_0:
        case DType::Q4_1:
        case DType::Q8_0: return 32;
        case DType::Q6_K: return 256;
    }
    return 1;
}

size_t blck_nbytes(DType d) {
    switch (d) {
        case DType::F32: return 4;
        case DType::F16: return 2;
        case DType::Q4_0: return 18;
        case DType::Q4_1: return 20;
        case DType::Q8_0: return 34;
        case DType::Q6_K: return 210;
    }
    return 0;
}

size_t row_nbytes(DType d, int64_t ne0) {
    int bs = blck_size(d);
    if (ne0 % bs != 0) throw std::runtime_error("row length not divisible by block size");
    return (size_t)(ne0 / bs) * blck_nbytes(d);
}

size_t tensor_nbytes(DType d, const int64_t* shape, int ndim) {
    if (ndim <= 0) return 0;
    int64_t nrows = 1;
    for (int i = 1; i < ndim; ++i) nrows *= shape[i];
    return (size_t)nrows * row_nbytes(d, shape[0]);
}

}  // namespace tinf
