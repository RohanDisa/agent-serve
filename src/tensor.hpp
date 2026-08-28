#pragma once

#include "common.hpp"

namespace tinf {

struct Tensor {
    std::string name;
    DType dtype = DType::F32;
    int ndim = 0;
    int64_t shape[4] = {1, 1, 1, 1};  // ggml order: shape[0] = innermost (ne0)
    const uint8_t* data = nullptr;
    size_t nbytes = 0;

    int64_t ne(int i) const { return i < ndim ? shape[i] : 1; }
    int64_t cols() const { return ne(0); }  // K for 2D weights
    int64_t rows() const { return ne(1); }  // N for 2D weights
    int64_t numel() const {
        int64_t n = 1;
        for (int i = 0; i < ndim; ++i) n *= shape[i];
        return n;
    }
};

const char* dtype_name(DType d);
DType dtype_from_ggml(int ggml_type);
int ggml_from_dtype(DType d);
int blck_size(DType d);
size_t blck_nbytes(DType d);
size_t row_nbytes(DType d, int64_t ne0);
size_t tensor_nbytes(DType d, const int64_t* shape, int ndim);

}  // namespace tinf
