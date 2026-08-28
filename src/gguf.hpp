#pragma once

#include "tensor.hpp"
#include <unordered_map>

namespace tinf {

class GGUF {
public:
    explicit GGUF(const std::string& path);

    const Tensor* find(const std::string& name) const;
    const Tensor& must(const std::string& name) const;

    uint64_t u64(const std::string& key, uint64_t def = 0) const;
    int64_t i64(const std::string& key, int64_t def = 0) const;
    float f32(const std::string& key, float def = 0) const;
    bool boolean(const std::string& key, bool def = false) const;
    std::string str(const std::string& key, const std::string& def = "") const;
    bool has(const std::string& key) const;

    const std::vector<std::string>& arr_str(const std::string& key) const;
    const std::vector<float>& arr_f32(const std::string& key) const;
    const std::vector<int32_t>& arr_i32(const std::string& key) const;

    const std::string& arch() const { return arch_; }
    const std::vector<Tensor>& tensors() const { return tensors_; }
    size_t file_size() const { return file_.size(); }
    std::string kv_arch(const char* field) const { return arch_ + "." + field; }

    // Weight bytes actually referenced by named tensors (for bandwidth ceiling).
    size_t weight_bytes() const;

private:
    MappedFile file_;
    std::string arch_;
    std::vector<Tensor> tensors_;
    std::unordered_map<std::string, int> by_name_;
    std::unordered_map<std::string, uint64_t> u64_;
    std::unordered_map<std::string, int64_t> i64_;
    std::unordered_map<std::string, float> f32_;
    std::unordered_map<std::string, bool> bool_;
    std::unordered_map<std::string, std::string> str_;
    std::unordered_map<std::string, std::vector<std::string>> arr_str_;
    std::unordered_map<std::string, std::vector<float>> arr_f32_;
    std::unordered_map<std::string, std::vector<int32_t>> arr_i32_;
    static const std::vector<std::string> k_empty_str;
    static const std::vector<float> k_empty_f;
    static const std::vector<int32_t> k_empty_i;
};

// Minimal GGUF writer used by the CI tiny-model generator.
void write_gguf(const std::string& path,
                const std::unordered_map<std::string, std::string>& meta_str,
                const std::unordered_map<std::string, uint64_t>& meta_u64,
                const std::unordered_map<std::string, float>& meta_f32,
                const std::vector<std::string>& tok_list,
                const std::vector<Tensor>& tensors_desc,
                const std::vector<std::vector<uint8_t>>& payloads);

}  // namespace tinf
