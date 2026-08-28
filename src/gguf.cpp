#include "gguf.hpp"
#include "quant.hpp"
#include <fstream>

namespace tinf {

const std::vector<std::string> GGUF::k_empty_str;
const std::vector<float> GGUF::k_empty_f;
const std::vector<int32_t> GGUF::k_empty_i;

namespace {

struct Cursor {
    const uint8_t* p;
    const uint8_t* end;
    explicit Cursor(const uint8_t* b, size_t n) : p(b), end(b + n) {}
    void need(size_t n) const {
        if (p + n > end) throw std::runtime_error("GGUF truncated");
    }
    template <typename T>
    T rd() {
        need(sizeof(T));
        T v;
        std::memcpy(&v, p, sizeof(T));
        p += sizeof(T);
        return v;
    }
    std::string rdstr() {
        uint64_t n = rd<uint64_t>();
        if (n > (uint64_t)(end - p)) throw std::runtime_error("GGUF string overflow");
        std::string s((const char*)p, (const char*)p + (size_t)n);
        p += n;
        return s;
    }
};

void skip_value(Cursor& c, uint32_t type);

void skip_array(Cursor& c) {
    uint32_t et = c.rd<uint32_t>();
    uint64_t n = c.rd<uint64_t>();
    for (uint64_t i = 0; i < n; ++i) skip_value(c, et);
}

void skip_value(Cursor& c, uint32_t type) {
    switch (type) {
        case 0: c.rd<uint8_t>(); break;
        case 1: c.rd<int8_t>(); break;
        case 2: c.rd<uint16_t>(); break;
        case 3: c.rd<int16_t>(); break;
        case 4: c.rd<uint32_t>(); break;
        case 5: c.rd<int32_t>(); break;
        case 6: c.rd<float>(); break;
        case 7: c.rd<uint8_t>(); break;
        case 8: c.rdstr(); break;
        case 9: skip_array(c); break;
        case 10: c.rd<uint64_t>(); break;
        case 11: c.rd<int64_t>(); break;
        case 12: c.rd<double>(); break;
        default: throw std::runtime_error("unknown GGUF type");
    }
}

uint64_t as_u64(Cursor& c, uint32_t type) {
    switch (type) {
        case 0: return c.rd<uint8_t>();
        case 1: return (uint64_t)(int64_t)c.rd<int8_t>();
        case 2: return c.rd<uint16_t>();
        case 3: return (uint64_t)(int64_t)c.rd<int16_t>();
        case 4: return c.rd<uint32_t>();
        case 5: return (uint64_t)(int64_t)c.rd<int32_t>();
        case 10: return c.rd<uint64_t>();
        case 11: return (uint64_t)c.rd<int64_t>();
        default: throw std::runtime_error("not an integer GGUF value");
    }
}

float as_f32(Cursor& c, uint32_t type) {
    if (type == 6) return c.rd<float>();
    if (type == 12) return (float)c.rd<double>();
    throw std::runtime_error("not a float GGUF value");
}

size_t align_up(size_t x, size_t a) { return (x + a - 1) / a * a; }

}  // namespace

GGUF::GGUF(const std::string& path) : file_(path) {
    Cursor c(file_.data(), file_.size());
    uint32_t magic = c.rd<uint32_t>();
    if (magic != 0x46554747) throw std::runtime_error("not a GGUF file: " + path);
    uint32_t version = c.rd<uint32_t>();
    if (version != 2 && version != 3) throw std::runtime_error("unsupported GGUF version");
    uint64_t n_tensors = c.rd<uint64_t>();
    uint64_t n_kv = c.rd<uint64_t>();

    for (uint64_t i = 0; i < n_kv; ++i) {
        std::string key = c.rdstr();
        uint32_t type = c.rd<uint32_t>();
        switch (type) {
            case 0: case 1: case 2: case 3: case 4: case 5: case 10: case 11:
                u64_[key] = as_u64(c, type);
                i64_[key] = (int64_t)u64_[key];
                break;
            case 6:
            case 12:
                f32_[key] = as_f32(c, type);
                break;
            case 7:
                bool_[key] = c.rd<uint8_t>() != 0;
                break;
            case 8:
                str_[key] = c.rdstr();
                break;
            case 9: {
                uint32_t et = c.rd<uint32_t>();
                uint64_t n = c.rd<uint64_t>();
                if (et == 8) {
                    auto& a = arr_str_[key];
                    a.resize((size_t)n);
                    for (uint64_t j = 0; j < n; ++j) a[j] = c.rdstr();
                } else if (et == 6) {
                    auto& a = arr_f32_[key];
                    a.resize((size_t)n);
                    for (uint64_t j = 0; j < n; ++j) a[j] = c.rd<float>();
                } else if (et == 4 || et == 5) {
                    auto& a = arr_i32_[key];
                    a.resize((size_t)n);
                    for (uint64_t j = 0; j < n; ++j)
                        a[j] = et == 4 ? (int32_t)c.rd<uint32_t>() : c.rd<int32_t>();
                } else {
                    for (uint64_t j = 0; j < n; ++j) skip_value(c, et);
                }
                break;
            }
            default:
                skip_value(c, type);
                break;
        }
    }

    arch_ = str("general.architecture", "llama");
    uint32_t alignment = (uint32_t)u64("general.alignment", 32);
    if (alignment == 0) alignment = 32;

    std::vector<uint64_t> offsets((size_t)n_tensors);
    tensors_.resize((size_t)n_tensors);
    for (uint64_t i = 0; i < n_tensors; ++i) {
        Tensor t;
        t.name = c.rdstr();
        t.ndim = (int)c.rd<uint32_t>();
        if (t.ndim < 1 || t.ndim > 4) throw std::runtime_error("bad ndim for " + t.name);
        for (int d = 0; d < t.ndim; ++d) t.shape[d] = (int64_t)c.rd<uint64_t>();
        t.dtype = dtype_from_ggml((int)c.rd<uint32_t>());
        offsets[i] = c.rd<uint64_t>();
        t.nbytes = tensor_nbytes(t.dtype, t.shape, t.ndim);
        tensors_[i] = std::move(t);
        by_name_[tensors_[i].name] = (int)i;
    }

    size_t data_off = align_up((size_t)(c.p - file_.data()), alignment);
    for (size_t i = 0; i < tensors_.size(); ++i) {
        size_t abs = data_off + (size_t)offsets[i];
        if (abs + tensors_[i].nbytes > file_.size())
            throw std::runtime_error("tensor out of range: " + tensors_[i].name);
        tensors_[i].data = file_.data() + abs;
    }
}

const Tensor* GGUF::find(const std::string& name) const {
    auto it = by_name_.find(name);
    if (it == by_name_.end()) return nullptr;
    return &tensors_[it->second];
}

const Tensor& GGUF::must(const std::string& name) const {
    const Tensor* t = find(name);
    if (!t) throw std::runtime_error("missing tensor " + name);
    return *t;
}

uint64_t GGUF::u64(const std::string& key, uint64_t def) const {
    auto it = u64_.find(key);
    return it == u64_.end() ? def : it->second;
}
int64_t GGUF::i64(const std::string& key, int64_t def) const {
    auto it = i64_.find(key);
    return it == i64_.end() ? def : it->second;
}
float GGUF::f32(const std::string& key, float def) const {
    auto it = f32_.find(key);
    return it == f32_.end() ? def : it->second;
}
bool GGUF::boolean(const std::string& key, bool def) const {
    auto it = bool_.find(key);
    return it == bool_.end() ? def : it->second;
}
std::string GGUF::str(const std::string& key, const std::string& def) const {
    auto it = str_.find(key);
    return it == str_.end() ? def : it->second;
}
bool GGUF::has(const std::string& key) const {
    return u64_.count(key) || f32_.count(key) || str_.count(key) || bool_.count(key) ||
           arr_str_.count(key) || arr_f32_.count(key) || arr_i32_.count(key);
}
const std::vector<std::string>& GGUF::arr_str(const std::string& key) const {
    auto it = arr_str_.find(key);
    return it == arr_str_.end() ? k_empty_str : it->second;
}
const std::vector<float>& GGUF::arr_f32(const std::string& key) const {
    auto it = arr_f32_.find(key);
    return it == arr_f32_.end() ? k_empty_f : it->second;
}
const std::vector<int32_t>& GGUF::arr_i32(const std::string& key) const {
    auto it = arr_i32_.find(key);
    return it == arr_i32_.end() ? k_empty_i : it->second;
}

size_t GGUF::weight_bytes() const {
    size_t n = 0;
    for (const auto& t : tensors_) n += t.nbytes;
    return n;
}

namespace {

void wr_raw(std::ostream& o, const void* p, size_t n) { o.write((const char*)p, (std::streamsize)n); }
template <typename T>
void wr(std::ostream& o, T v) { wr_raw(o, &v, sizeof(v)); }
void wrstr(std::ostream& o, const std::string& s) {
    wr<uint64_t>(o, s.size());
    wr_raw(o, s.data(), s.size());
}

}  // namespace

void write_gguf(const std::string& path,
                const std::unordered_map<std::string, std::string>& meta_str,
                const std::unordered_map<std::string, uint64_t>& meta_u64,
                const std::unordered_map<std::string, float>& meta_f32,
                const std::vector<std::string>& tok_list,
                const std::vector<Tensor>& tensors_desc,
                const std::vector<std::vector<uint8_t>>& payloads) {
    std::ofstream o(path, std::ios::binary);
    if (!o) throw std::runtime_error("cannot write " + path);
    wr<uint32_t>(o, 0x46554747);
    wr<uint32_t>(o, 3);
    wr<uint64_t>(o, tensors_desc.size());
    uint64_t n_kv = meta_str.size() + meta_u64.size() + meta_f32.size();
    if (!tok_list.empty()) n_kv += 1;
    wr<uint64_t>(o, n_kv);
    for (const auto& kv : meta_str) {
        wrstr(o, kv.first);
        wr<uint32_t>(o, 8);
        wrstr(o, kv.second);
    }
    for (const auto& kv : meta_u64) {
        wrstr(o, kv.first);
        wr<uint32_t>(o, 10);
        wr<uint64_t>(o, kv.second);
    }
    for (const auto& kv : meta_f32) {
        wrstr(o, kv.first);
        wr<uint32_t>(o, 6);
        wr<float>(o, kv.second);
    }
    if (!tok_list.empty()) {
        wrstr(o, "tokenizer.ggml.tokens");
        wr<uint32_t>(o, 9);
        wr<uint32_t>(o, 8);
        wr<uint64_t>(o, tok_list.size());
        for (const auto& t : tok_list) wrstr(o, t);
    }
    const uint32_t align = 32;
    std::vector<uint64_t> offs(tensors_desc.size());
    uint64_t acc = 0;
    for (size_t i = 0; i < tensors_desc.size(); ++i) {
        offs[i] = acc;
        acc += payloads[i].size();
        acc = (acc + align - 1) / align * align;
    }
    for (size_t i = 0; i < tensors_desc.size(); ++i) {
        const Tensor& t = tensors_desc[i];
        wrstr(o, t.name);
        wr<uint32_t>(o, (uint32_t)t.ndim);
        for (int d = 0; d < t.ndim; ++d) wr<uint64_t>(o, (uint64_t)t.shape[d]);
        wr<uint32_t>(o, (uint32_t)ggml_from_dtype(t.dtype));
        wr<uint64_t>(o, offs[i]);
    }
    size_t pos = (size_t)o.tellp();
    size_t pad = (align - (pos % align)) % align;
    for (size_t i = 0; i < pad; ++i) wr<uint8_t>(o, 0);
    for (size_t i = 0; i < payloads.size(); ++i) {
        wr_raw(o, payloads[i].data(), payloads[i].size());
        size_t p2 = (size_t)o.tellp();
        size_t pd = (align - (p2 % align)) % align;
        if (i + 1 < payloads.size())
            for (size_t j = 0; j < pd; ++j) wr<uint8_t>(o, 0);
    }
}

}  // namespace tinf
