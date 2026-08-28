#pragma once

#include "gguf.hpp"
#include <unordered_map>

namespace tinf {

class Tokenizer {
public:
    explicit Tokenizer(const GGUF& g);

    std::vector<int> encode(const std::string& text, bool add_bos) const;
    std::string decode(const std::vector<int>& ids) const;
    std::string decode_one(int id) const;

    int bos() const { return bos_; }
    int eos() const { return eos_; }
    int n_vocab() const { return (int)vocab_.size(); }

private:
    std::vector<std::string> vocab_;
    std::unordered_map<std::string, int> token_to_id_;
    std::unordered_map<std::string, int> ranks_;
    std::vector<std::string> specials_;
    std::string byte_to_piece_[256];
    int byte_from_piece_[1 << 16];
    int bos_ = 1;
    int eos_ = 2;
    bool add_bos_ = true;

    std::vector<int> bpe(const std::string& piece) const;
    void split_special(const std::string& text, std::vector<std::string>& chunks) const;
};

}  // namespace tinf
