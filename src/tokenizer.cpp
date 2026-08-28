#include "tokenizer.hpp"
#include <algorithm>
#include <cctype>
#include <limits>
#include <sstream>
#include <unordered_set>

namespace tinf {

static void gpt2_bytes_unicode(std::string byte_to_piece[256], int byte_from_piece[1 << 16]) {
    for (int i = 0; i < (1 << 16); ++i) byte_from_piece[i] = -1;
    std::vector<int> bs;
    for (int i = 33; i <= 126; ++i) bs.push_back(i);
    for (int i = 161; i <= 172; ++i) bs.push_back(i);
    for (int i = 174; i <= 255; ++i) bs.push_back(i);
    std::vector<int> cs = bs;
    int n = 0;
    for (int b = 0; b < 256; ++b) {
        if (std::find(bs.begin(), bs.end(), b) == bs.end()) {
            bs.push_back(b);
            cs.push_back(256 + n);
            ++n;
        }
    }
    for (size_t i = 0; i < bs.size(); ++i) {
        int cp = cs[i];
        std::string s;
        if (cp < 0x80) {
            s.push_back((char)cp);
        } else if (cp < 0x800) {
            s.push_back((char)(0xC0 | (cp >> 6)));
            s.push_back((char)(0x80 | (cp & 0x3F)));
        } else {
            s.push_back((char)(0xE0 | (cp >> 12)));
            s.push_back((char)(0x80 | ((cp >> 6) & 0x3F)));
            s.push_back((char)(0x80 | (cp & 0x3F)));
        }
        byte_to_piece[bs[i]] = s;
        byte_from_piece[cp] = bs[i];
    }
}

Tokenizer::Tokenizer(const GGUF& g) {
    gpt2_bytes_unicode(byte_to_piece_, byte_from_piece_);
    vocab_ = g.arr_str("tokenizer.ggml.tokens");
    token_to_id_.reserve(vocab_.size());
    for (int i = 0; i < (int)vocab_.size(); ++i) token_to_id_[vocab_[i]] = i;

    const auto& merges = g.arr_str("tokenizer.ggml.merges");
    for (int i = 0; i < (int)merges.size(); ++i) ranks_[merges[i]] = i;

    bos_ = (int)g.u64("tokenizer.ggml.bos_token_id", 1);
    eos_ = (int)g.u64("tokenizer.ggml.eos_token_id", 2);
    add_bos_ = g.boolean("tokenizer.ggml.add_bos_token", true);

    for (int i = 0; i < (int)vocab_.size(); ++i) {
        const std::string& t = vocab_[i];
        if (t.size() >= 3 && t.front() == '<' && t.back() == '>') specials_.push_back(t);
    }
    std::sort(specials_.begin(), specials_.end(),
              [](const std::string& a, const std::string& b) { return a.size() > b.size(); });
}

void Tokenizer::split_special(const std::string& text, std::vector<std::string>& chunks) const {
    size_t i = 0;
    while (i < text.size()) {
        bool hit = false;
        for (const auto& sp : specials_) {
            if (i + sp.size() <= text.size() && text.compare(i, sp.size(), sp) == 0) {
                chunks.push_back(sp);
                i += sp.size();
                hit = true;
                break;
            }
        }
        if (hit) continue;
        size_t j = i + 1;
        while (j < text.size()) {
            bool nxt = false;
            for (const auto& sp : specials_) {
                if (j + sp.size() <= text.size() && text.compare(j, sp.size(), sp) == 0) {
                    nxt = true;
                    break;
                }
            }
            if (nxt) break;
            ++j;
        }
        chunks.push_back(text.substr(i, j - i));
        i = j;
    }
}

static std::vector<std::string> pretok(const std::string& s) {
    // Approximate Llama-3 / GPT-2 split: keep a leading space with the next word.
    std::vector<std::string> out;
    size_t i = 0;
    while (i < s.size()) {
        if (std::isspace((unsigned char)s[i])) {
            size_t j = i;
            while (j < s.size() && std::isspace((unsigned char)s[j])) ++j;
            // attach last space to the following word if any
            if (j < s.size() && !std::isspace((unsigned char)s[j])) {
                size_t k = j;
                while (k < s.size() && !std::isspace((unsigned char)s[k]) &&
                       (std::isalnum((unsigned char)s[k]) || s[k] == '\''))
                    ++k;
                out.push_back(s.substr(j - 1, k - (j - 1)));
                i = k;
            } else {
                out.push_back(s.substr(i, j - i));
                i = j;
            }
            continue;
        }
        size_t j = i;
        if (std::isalpha((unsigned char)s[i]) || s[i] == '\'') {
            while (j < s.size() && (std::isalnum((unsigned char)s[j]) || s[j] == '\'')) ++j;
        } else if (std::isdigit((unsigned char)s[i])) {
            while (j < s.size() && std::isdigit((unsigned char)s[j])) ++j;
        } else {
            ++j;
        }
        out.push_back(s.substr(i, j - i));
        i = j;
    }
    return out;
}

std::vector<int> Tokenizer::bpe(const std::string& piece) const {
    auto it = token_to_id_.find(piece);
    if (it != token_to_id_.end()) return {it->second};

    std::string mapped;
    for (unsigned char ch : piece) mapped += byte_to_piece_[ch];

    std::vector<std::string> word;
    // split mapped into unicode-ish characters (1-3 byte utf8 from our table)
    for (size_t i = 0; i < mapped.size();) {
        size_t n = 1;
        unsigned char c = (unsigned char)mapped[i];
        if ((c & 0x80) == 0) n = 1;
        else if ((c & 0xE0) == 0xC0) n = 2;
        else n = 3;
        word.push_back(mapped.substr(i, n));
        i += n;
    }
    if (word.empty()) return {};

    while (word.size() > 1) {
        int best = std::numeric_limits<int>::max();
        int best_i = -1;
        for (int i = 0; i + 1 < (int)word.size(); ++i) {
            std::string pair = word[i] + " " + word[i + 1];
            auto r = ranks_.find(pair);
            if (r != ranks_.end() && r->second < best) {
                best = r->second;
                best_i = i;
            }
        }
        if (best_i < 0) break;
        word[best_i] = word[best_i] + word[best_i + 1];
        word.erase(word.begin() + best_i + 1);
    }

    std::vector<int> ids;
    ids.reserve(word.size());
    for (const auto& w : word) {
        auto id = token_to_id_.find(w);
        if (id == token_to_id_.end()) {
            // byte fallback
            for (unsigned char ch : w) {
                auto bid = token_to_id_.find(byte_to_piece_[ch]);
                if (bid != token_to_id_.end()) ids.push_back(bid->second);
            }
        } else {
            ids.push_back(id->second);
        }
    }
    return ids;
}

std::vector<int> Tokenizer::encode(const std::string& text, bool add_bos) const {
    std::vector<int> ids;
    if (add_bos && add_bos_) ids.push_back(bos_);
    if (vocab_.empty()) return ids;
    std::vector<std::string> chunks;
    split_special(text, chunks);
    for (const auto& ch : chunks) {
        auto sp = token_to_id_.find(ch);
        if (sp != token_to_id_.end()) {
            ids.push_back(sp->second);
            continue;
        }
        for (const auto& piece : pretok(ch)) {
            auto more = bpe(piece);
            ids.insert(ids.end(), more.begin(), more.end());
        }
    }
    return ids;
}

std::string Tokenizer::decode_one(int id) const {
    if (id < 0 || id >= (int)vocab_.size()) return "";
    const std::string& t = vocab_[id];
    std::string out;
    for (size_t i = 0; i < t.size();) {
        unsigned char c = (unsigned char)t[i];
        int cp = 0, n = 1;
        if ((c & 0x80) == 0) {
            cp = c;
            n = 1;
        } else if ((c & 0xE0) == 0xC0 && i + 1 < t.size()) {
            cp = ((c & 0x1F) << 6) | ((unsigned char)t[i + 1] & 0x3F);
            n = 2;
        } else if ((c & 0xF0) == 0xE0 && i + 2 < t.size()) {
            cp = ((c & 0x0F) << 12) | (((unsigned char)t[i + 1] & 0x3F) << 6) |
                 ((unsigned char)t[i + 2] & 0x3F);
            n = 3;
        } else {
            out.push_back((char)c);
            ++i;
            continue;
        }
        int b = (cp < (1 << 16)) ? byte_from_piece_[cp] : -1;
        if (b >= 0) out.push_back((char)b);
        else out.append(t, i, n);
        i += n;
    }
    return out;
}

std::string Tokenizer::decode(const std::vector<int>& ids) const {
    std::string s;
    for (int id : ids) s += decode_one(id);
    return s;
}

}  // namespace tinf
