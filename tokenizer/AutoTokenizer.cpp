//
// Created by 19373 on 2025/9/6.
//
#include <fstream>
#include <ostream>
#include "tokenizer/AutoTokenizer.h"

namespace fg42 {
    static std::string load_bytes_from_file(const std::string& path) {
        std::ifstream fs(path, std::ios::in | std::ios::binary);
        if (fs.fail()) {
            throw std::runtime_error("Failed to open file " + path);
        }
        std::string data;
        fs.seekg(0, std::ios::end);
        size_t size = fs.tellg();
        fs.seekg(0, std::ios::beg);
        data.resize(size);
        fs.read(data.data(), size);
        return data;
    }

    AutoTokenizer::AutoTokenizer(const std::string& tokenizer_json_file_path) :tok_(nullptr) {
        auto blob = load_bytes_from_file(tokenizer_json_file_path);
        this->tok_ = tokenizers::Tokenizer::FromBlobJSON(blob);
    }

    std::size_t AutoTokenizer::vocab_size() const {
        return this->tok_->GetVocabSize();
    }

    std::vector<std::int32_t> AutoTokenizer::encode(const std::string& text) {
        return this->tok_->Encode(text);
    }

    std::string AutoTokenizer::decode(const std::vector<std::int32_t>& input_ids) {
        return this->tok_->Decode(input_ids);
    }
} // fg42