//
// Created by 19373 on 2025/9/6.
//

#ifndef FG42_AUTOTOKENIZER_H
#define FG42_AUTOTOKENIZER_H
#include <string>
#include <vector>
#include "external/tokenizers-cpp/include/tokenizers_cpp.h"

namespace fg42 {
    class AutoTokenizer {
    private:
        std::unique_ptr<tokenizers::Tokenizer> tok_;

    public:
        explicit AutoTokenizer(const std::string& tokenizer_json_file_path);
        ~AutoTokenizer() = default;

        [[nodiscard]] std::size_t vocab_size() const;

        std::vector<std::int32_t> encode(const std::string& text);
        std::string decode(const std::vector<std::int32_t>& input_ids);
    };
} // fg42

#endif //FG42_AUTOTOKENIZER_H