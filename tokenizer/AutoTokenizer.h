//
// Created by 19373 on 2025/9/6.
//

#ifndef FG42_AUTOTOKENIZER_H
#define FG42_AUTOTOKENIZER_H
#include <string>
#include <vector>
#include <optional>
#include <unordered_map>
#include "external/tokenizers-cpp/include/tokenizers_cpp.h"
#include "external/Jinja2Cpp/include/jinja2cpp/template.h"
#include "tensor/Tensor.h"

namespace fg42 {
    class AutoTokenizer {
    private:
        jinja2::Template chat_template_;
        std::optional<std::string> bos_token_;
        std::optional<std::string> eos_token_;
        std::unique_ptr<tokenizers::Tokenizer> tok_;

    public:
        typedef std::vector<std::unordered_map<std::string, std::string>> MessageType;

        explicit AutoTokenizer(const std::string& tokenizer_dir_path);
        ~AutoTokenizer() = default;

        [[nodiscard]] std::size_t vocab_size() const;

        std::vector<std::int32_t> encode(const std::string& text);
        std::string decode(const std::vector<std::int32_t>& input_ids);

        Tensor encode_to_tensor(const std::string& text);
        std::string decode_from_tensor(const Tensor& input_tensor);

        std::string apply_chat_template(const MessageType& messages);

    private:
        void load_vocabulary(const std::string& tokenizer_dir_path);
        void load_config(const std::string& tokenizer_dir_path);
        [[nodiscard]] jinja2::ValuesMap build_chat_template_context(const MessageType& messages) const;
    };
} // fg42

#endif //FG42_AUTOTOKENIZER_H