//
// Created by 19373 on 2025/9/6.
//
#include <filesystem>
#include <fstream>
#include "nlohmann/json.hpp"
#include "tokenizer/AutoTokenizer.h"

namespace fg42 {
    namespace fs = std::filesystem;

    static std::string load_bytes_from_file(const fs::path& p) {
        std::ifstream fs(p, std::ios::in | std::ios::binary);
        if (fs.fail()) {
            throw std::runtime_error("Failed to open file " + p.string());
        }
        std::string data;
        fs.seekg(0, std::ios::end);
        size_t size = fs.tellg();
        fs.seekg(0, std::ios::beg);
        data.resize(size);
        fs.read(data.data(), size);
        return data;
    }

    AutoTokenizer::AutoTokenizer(const std::string& tokenizer_dir_path) : chat_template_(), tok_(nullptr) {
        this->load_vocabulary(tokenizer_dir_path);
        this->load_config(tokenizer_dir_path);
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

    std::string AutoTokenizer::apply_chat_template(const MessageType& messages) {
        // 构建模板
        auto context = this->build_chat_template_context(messages);
        // 渲染模板
        auto result = this->chat_template_.RenderAsString(context);
        if (!result.has_value()) {
            return "";
        }
        return result.value();
    }

    void AutoTokenizer::load_vocabulary(const std::string& tokenizer_dir_path) {
        fs::path dir_path = tokenizer_dir_path;
        fs::path tokenizer_json_file_path = dir_path / "tokenizer.json";

        auto blob = load_bytes_from_file(tokenizer_json_file_path);
        this->tok_ = tokenizers::Tokenizer::FromBlobJSON(blob);
    }

    void AutoTokenizer::load_config(const std::string& tokenizer_dir_path) {
        fs::path dir_path = tokenizer_dir_path;
        fs::path tokenizer_config_json_file_path = dir_path / "tokenizer_config.json";

        std::ifstream fs(tokenizer_config_json_file_path, std::ios::in);
        if (fs.fail()) {
            throw std::runtime_error("Failed to open file " + tokenizer_config_json_file_path.string());
        }

        auto config_json = nlohmann::json::parse(fs);
        this->chat_template_.Load(config_json["chat_template"].get<std::string>());

        if (config_json.find("bos_token") != config_json.end() && !config_json["bos_token"].is_null()) {
            this->bos_token_ = config_json["bos_token"].get<std::string>();
        }
        if (config_json.find("eos_token") != config_json.end() && !config_json["eos_token"].is_null()) {
            this->eos_token_ = config_json["eos_token"].get<std::string>();
        }
    }

    jinja2::ValuesMap AutoTokenizer::build_chat_template_context(const MessageType& messages) const {
        // 将消息转换为模板渲染器期望的格式
        jinja2::ValuesList msg_list;
        for (const auto& msg : messages) {
            // 确保每条消息都有role和content字段
            if ((msg.find("role") != msg.end()) && (msg.find("content") != msg.end())) {
                jinja2::ValuesMap tmp_map = {
                    {"role", msg.at("role")},
                    {"content", msg.at("content")},
                };
                msg_list.emplace_back(tmp_map);
            }
        }

        // 创建模板上下文并添加消息
        jinja2::ValuesMap context = {
            {"messages", msg_list},
        };

        // 添加特殊标记（如果tokenizer中有定义）
        if (this->bos_token_.has_value()) {
            context.emplace("bos_token", this->bos_token_.value());
        }
        if (this->eos_token_.has_value()) {
            context.emplace("eos_token", this->eos_token_.value());
        }
        return context;
    }
} // fg42