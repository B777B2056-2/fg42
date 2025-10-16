//
// Created by B777B2056-2 on 2025/9/5.
//
#include <algorithm>
#include "model/Qwen2.h"
#include "sampler/Sampler.h"

namespace fg42 {
    Qwen2ForCausalLM::Qwen2ForCausalLM(const std::string& dir_path, DeviceType device_type,
            std::int32_t padding_idx, KVCacheImpl kv_cache_impl)
        : BaseModel(dir_path, device_type, padding_idx, kv_cache_impl),
        embedding_layer_(model_config_), norm_layer_(model_config_, "norm"),
        lm_head_layer_(model_config_, embedding_layer_) {
        this->load_model_weights(dir_path);
        this->init_weights();
    }

    Tensor Qwen2ForCausalLM::forward(const Tensor& input) {
        // 1. 输入Embedding
        auto emb_output = this->embedding_layer_.forward({&input});
        // 2. Decoder
        Tensor hidden_states = emb_output;
        for (auto&& module : this->module_list_) {
            hidden_states = module->forward({&hidden_states});
        }
        // 3. 输出RMSNorm
        auto norm_output = this->norm_layer_.forward({&hidden_states});
        // 4. 输出线性层
        auto lm_head_output = this->lm_head_layer_.forward({&norm_output});
        // 5. 返回
        return lm_head_output;
    }

    bool Qwen2ForCausalLM::weight_need_transpose(const std::string& key) const {
        // Linear层权重需要转置后存入
        std::vector<std::string> linear_layer_name_key_words = {
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        };

        return std::any_of(linear_layer_name_key_words.begin(),
            linear_layer_name_key_words.end(),
            [key](const std::string& name) -> bool {
                if (key.find("weight") == std::string::npos) {
                    return false;   // 确保只有权重转置
                }
                if (key.find(name) != std::string::npos) {
                    return true;
                }
                return false;
            });
    }

    void Qwen2ForCausalLM::init_weights() {
        // 初始化Embedding层权重
        this->embedding_layer_.init_weights(this->state_dict_);
        // 初始化Decoder权重
        for (int i = 0; i < this->model_config_.num_hidden_layers; ++i) {
            auto decoder_layer = std::make_unique<Qwen2DecoderLayer>(i, kv_cache_.get(), this->model_config_);
            decoder_layer->init_weights(this->state_dict_);
            this->module_list_.emplace_back(std::move(decoder_layer));
        }
        // 初始化RMSNorm权重
        this->norm_layer_.init_weights(this->state_dict_);
        // 初始化lm_head层权重
        this->lm_head_layer_.init_weights(this->state_dict_);
    }

    void Qwen2ForCausalLM::set_seq_len(std::size_t seq_len) {
        for (auto&& module : this->module_list_) {
            module->set_seq_len(seq_len);
        }
    }

    void Qwen2ForCausalLM::clear_kv_cache() {
        kv_cache_->clear();
    }
} // fg42