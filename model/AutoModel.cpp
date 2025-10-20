//
// Created by 19373 on 2025/10/20.
//

#include "model/AutoModel.h"
#include "model/Qwen2.h"

namespace fg42 {
    AutoModel::AutoModel(const std::string& dir_path, DeviceType device_type, std::int32_t padding_idx,
            KVCacheImpl kv_cache_impl) {
        auto model_config = parse_model_config(dir_path);
        if (model_config.architecture == "Qwen2ForCausalLM") {
            model_ = std::make_unique<Qwen2ForCausalLM>(dir_path, device_type, padding_idx, kv_cache_impl);
        }
    }

    std::vector<std::vector<std::int32_t>> AutoModel::generate(SamplerConfig sampler_config,
        const std::vector<std::vector<std::int32_t>>& input_ids, std::size_t max_length,
        BaseModel::StreamHandler stream_handler) {
        return model_->generate(sampler_config, input_ids, max_length, stream_handler);
    }
} // fg42