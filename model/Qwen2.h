//
// Created by B777B2056-2 on 2025/9/5.
//

#ifndef FG42_QWEN2_H
#define FG42_QWEN2_H
#include "model/BaseModel.h"
#include "model/layers/Qwen2Layers.h"
#include <vector>
#include <memory>

namespace fg42 {
    class Qwen2ForCausalLM final : public BaseModel {
    public:
        Qwen2ForCausalLM() = delete;
        Qwen2ForCausalLM(const std::string& dir_path, DeviceType device_type,
            std::int32_t padding_idx, DataType data_type=DataType::Unknown,
            KVCacheImpl kv_cache_impl=KVCacheImpl::Dynamic);
        ~Qwen2ForCausalLM() override = default;

    private:
        Qwen2EmbeddingLayer embedding_layer_;
        std::vector<std::unique_ptr<Qwen2DecoderLayer>> module_list_;
        Qwen2RMSNormLayer norm_layer_;
        Qwen2LMHeadLayer lm_head_layer_;

        [[nodiscard]] bool weight_need_transpose(const std::string& key) const override;
        void init_weights();

        Tensor forward(const Tensor& input) override;

        void set_seq_len(std::size_t seq_len) override;
        void clear_kv_cache() override;
    };
} // fg42

#endif //FG42_QWEN2_H