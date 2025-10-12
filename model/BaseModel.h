//
// Created by B777B2056-2 on 2025/9/4.
//

#ifndef FG42_MODELLOADER_H
#define FG42_MODELLOADER_H
#include <string>
#include <unordered_map>
#include <memory>
#include <tuple>
#include "cache/KVCache.h"
#include "nlohmann/json.hpp"
#include "model/Common.h"
#include "sampler/Sampler.h"

namespace fg42 {
    class BaseModel {
    public:
        using StreamHandler = std::function<void(std::size_t, std::int32_t)>;

        BaseModel() = delete;
        BaseModel(const std::string& dir_path, DeviceType device_type,
            std::int32_t padding_idx, DataType data_type=DataType::Unknown,
            KVCacheImpl kv_cache_impl=KVCacheImpl::Dynamic);
        BaseModel(const BaseModel&) = delete;
        BaseModel& operator=(const BaseModel&) = delete;
        BaseModel(BaseModel&&) = delete;
        BaseModel& operator=(BaseModel&&) = delete;
        virtual ~BaseModel() = default;

        std::vector<std::vector<std::int32_t>> generate(SamplerConfig sampler_config,
            const std::vector<std::vector<std::int32_t>>& input_ids, std::size_t max_length,
            StreamHandler stream_handler=nullptr);

    protected:
        DeviceType device_type_;
        ModelConfig model_config_;
        StateDict state_dict_;
        std::unique_ptr<BaseKVCache> kv_cache_;
        std::int32_t padding_idx_;
        KVCacheImpl kv_cache_impl_;
        std::unordered_map<std::size_t, bool> token_generate_status_;   // key为batch idx, value为是否生成完成

        void load_model_config(const std::string& dir_path, DataType data_type);
        void load_model_weights(const std::string& dir_path, DataType data_type);
        [[nodiscard]] virtual bool weight_need_transpose(const std::string& key) const = 0;
        virtual Tensor forward(const Tensor& input) = 0;

        [[nodiscard]] Tensor get_token_from_last_logits(SamplerConfig config, const Tensor& outputs) const;
        void parse_tokens_from_tensor(const Tensor& tokens_tensor,
            std::vector<std::vector<std::int32_t>>& tokens, StreamHandler stream_handler) const;
        void init_token_generate_status(std::size_t batch_size);
        [[nodiscard]] bool check_finished(const std::vector<std::vector<std::int32_t>>& tokens, std::size_t max_length);

        [[nodiscard]] std::tuple<Tensor, std::size_t> prefill(SamplerConfig config,
            const Tensor& input_ids_batch);
        [[nodiscard]] std::vector<std::vector<std::int32_t>> decode(SamplerConfig config,
            const Tensor& prefilled_batch, std::size_t seq_len, std::size_t max_length, StreamHandler stream_handler);

        void left_padding(std::vector<std::vector<std::int32_t>>& input_ids) const;
        void output_tokens_post_processing(std::size_t after_padding_seq_len,
            std::vector<std::vector<std::int32_t>>& tokens) const;

        virtual void set_seq_len(std::size_t seq_len) = 0;
        virtual void clear_kv_cache() = 0;
    };
} // fg42

#endif //FG42_MODELLOADER_H