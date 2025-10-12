//
// Created by B777B2056-2 on 2025/10/10.
//

#ifndef FG42_MODEL_COMMON_H
#define FG42_MODEL_COMMON_H
#include <string>
#include <unordered_map>
#include "tensor/Tensor.h"

namespace fg42 {
    typedef std::unordered_map<std::string, Tensor> StateDict;

    struct ModelConfig {
        std::string architecture;
        std::size_t bos_token_id = 0;
        std::size_t eos_token_id = 0;
        std::size_t num_attention_heads = 0;
        std::size_t num_key_value_heads = 0;
        std::size_t hidden_size = 0;
        std::size_t num_hidden_layers = 0;
        std::size_t vocab_size = 0;
        std::size_t max_position_embeddings = 0;
        float rms_norm_eps = 0.0f;
        float rope_theta = 0.0f;
        bool use_cache = false;
        DataType data_type = DataType::Unknown;
    };
}   // namespace fg42

#endif //FG42_MODEL_COMMON_H