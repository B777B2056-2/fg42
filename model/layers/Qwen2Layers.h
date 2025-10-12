//
// Created by B777B2056-2 on 2025/9/18.
//

#ifndef FG42_QWEN2LAYERS_H
#define FG42_QWEN2LAYERS_H
#include "model/layers/BaseLayer.h"
#include <tuple>
#include "operator/Attention.h"

namespace fg42 {
    class Qwen2EmbeddingLayer final : public BaseLayer {
    public:
        explicit Qwen2EmbeddingLayer(const ModelConfig& model_config);
        ~Qwen2EmbeddingLayer() override = default;

        void init_weights(const StateDict& state_dict) override;
        Tensor forward(const std::vector<const Tensor*>& input) override;
        [[nodiscard]] const Tensor* get_embedding_weight() const;

    private:
        const Tensor* embedding_weight_ = nullptr;
    };

    class Qwen2RotaryEmbeddingLayer final : public BaseLayer {
    public:
        Qwen2RotaryEmbeddingLayer(DataType data_type, DeviceType device_type, const ModelConfig& model_config);
        ~Qwen2RotaryEmbeddingLayer() override = default;

        std::tuple<Tensor, Tensor> apply_rotary_pos_emb(const Tensor& q, const Tensor& k, std::size_t seq_len);

    private:
        DataType data_type_;
        DeviceType device_type_;
        std::size_t dim_;
        std::size_t max_position_embeddings_;
        float base_;
        std::size_t max_seq_len_cached_;
        Tensor inv_freq_;
        Tensor cos_cached_;
        Tensor sin_cached_;

        void init_inv_freq(std::size_t dim, float base);
        void set_cos_sin_cache(std::size_t seq_len);
        std::tuple<Tensor, Tensor> get_cached_cos_sin(std::size_t seq_len, std::size_t kv_seq_len);
        [[nodiscard]] Tensor rotate_half(const Tensor& x) const;

        void init_weights(const StateDict&) override {}
        Tensor forward(const std::vector<const Tensor*>&) override { return {}; }
    };

    class Qwen2AttentionLayer final : public BaseLayer {
    public:
        Qwen2AttentionLayer(int idx, BaseKVCache* cache, const ModelConfig& model_config);
        ~Qwen2AttentionLayer() override;

        void init_weights(const StateDict& state_dict) override;
        Tensor forward(const std::vector<const Tensor*>& input) override;

        [[nodiscard]] std::size_t get_seq_len() const;
        void set_seq_len(std::size_t seq_len);

    private:
        int idx_in_block_;
        BaseKVCache* kv_cache_;
        std::size_t seq_len_;
        Qwen2RotaryEmbeddingLayer* rope_impl_;
        kernel::MultiQueryAttention* mqa_;
    };

    class Qwen2FFNLayer final : public BaseLayer {
    public:
        Qwen2FFNLayer(int idx, const ModelConfig& model_config);
        ~Qwen2FFNLayer() override = default;

        void init_weights(const StateDict& state_dict) override;
        Tensor forward(const std::vector<const Tensor*>& input) override;

    private:
        int idx_in_block_ = -1;
        const Tensor* mlp_gate_weight_ = nullptr;
        const Tensor* mlp_up_weight_ = nullptr;
        const Tensor* mlp_down_weight_ = nullptr;

        const Tensor* mlp_gate_bias_ = nullptr;
        const Tensor* mlp_up_bias_ = nullptr;
        const Tensor* mlp_down_bias_ = nullptr;
    };

    class Qwen2RMSNormLayer final : public BaseLayer {
    public:
        explicit Qwen2RMSNormLayer(const ModelConfig& model_config, std::string name = "");
        Qwen2RMSNormLayer(int idx, const ModelConfig& model_config, std::string name = "");
        ~Qwen2RMSNormLayer() override = default;

        void init_weights(const StateDict& state_dict) override;;
        Tensor forward(const std::vector<const Tensor*>& input) override;

    private:
        float eps_;
        int idx_in_block_ = -1;
        const Tensor* weight_ = nullptr;
    };

    class Qwen2DecoderLayer final : public BaseLayer {
    public:
        Qwen2DecoderLayer(int idx, BaseKVCache* cache, const ModelConfig& model_config);
        ~Qwen2DecoderLayer() override = default;

        void init_weights(const StateDict& state_dict) override;
        Tensor forward(const std::vector<const Tensor*>& input) override;

        [[nodiscard]] std::size_t get_seq_len() const;
        void set_seq_len(std::size_t seq_len);

    private:
        Qwen2AttentionLayer attention_layer_;
        Qwen2FFNLayer ffn_layer_;
        Qwen2RMSNormLayer input_norm_layer_;
        Qwen2RMSNormLayer post_attention_norm_layer_;
    };

    class Qwen2LMHeadLayer final : public BaseLayer {
    public:
        Qwen2LMHeadLayer(const ModelConfig& model_config, const Qwen2EmbeddingLayer& embedding_layer);
        ~Qwen2LMHeadLayer() override;

        void init_weights(const StateDict& state_dict) override;
        Tensor forward(const std::vector<const Tensor*>& input) override;

    private:
        bool is_from_embedding_;
        const Qwen2EmbeddingLayer* embedding_layer_;
        const Tensor* lm_head_weight_;
        const Tensor* lm_head_bias_;
    };
}   // fg42

#endif //FG42_QWEN2LAYERS_H