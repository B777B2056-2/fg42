//
// Created by B777B2056-2 on 2025/9/18.
//
#include "model/layers/Qwen2Layers.h"

#include <iostream>
#include <utility>
#include "model/BaseModel.h"
#include "operator/ActivationOperator.h"
#include "operator/ArithmeticOperator.h"
#include "operator/Attention.h"
#include "operator/EmbeddingOperator.h"
#include "operator/NormOperator.h"
#include "util/util.h"

namespace fg42 {
    Qwen2EmbeddingLayer::Qwen2EmbeddingLayer(const ModelConfig &model_config)
    : BaseLayer(model_config, "embed_tokens") {}

    void Qwen2EmbeddingLayer::init_weights(const StateDict& state_dict) {
        const auto* embedding_weight = get_weight_from_state_dict(state_dict,
            build_weight_tensor_name(this->layer_name()), true);
        this->embedding_weight_ = embedding_weight;
    }

    Tensor Qwen2EmbeddingLayer::forward(const std::vector<const Tensor*>& input) {
        kernel::EmbeddingOperator op(*this->embedding_weight_);
        return op.forward(input, nullptr);
    }

    const Tensor * Qwen2EmbeddingLayer::get_embedding_weight() const {
        return this->embedding_weight_;
    }

    Qwen2RotaryEmbeddingLayer::Qwen2RotaryEmbeddingLayer(
        DataType data_type, DeviceType device_type, const ModelConfig& model_config)
        : BaseLayer(model_config, "rope"),
        data_type_(data_type), device_type_(device_type),
        dim_(model_config.hidden_size/model_config.num_attention_heads),
        max_position_embeddings_(model_config.max_position_embeddings), base_(model_config.rope_theta),
        max_seq_len_cached_(model_config.max_position_embeddings),
        inv_freq_(data_type, device_type, {dim_ / 2}) {
        this->init_inv_freq(dim_, model_config.rope_theta);
        this->set_cos_sin_cache(model_config.max_position_embeddings);
    }

    void Qwen2RotaryEmbeddingLayer::init_inv_freq(std::size_t dim, float base) {
        for (std::size_t b = 0; b < dim; b += 2) {
            float val = 1.0f / std::pow(base, static_cast<float>(b) / static_cast<float>(dim));
            inv_freq_.index_fill({b / 2}, val);
        }
    }

    void Qwen2RotaryEmbeddingLayer::set_cos_sin_cache(std::size_t seq_len) {
        this->max_seq_len_cached_ = seq_len;

        // 索引向量
        Tensor t(data_type_, device_type_, {max_seq_len_cached_});
        for (std::size_t b = 0; b < max_seq_len_cached_; ++b) {
            t.index_fill({b}, static_cast<float>(b));
        }

        // outer
        kernel::VecOuterOperator outer_op;
        auto freqs = outer_op.forward({&t, &inv_freq_}, nullptr);

        // concat
        auto emb = Tensor::concat(freqs, freqs, Tensor::ConcatDim::eColWise);

        // 计算正弦余弦
        kernel::CosineOperator cos_op;
        cos_cached_ = cos_op.forward({&emb}, nullptr);

        kernel::SineOperator sin_op;
        sin_cached_ = sin_op.forward({&emb}, nullptr);
    }

    std::tuple<Tensor, Tensor> Qwen2RotaryEmbeddingLayer::get_cached_cos_sin(std::size_t seq_len, std::size_t kv_seq_len) {
        if (seq_len > max_seq_len_cached_) {
            this->set_cos_sin_cache(seq_len);
        }

        Tensor cos, sin;

        // 如果使用了kv cache，且在decode阶段，则只返回索引在seq_len的行
        if (model_config_->use_cache && (kv_seq_len == 1)) {
            cos = cos_cached_.view({1, cos_cached_.shape().at(1)},
                cos_cached_.data({seq_len - 1, 0}));
            sin = sin_cached_.view({1, sin_cached_.shape().at(1)},
                sin_cached_.data({seq_len - 1, 0}));
        } else {
            cos = cos_cached_.view({seq_len, cos_cached_.shape().at(1)});
            sin = sin_cached_.view({seq_len, sin_cached_.shape().at(1)});
        }

        return std::make_tuple(cos, sin);
    }

    Tensor Qwen2RotaryEmbeddingLayer::rotate_half(const Tensor& x) const {
        kernel::RotateHalfOperator rotate_half_op;
        return rotate_half_op.forward({&x}, nullptr);
    }

    std::tuple<Tensor, Tensor> Qwen2RotaryEmbeddingLayer::apply_rotary_pos_emb(
        const Tensor& q, const Tensor& k, std::size_t seq_len) {
        if (seq_len == 0) {
            if (!model_config_->use_cache) {
                seq_len = q.shape().at(2);
            } else {
                throw std::runtime_error("seq_len must be set when use kv cache");
            }
        }
        std::size_t kv_seq_len = k.shape().at(2);

        auto cos_sin = get_cached_cos_sin(seq_len, kv_seq_len);
        auto cos = std::get<0>(cos_sin);
        auto sin = std::get<1>(cos_sin);

        auto embed_func = [this, &cos, &sin](const Tensor& x) -> Tensor {
            auto mul_tensor = x.mul(cos);
            auto rotate_half_tensor = rotate_half(x);
            auto rotate_half_mul_tensor = rotate_half_tensor.mul(sin);
            return mul_tensor + rotate_half_mul_tensor;
        };

        auto q_embed = embed_func(q);
        auto k_embed = embed_func(k);
        return std::make_tuple(q_embed, k_embed);
    }

    Qwen2AttentionLayer::Qwen2AttentionLayer(int idx, BaseKVCache* cache, const ModelConfig& model_config)
    : BaseLayer(model_config, "self_attn"), idx_in_block_(idx), kv_cache_(cache),
    seq_len_(0), rope_impl_(nullptr), mqa_(nullptr) {}

    Qwen2AttentionLayer::~Qwen2AttentionLayer() {
        delete rope_impl_;
        delete mqa_;
    }

    void Qwen2AttentionLayer::init_weights(const StateDict& state_dict) {
        const auto* q_proj_ = get_weight_from_state_dict(
            state_dict, build_weight_tensor_name(this->layer_name()+".q_proj", idx_in_block_), true);

        const auto* q_bias_ = get_weight_from_state_dict(
            state_dict, build_bias_tensor_name(this->layer_name()+".q_proj", idx_in_block_), true);

        const auto* k_proj_ = get_weight_from_state_dict(
            state_dict, build_weight_tensor_name(this->layer_name()+".k_proj", idx_in_block_), true);
        const auto* k_bias_ = get_weight_from_state_dict(
            state_dict, build_bias_tensor_name(this->layer_name()+".k_proj", idx_in_block_), true);

        const auto* v_proj_ = get_weight_from_state_dict(
            state_dict, build_weight_tensor_name(this->layer_name()+".v_proj", idx_in_block_), true);
        const auto* v_bias_ = get_weight_from_state_dict(
            state_dict, build_bias_tensor_name(this->layer_name()+".v_proj", idx_in_block_), true);

        const auto* o_proj_ = get_weight_from_state_dict(
            state_dict, build_weight_tensor_name(this->layer_name()+".o_proj", idx_in_block_), true);
        const auto* o_bias_ = get_weight_from_state_dict(
            state_dict, build_bias_tensor_name(this->layer_name()+".o_proj", idx_in_block_));

        auto data_type = q_proj_->data_type();
        auto device_type = q_proj_->device_type();

        rope_impl_ = new Qwen2RotaryEmbeddingLayer(data_type, device_type, *model_config_);
        mqa_ = new kernel::MultiQueryAttention(
            model_config_->num_attention_heads,
            model_config_->num_key_value_heads, idx_in_block_, kv_cache_,
            q_proj_, k_proj_, v_proj_, o_proj_,
            q_bias_, k_bias_, v_bias_, o_bias_);
    }

    Tensor Qwen2AttentionLayer::forward(const std::vector<const Tensor*>& input) {
        mqa_->set_rope_apply_func([this](const Tensor& q, const Tensor& k) -> std::tuple<Tensor, Tensor> {
            return this->rope_impl_->apply_rotary_pos_emb(q, k, seq_len_);
        });

        return mqa_->forward(input, nullptr);
    }

    std::size_t Qwen2AttentionLayer::get_seq_len() const {
        return seq_len_;
    }

    void Qwen2AttentionLayer::set_seq_len(std::size_t seq_len) {
        seq_len_ = seq_len;
    }

    Qwen2FFNLayer::Qwen2FFNLayer(int idx, const ModelConfig &model_config)
    : BaseLayer(model_config, "mlp"), idx_in_block_(idx) {}

    void Qwen2FFNLayer::init_weights(const StateDict &state_dict) {
        // gate
        auto key_gate_weight = build_weight_tensor_name(this->layer_name()+".gate_proj", idx_in_block_);
        auto key_gate_bias = build_bias_tensor_name(this->layer_name()+".gate_proj", idx_in_block_);
        mlp_gate_weight_ = get_weight_from_state_dict(state_dict, key_gate_weight, true);
        mlp_gate_bias_ = get_weight_from_state_dict(state_dict, key_gate_bias);
        // up
        auto key_up_weight = build_weight_tensor_name(this->layer_name()+".up_proj", idx_in_block_);
        auto key_up_bias = build_bias_tensor_name(this->layer_name()+".up_proj", idx_in_block_);
        mlp_up_weight_ = get_weight_from_state_dict(state_dict, key_up_weight, true);
        mlp_up_bias_ = get_weight_from_state_dict(state_dict, key_up_bias);
        // down
        auto key_down_weight = build_weight_tensor_name(this->layer_name()+".down_proj", idx_in_block_);
        auto key_down_bias = build_bias_tensor_name(this->layer_name()+".down_proj", idx_in_block_);
        mlp_down_weight_ = get_weight_from_state_dict(state_dict, key_down_weight, true);
        mlp_down_bias_ = get_weight_from_state_dict(state_dict, key_down_bias);
    }

    Tensor Qwen2FFNLayer::forward(const std::vector<const Tensor*>& input) {
        const auto* attn = input.at(0);
        // 1. 门控MLP层
        auto mlp_gate = attn->matmul(*this->mlp_gate_weight_);
        if (this->mlp_gate_bias_ != nullptr) {
            mlp_gate += *this->mlp_gate_bias_;
        }
        // 2. 上投影MLP层
        auto mlp_up = attn->matmul(*this->mlp_up_weight_);
        if (this->mlp_up_bias_ != nullptr) {
            mlp_up += *this->mlp_up_bias_;
        }
        // 3. 激活函数
        kernel::SiLUActivationOperator silu_op;
        auto mlp_gate_act = silu_op.forward({&mlp_gate}, nullptr);
        // 4. gate 与 up 逐元素乘
        auto mlp_gate_up_mul = mlp_gate_act.mul(mlp_up);
        // 4. 下投影MLP层
        auto ffn_output = mlp_gate_up_mul.matmul(*this->mlp_down_weight_);
        if (this->mlp_down_bias_ != nullptr) {
            ffn_output += *this->mlp_down_bias_;
        }
        return ffn_output;
    }

    Qwen2RMSNormLayer::Qwen2RMSNormLayer(const ModelConfig& model_config, std::string name)
    : BaseLayer(model_config, std::move(name)), eps_(model_config.rms_norm_eps) {}

    Qwen2RMSNormLayer::Qwen2RMSNormLayer(int idx, const ModelConfig& model_config, std::string name)
    : BaseLayer(model_config, std::move(name)), eps_(model_config.rms_norm_eps), idx_in_block_(idx) {}

    void Qwen2RMSNormLayer::init_weights(const StateDict& state_dict) {
        std::string weight_name;
        if (idx_in_block_ == -1) {
            weight_name = build_weight_tensor_name(this->layer_name());
        } else {
            weight_name = build_weight_tensor_name(this->layer_name(), idx_in_block_);
        }
        weight_ = get_weight_from_state_dict(state_dict, weight_name, true);
    }

    Tensor Qwen2RMSNormLayer::forward(const std::vector<const Tensor*>& input) {
        kernel::RMSNormOperator op(this->eps_);
        Tensor normed = op.forward(input, nullptr);
        return normed.mul(*this->weight_);
    }

    Qwen2DecoderLayer::Qwen2DecoderLayer(int idx, BaseKVCache* cache, const ModelConfig& model_config)
    : BaseLayer(model_config, "decoder"), attention_layer_(idx, cache, model_config),
    ffn_layer_(idx, model_config), input_norm_layer_(idx, model_config, "input_layernorm"),
    post_attention_norm_layer_(idx, model_config, "post_attention_layernorm") {}

    void Qwen2DecoderLayer::init_weights(const StateDict& state_dict) {
        this->attention_layer_.init_weights(state_dict);
        this->ffn_layer_.init_weights(state_dict);
        this->input_norm_layer_.init_weights(state_dict);
        this->post_attention_norm_layer_.init_weights(state_dict);
    }

    Tensor Qwen2DecoderLayer::forward(const std::vector<const Tensor*>& input) {
        if (input.size() != 1) {
            throw std::runtime_error("Qwen2DecoderModule::forward: input size must be 1");
        }

        const auto* input_hidden_states = input.at(0);

        // 输入RMS标准化
        auto hidden_states = input_norm_layer_.forward({input_hidden_states});

        // 自注意力
        const Tensor* q = &hidden_states;
        const Tensor* k = &hidden_states;
        const Tensor* v = &hidden_states;

        // 1. 注意力层
        auto attn_residual = *input_hidden_states;    // 注意力层残差块
        auto attn = attention_layer_.forward({q, k, v});
        attn += attn_residual;  // 注意力层残差连接

        // 2. FFN层
        auto ffn_residual = attn; // FFN层残差块
        auto post_attention_norm = post_attention_norm_layer_.forward({&attn});   // 注意力输出RMS标准化
        auto ffn_output = ffn_layer_.forward({&post_attention_norm});
        ffn_output += ffn_residual; // FFN层残差连接

        // 3. 返回
        return ffn_output;
    }

    std::size_t Qwen2DecoderLayer::get_seq_len() const {
        return this->attention_layer_.get_seq_len();
    }

    void Qwen2DecoderLayer::set_seq_len(std::size_t seq_len) {
        this->attention_layer_.set_seq_len(seq_len);
    }

    Qwen2LMHeadLayer::Qwen2LMHeadLayer(const ModelConfig& model_config, const Qwen2EmbeddingLayer& embedding_layer)
    : BaseLayer(model_config, "lm_head"), is_from_embedding_(false),
    embedding_layer_(&embedding_layer), lm_head_weight_(nullptr), lm_head_bias_(nullptr) {}

    Qwen2LMHeadLayer::~Qwen2LMHeadLayer() {
        if (is_from_embedding_ && lm_head_weight_ != nullptr) {
            delete lm_head_weight_;
        }
    }

    void Qwen2LMHeadLayer::init_weights(const StateDict &state_dict) {
        lm_head_weight_ = get_weight_from_state_dict(state_dict, build_weight_tensor_name(this->layer_name()));
        // 如果state_dict中没有该层权重，则embedding_weight转置后为lm_head_weight
        if (!lm_head_weight_) {
            is_from_embedding_ = true;
            lm_head_weight_ = new Tensor(embedding_layer_->get_embedding_weight()->transpose());
            return;
        }
        // 如果有该层权重，则加载对应的bias
        lm_head_bias_ = get_weight_from_state_dict(state_dict, build_bias_tensor_name(this->layer_name()));
    }

    Tensor Qwen2LMHeadLayer::forward(const std::vector<const Tensor *> &input) {
        const auto* norm_output = input.at(0);
        auto lm_head_output = norm_output->matmul(*this->lm_head_weight_);
        if (lm_head_bias_ != nullptr) {
            lm_head_output += *lm_head_bias_;
        }
        return lm_head_output;
    }
}   // fg42