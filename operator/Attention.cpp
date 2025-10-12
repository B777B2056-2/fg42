//
// Created by B777B2056-2 on 2025/9/13.
//
#include "Eigen/Core"
#include "operator/Attention.h"

#include <fstream>
#include <iostream>

#include "Factory.h"
#include "operator/ArithmeticOperator.h"
#include "operator/ActivationOperator.h"
#include "memory/BaseAllocator.h"
#include "memory/Common.h"

namespace fg42::kernel {
    BaseAttention::BaseAttention(const Tensor *wq, const Tensor *wk, const Tensor *wv,
                                 const Tensor *q_bias, const Tensor *k_bias, const Tensor *v_bias)
        : BaseOperator(),
        wq_(wq), wk_(wk), wv_(wv), q_bias_(q_bias), k_bias_(k_bias), v_bias_(v_bias) {
    }

    void BaseAttention::check_attn_weights() const {
        if (this->wq_->empty() || this->wv_->empty() || this->wv_->empty()) {
            throw std::runtime_error("Attention weights must not be empty.");
        }
        if (!is_same_device_and_data_type({
            this->wq_, this->wv_, this->wk_, this->q_bias_, this->q_bias_, this->v_bias_})) {
            throw std::runtime_error("Attention weights must be same data type.");
        }
        if (!is_all_same_shape({this->wq_, this->wv_, this->wk_})) {
            throw std::runtime_error("Attention weights must be same shape or all.");
        }
        if (this->wq_->shape().size() != 2) {
            throw std::runtime_error("Attention weights must have 2 dimensions.");
        }
    }

    ScaledDotProductAttention::ScaledDotProductAttention() : BaseOperator(), num_kv_groups_(1) {}

    ScaledDotProductAttention::ScaledDotProductAttention(std::size_t num_heads, std::size_t num_kv_heads)
    : BaseOperator(), num_kv_groups_(1) {
        num_kv_groups_ = num_heads / num_kv_heads; // 必须可被整除
    }

    void ScaledDotProductAttention::check(const std::vector<const Tensor*>& input_tensors) {
        BaseOperator::check(input_tensors);

        if (input_tensors.size() != 3) {
            throw std::runtime_error("ScaledDotProductAttention requires 3 input tensors: q, k, v.");
        }
        if (!is_all_same_shape(input_tensors)) {
            throw std::runtime_error("ScaledDotProductAttention requires all input tensors to have the same shape.");
        }
        if (input_tensors[0]->shape().size() < 2) {
            throw std::runtime_error("ScaledDotProductAttention requires input tensors to have at least 2 dimensions.");
        }
    }

    Tensor ScaledDotProductAttention::create_causal_mask(const Tensor& q, const Tensor& k) const {
        std::size_t l = q.shape().at(q.shape().size() - 2);
        std::size_t s = k.shape().at(k.shape().size() - 2);

        kernel::CausalMaskOperator op(q.data_type(), q.device_type(), l, s);
        return op.forward({}, nullptr);
    }

    // 缩放点积注意力计算：scores = softmax(q * k^T / sqrt(d_k)) * v
    Tensor ScaledDotProductAttention::forward(const std::vector<const Tensor*>& input_tensors, void* stream) {
        Tensor q = *input_tensors[0];
        Tensor k = *input_tensors[1];
        Tensor v = *input_tensors[2];

        if (num_kv_groups_ > 1) {
            k = repeat_kv_kernel_func(k, num_kv_groups_, stream);
            v = repeat_kv_kernel_func(v, num_kv_groups_, stream);
        }

        // a = q * k^T
        Tensor attn_weights = q.matmul(k.transpose());

        // a = a / sqrt(d_k)
        auto scale_factor = static_cast<float>(1.0 / std::sqrt(q.shape().back()));
        attn_weights = attn_weights.mul(scale_factor);

        // 加上因果掩码（仅对不使用cache或prefill阶段生效）
        bool need_causal_mask = (q.shape().at(q.shape().size() - 2) > 1);
        if (need_causal_mask) {
            // 创建因果掩码
            Tensor causal_mask = create_causal_mask(q, k);
            attn_weights += causal_mask;
        }

        // a = softmax(a)
        attn_weights =  SoftmaxActivationOperator().forward({&attn_weights}, stream);
        // scores = a * v
        Tensor scores = attn_weights.matmul(v);
        return scores;
    }

    MultiQueryAttention::MultiQueryAttention(
        std::size_t num_heads, std::size_t num_kv_heads, std::size_t layer_idx, BaseKVCache* cache,
        const Tensor *wq, const Tensor *wk, const Tensor *wv, const Tensor *wo,
        const Tensor *q_bias, const Tensor *k_bias,
        const Tensor *v_bias, const Tensor *o_bias)
        : BaseAttention(wq, wk, wv, q_bias, k_bias, v_bias),
        num_heads_(num_heads), head_dim_(0), num_kv_groups_(0), num_kv_heads_(num_kv_heads),
        wo_(wo), o_bias_(o_bias), spda_(num_heads, num_kv_heads),
        layer_idx_(layer_idx), kv_cache_instance_(cache) {
        std::size_t hidden_size = wq->shape().front();
        if (hidden_size % num_heads_ != 0) {
            throw std::runtime_error("hidden_size must be divisible by num_heads.");
        }
        if (num_heads_ % num_kv_heads_ != 0) {
            throw std::runtime_error("num_heads must be divisible by num_kv_heads.");
        }
        head_dim_ = hidden_size / num_heads_;
        this->MultiQueryAttention::check_attn_weights();
    }

    void MultiQueryAttention::set_rope_apply_func(RopeHandler f) {
        rope_apply_func_ = std::move(f);
    }

    void MultiQueryAttention::split_multi_heads(Tensor& x) const {
        auto shape = x.shape();
        const std::size_t d_model = shape.at(2);
        const std::size_t d_k = d_model / num_heads_;

        x.reshape({shape.at(0), shape.at(1), this->num_heads_, d_k});
        x = x.transpose(1, 2);
    }

    void MultiQueryAttention::concat_multi_heads(Tensor& x) const {
        x = x.transpose(1, 2);
        x.reshape({x.shape().at(0), x.shape().at(1), x.shape().at(2) * x.shape().at(3)});
    }

    void MultiQueryAttention::split_kv_multi_heads(Tensor& x) const {
        auto shape = x.shape();
        x.reshape({shape.at(0), shape.at(1), this->num_kv_heads_, this->head_dim_});
        x = x.transpose(1, 2);
    }

    Tensor MultiQueryAttention::forward(const std::vector<const Tensor*>& input_tensors, void* stream) {
        this->check(input_tensors);

        const Tensor* input_q = input_tensors[0];
        const Tensor* input_k = input_tensors[1];
        const Tensor* input_v = input_tensors[2];

        // 构造q k v（如果需要偏置，则加上偏置）
        auto q = input_q->matmul(*this->wq_);
        if (this->q_bias_ != nullptr) {
            q += *(this->q_bias_);
        }
        auto k = input_k->matmul(*this->wk_);
        if (this->k_bias_ != nullptr) {
            k += *(this->k_bias_);
        }
        auto v = input_v->matmul(*this->wv_);
        if (this->v_bias_ != nullptr) {
            v += *(this->v_bias_);
        }

        // Q直接拆分多头
        this->split_multi_heads(q);

        // K V 按head dim拆分多头
        this->split_kv_multi_heads(k);
        this->split_kv_multi_heads(v);

        // 应用旋转编码
        if (this->rope_apply_func_) {
            auto [q_rope, k_rope] = this->rope_apply_func_(q, k);
            q = q_rope;
            k = k_rope;
        }

        // 更新kv cache & 计算注意力
        Tensor attn_weights;
        if (this->kv_cache_instance_ != nullptr) {
            attn_weights = this->kv_cache_instance_->apply_attention(layer_idx_, q, k, v, stream);
        } else {
            attn_weights = spda_.forward({&q, &k, &v}, stream);
        }

        // 合并多头
        this->concat_multi_heads(attn_weights);

        // 投影输出
        auto output = attn_weights.matmul(*this->wo_);
        if (this->o_bias_ != nullptr) {
            output += *(this->o_bias_);
        }
        return output;
    }

    void MultiQueryAttention::check_attn_weights() const {
        if (this->wq_->empty() || this->wv_->empty() || this->wv_->empty()) {
            throw std::runtime_error("Attention weights must not be empty.");
        }

        if (!is_same_device_and_data_type({
            this->wq_, this->wv_, this->wk_, this->wv_, this->wo_,
            this->q_bias_, this->q_bias_, this->v_bias_, this->o_bias_,
        })) {
            throw std::runtime_error("Attention weights must be same data type.");
        }

        // 多头数量必须能被embedding_dim整除
        const std::size_t d_model = this->wq_->shape().front();
        if (d_model % num_heads_ != 0) {
            throw std::runtime_error("d_model must be divisible by num_heads.");
        }

        if (this->num_kv_heads_ == this->num_heads_) {  // MHA
            // 1. Q K V 投影矩阵形状相同
            if (!is_all_same_shape({this->wq_, this->wv_, this->wk_})) {
                throw std::runtime_error("Attention weights must be same shape or all.");
            }
            // 2. 投影矩阵形状为2维
            if (this->wq_->shape().size() != 2) {
                throw std::runtime_error("Attention weights must have 2 dimensions.");
            }
        } else if (this->num_kv_heads_ < this->num_heads_) {    // MQA
            // 1. MQA必须要下投影矩阵
            if (this->wo_ == nullptr || this->wo_->empty()) {
                throw std::runtime_error("Multi-query attention must have o_proj.");
            }
            // 2. num_heads_ 必须能被 num_kv_heads_ 整除
            if (this->num_heads_ % this->num_kv_heads_ != 0) {
                throw std::runtime_error("num_heads must be divisible by num_kv_heads.");
            }
            // 3. Q投影矩阵维度为d_model x d_model
            if (!Tensor::shape_equal(this->wq_->shape(), {d_model, d_model})) {
                throw std::runtime_error("Q projection matrix must have shape (d_model, d_model).");
            }
            // 4. K V投影矩阵维度为d_model x (num_kv_heads_ * head_dim)
            if (!Tensor::shape_equal(this->wk_->shape(), {d_model, this->num_kv_heads_ * this->head_dim_}) ||
                !Tensor::shape_equal(this->wv_->shape(), {d_model, this->num_kv_heads_ * this->head_dim_})) {
                throw std::runtime_error("K V projection matrix must have shape (d_model, head_dim).");
            }
        } else {
            // num_kv_heads 必须小于 num_heads
            throw std::runtime_error("num_kv_heads must be less than or equal to num_heads.");
        }
    }

    void MultiQueryAttention::check(const std::vector<const Tensor *> &input_tensors) {
        BaseAttention::check(input_tensors);

        if (input_tensors.size() != 3) {
            throw std::runtime_error("Attention requires at 3 input tensors: q, k, v.");
        }

        // 多头数量必须能被embedding_dim整除
        const std::size_t d_model = input_tensors[0]->shape().at(2);
        if (d_model % this->num_heads_ != 0) {
            throw std::runtime_error("d_model must be divisible by num_heads.");
        }

        if (this->num_kv_heads_ == this->num_heads_) {  // MHA
            if (!is_all_same_shape(input_tensors)) {
                throw std::runtime_error("Multi-head attention requires all input tensors to have the same shape.");
            }
        } else if (this->num_kv_heads_ < this->num_heads_) {    // MQA

        } else {
            throw std::runtime_error("num_kv_heads must be less than or equal to num_heads.");
        }
    }
} // kernel
// fg42