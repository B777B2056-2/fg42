//
// Created by B777B2056-2 on 2025/9/13.
//

#ifndef FG42_ATTENTION_H
#define FG42_ATTENTION_H
#include "operator/BaseOperator.h"
#include "cache/KVCache.h"

namespace fg42::kernel {
    class BaseAttention : public BaseOperator {
    public:
        BaseAttention(
            const Tensor* wq, const Tensor* wk, const Tensor* wv,
            const Tensor* q_bias, const Tensor* k_bias, const Tensor* v_bias);

    protected:
        const Tensor* wq_;
        const Tensor* wk_;
        const Tensor* wv_;

        const Tensor* q_bias_;
        const Tensor* k_bias_;
        const Tensor* v_bias_;

        virtual void check_attn_weights() const;
    };

    class ScaledDotProductAttention : public BaseOperator {
    public:
        ScaledDotProductAttention();
        ScaledDotProductAttention(std::size_t num_heads, std::size_t num_kv_heads);
        ~ScaledDotProductAttention() override = default;

        Tensor forward(const std::vector<const Tensor*>& input_tensors, void* stream) override;

    private:
        std::size_t num_kv_groups_;

        void check(const std::vector<const Tensor*>& input_tensors) override;
        [[nodiscard]] Tensor create_causal_mask(const Tensor& q, const Tensor& k) const;
    };

    class MultiQueryAttention : public BaseAttention {
    public:
        using RopeHandler = std::function<std::tuple<Tensor, Tensor>(const Tensor&, const Tensor&)>;

        MultiQueryAttention(std::size_t num_heads, std::size_t num_kv_heads, std::size_t layer_idx,
            BaseKVCache* cache, const Tensor* wq, const Tensor* wk, const Tensor* wv, const Tensor* wo,
            const Tensor* q_bias, const Tensor* k_bias, const Tensor* v_bias, const Tensor* o_bias);

        void set_rope_apply_func(RopeHandler f);

        Tensor forward(const std::vector<const Tensor*>& input_tensors, void* stream) override;

    private:
        std::size_t num_heads_;
        std::size_t head_dim_;
        std::size_t num_kv_groups_;
        std::size_t num_kv_heads_;

        const Tensor* wo_;
        const Tensor* o_bias_;

        ScaledDotProductAttention spda_;
        RopeHandler rope_apply_func_;

        std::size_t layer_idx_;
        BaseKVCache* kv_cache_instance_;

        void check_attn_weights() const override;
        void check(const std::vector<const Tensor*>& input_tensors) override;
        void split_multi_heads(Tensor& x) const;
        void concat_multi_heads(Tensor& x) const;
        void split_kv_multi_heads(Tensor& x) const;
    };
} // kernel
// fg42

#endif //FG42_ATTENTION_H