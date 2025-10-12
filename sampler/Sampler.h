//
// Created by B777B2056-2 on 2025/9/18.
//

#ifndef FG42_BASESAMPLER_H
#define FG42_BASESAMPLER_H
#include <unordered_map>
#include <memory>
#include "tensor/Tensor.h"
#include "operator/ActivationOperator.h"

namespace fg42 {
    enum class SamplingMethod {
        GreedySampling = 0,
        TemperatureSampling,
    };

    struct SamplerConfig {
        SamplingMethod method;
        float temperature;
    };

    class BaseSampler {
    public:
        explicit BaseSampler(SamplerConfig config) : config_(config) {};
        BaseSampler(const BaseSampler&) = delete;
        BaseSampler& operator=(const BaseSampler&) = delete;
        BaseSampler(BaseSampler&&) = delete;
        BaseSampler& operator=(BaseSampler&&) = delete;
        virtual ~BaseSampler() = default;

        /*
         * 获取下一个token
         * @param last_logits 最后一个时间步的logits，维度为(batch_size, 1, vocab_size)
         * @param return 返回下一个token的索引，维度为(batch_size, 1)
         */
        virtual Tensor sampling(const Tensor& last_logits) = 0;

    protected:
        SamplerConfig config_;
    };

    /* 贪心采样 */
    class GreedySampler : public BaseSampler {
    public:
        using BaseSampler::BaseSampler;
        ~GreedySampler() override = default;

        Tensor sampling(const Tensor& last_logits) override;

    private:
        kernel::SoftmaxActivationOperator softmax_op_;
    };

    /* Temperature采样 */
    class TemperatureSampler : public BaseSampler {
    public:
        explicit TemperatureSampler(SamplerConfig config);
        ~TemperatureSampler() override = default;

        Tensor sampling(const Tensor& last_logits) override;

    private:
        kernel::SoftmaxActivationOperator softmax_op_;
    };

    /* 采样执行器 */
    Tensor apply_sampling(SamplerConfig config, const Tensor& last_logits);
} // fg42

#endif //FG42_BASESAMPLER_H