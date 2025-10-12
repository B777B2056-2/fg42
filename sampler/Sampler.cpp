//
// Created by B777B2056-2 on 2025/9/19.
//
#include "sampler/Sampler.h"
#include <random>
#include <functional>
#include "operator/ArithmeticOperator.h"

namespace fg42 {
    Tensor GreedySampler::sampling(const Tensor &last_logits) {
        auto prbos = softmax_op_.forward({&last_logits}, nullptr);

        kernel::VecOrMatrixArgmaxOperator argmax_op(1);
        return argmax_op.forward({&prbos}, nullptr);
    }

    TemperatureSampler::TemperatureSampler(SamplerConfig config)
    : BaseSampler(config), softmax_op_(config.temperature) {
        if (std::isless(config.temperature, 0.0f) || std::isgreater(config.temperature, 2.0f)) {
            throw std::invalid_argument("Temperature paramis not a valid");
        }
    }

    Tensor TemperatureSampler::sampling(const Tensor& last_logits) {
        auto prbos = softmax_op_.forward({&last_logits}, nullptr);

        kernel::MultinomialOperator multinomial_op(1,
            [&prbos](std::size_t) -> std::size_t { return prbos.shape().at(1); });
        return multinomial_op.forward({&prbos}, nullptr);
    }

    Tensor apply_sampling(SamplerConfig config, const Tensor& last_logits) {
        switch (config.method) {
            case SamplingMethod::GreedySampling:
                return GreedySampler(config).sampling(last_logits);
            case SamplingMethod::TemperatureSampling:
                return TemperatureSampler(config).sampling(last_logits);
            default:
                throw std::runtime_error("Sampler not found");
        }
    }
} // fg42