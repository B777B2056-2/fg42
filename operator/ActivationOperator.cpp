//
// Created by B777B2056-2 on 2025/9/9.
//
#include "operator/Factory.h"
#include "operator/ActivationOperator.h"

#include "ArithmeticOperator.h"

namespace fg42::kernel {
    SoftmaxActivationOperator::SoftmaxActivationOperator(std::optional<float> t) : BaseOperator(), t_(t) {}

    Tensor SoftmaxActivationOperator::forward(const std::vector<const Tensor*>& input_tensors, void* stream) {
        this->check(input_tensors);

        const auto* input = input_tensors[0];
        return softmax_kernel_func(*input, t_, stream);
    }

    void SoftmaxActivationOperator::check(const std::vector<const Tensor*>& input_tensors) {
        BaseOperator::check(input_tensors);

        if (input_tensors.size() != 1) {
            throw std::runtime_error("SoftmaxActivationOperator only supports one input tensor.");
        }

        if (input_tensors[0]->empty()) {
            throw std::runtime_error("SoftmaxActivationOperator input tensor is empty.");
        }
    }

    SiLUActivationOperator::SiLUActivationOperator() : BaseOperator() {}

    Tensor SiLUActivationOperator::forward(const std::vector<const Tensor *> &input_tensors, void *stream) {
        this->check(input_tensors);

        const auto* input = input_tensors[0];
        return silu_kernel_func(*input, stream);
    }

    void SiLUActivationOperator::check(const std::vector<const Tensor *> &input_tensors) {
        BaseOperator::check(input_tensors);

        if (input_tensors.size() != 1) {
            throw std::runtime_error("SiLUTActivationOperator only supports one input tensor.");
        }

        if (input_tensors[0]->empty()) {
            throw std::runtime_error("SiLUTActivationOperator input tensor is empty.");
        }
    }
} // kernel
// fg42