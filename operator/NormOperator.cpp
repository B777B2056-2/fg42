//
// Created by B777B2056-2 on 2025/9/10.
//
#include <Eigen/Core>
#include "operator/Factory.h"
#include "operator/NormOperator.h"

namespace fg42::kernel {
    RMSNormOperator::RMSNormOperator(float eps)
        : BaseOperator(), eps_(eps) {}

    Tensor RMSNormOperator::forward(const std::vector<const Tensor*>& input_tensors, void* stream) {
        this->check(input_tensors);

        const Tensor& input = *input_tensors[0];
        auto rme = rms_norm_kernel_func(input, this->eps_, stream);
        return rme;
    }

    void RMSNormOperator::check(const std::vector<const Tensor*>& input_tensors) {
        BaseOperator::check(input_tensors);

        if (input_tensors.size() != 1) {
            throw std::runtime_error("RMSNormOperator::check: input tensors size must be 1.");
        }
        if (input_tensors[0]->empty()) {
            throw std::runtime_error("RMSNormOperator::check: input tensors must not be empty.");
        }
    }
} // kernel
// fg42