//
// Created by B777B2056-2 on 2025/9/7.
//
#include <stdexcept>
#include "operator/EmbeddingOperator.h"
#include "operator/Factory.h"

namespace fg42::kernel {
    EmbeddingOperator::EmbeddingOperator(const Tensor& weight_tensor)
        : BaseOperator(), weight_tensor_(&weight_tensor) {}

    void EmbeddingOperator::check(const std::vector<const Tensor*>& input_tensors) {
        BaseOperator::check(input_tensors);

        if (input_tensors.size() != 1) {
            throw std::invalid_argument("Input tensor should be a one tensor");
        }
        // 每个元素必须是行向量，数据类型为int，且位于cpu上
        const auto* input_tensor = input_tensors[0];
        if (input_tensor->shape().size() < 2 || input_tensor->shape().size() > 3) {
            throw std::invalid_argument("Input tensor's element should be a 1-D Tensor");
        }
        if (input_tensor->data_type() != DataType::Int32) {
            throw std::invalid_argument("Input tensor should be int32");
        }
    }

    Tensor EmbeddingOperator::forward(const std::vector<const Tensor*>& input_tensors, void* stream) {
        this->check(input_tensors);

        const auto* input_tensor = input_tensors[0];
        return embedding_kernel_func(this->weight_tensor_, *input_tensor, stream);
    }
} // kernel
// fg42