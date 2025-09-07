//
// Created by 19373 on 2025/9/7.
//
#include <stdexcept>
#include "operator/EmbeddingOperator.h"
#include "operator/Factory.h"

namespace fg42 {
    namespace kernel {
        EmbeddingOperator::EmbeddingOperator(const Tensor& weight_tensor,
                const std::string& name , std::optional<EmbeddingOperatorOptions> options)
            : BaseOperatorWithWeight(weight_tensor, name), options_(options) {
            if (!this->EmbeddingOperator::check_weights(weight_tensor)) {
                throw std::runtime_error("Wrong weight tensor");
            }
        }

        bool EmbeddingOperator::check(const std::vector<const Tensor*>& input_tensors) const {
            if (input_tensors.size() != 1) {
                return false;
            }
            // 每个元素必须是行向量，数据类型为int，且位于cpu上
            const auto* input_tensor = input_tensors[0];
            if (input_tensor->shape().size() < 2 || input_tensor->shape().size() > 3) {
                return false;
            }
            if (input_tensor->data_type() != DataType::Int32) {
                return false;
            }
            if (input_tensor->device_type() != DeviceType::CPU) {
                return false;
            }
            return true;
        }

        bool EmbeddingOperator::check_weights(const Tensor& weight_tensor) const {
            return true;
        }

        Tensor EmbeddingOperator::forward(const std::vector<const Tensor*>& input_tensors, void* stream) {
            if (!this->check(input_tensors)) {
                throw std::runtime_error("EmbeddingOperator: check failed");
            }

            const auto* input_tensor = input_tensors[0];
            auto f = EmbeddingKernelFunc(input_tensor->device_type());
            return f(this->weight_tensor_, *input_tensor, stream);
        }
    } // kernel
} // fg42