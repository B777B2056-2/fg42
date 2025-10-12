//
// Created by B777B2056-2 on 2025/9/4.
//
#include <string>
#include <unordered_map>
#include "operator/BaseOperator.h"

#include <unordered_set>

namespace fg42::kernel {
    bool BaseOperator::is_same_device_and_data_type(
        const std::vector<const Tensor*>& input_tensors) const {
        std::unordered_set<DeviceType> device_type_set;
        std::unordered_set<DataType> data_type_set;
        for (const auto* tensor : input_tensors) {
            if (tensor == nullptr) {
                continue;
            }
            device_type_set.insert(tensor->device_type());
            data_type_set.insert(tensor->data_type());
        }
        return device_type_set.size() == 1 && data_type_set.size() == 1;
    }

    bool BaseOperator::is_all_same_shape(const std::vector<const Tensor*>& input_tensors) const {
        for (std::size_t i = 0; i < input_tensors.size() - 1; ++i) {
            if (!Tensor::shape_equal(input_tensors[i]->shape(), input_tensors[i + 1]->shape())) {
                return false;
            }
        }
        return true;
    }

    void BaseOperator::check(const std::vector<const Tensor*>& input_tensors) {
        if (input_tensors.empty()) {
            throw std::invalid_argument("input tensors is empty");
        }

        if (!this->is_same_device_and_data_type(input_tensors)) {
            throw std::invalid_argument("input tensors have different device type or data type");
        }

        for (const auto* tensor : input_tensors) {
            if (tensor == nullptr) {
                throw std::invalid_argument("input tensor is nullptr");
            }
            if (tensor->empty()) {
                throw std::invalid_argument("input tensor is empty");
            }
        }
    }
}
