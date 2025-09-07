//
// Created by 19373 on 2025/9/4.
//
#include <string>
#include "operator/BaseOperator.h"

namespace fg42::kernel {
    BaseOperator::BaseOperator(DeviceType device_type, std::string name)
            : device_type_(device_type), name_(std::move(name)) {}

    DeviceType BaseOperator::device_type() const { return this->device_type_; }
    const std::string& BaseOperator::name() const { return this->name_; }

    BaseOperatorWithWeight::BaseOperatorWithWeight(const Tensor& weight_tensor, const std::string& name)
            : BaseOperator(DeviceType::Unknown, name), weight_tensor_(&weight_tensor) {
        if (!weight_tensor.empty()) {
            this->device_type_ = weight_tensor.device_type();
        }
    }
}
