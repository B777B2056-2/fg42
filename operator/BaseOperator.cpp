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

    BaseOperatorWithWeight::BaseOperatorWithWeight(const std::vector<const Tensor*>& weight_tensors, const std::string& name)
            : BaseOperator(DeviceType::Unknown, name), weight_tensors_(weight_tensors) {
        if (!weight_tensors.empty()) {
            this->device_type_ = weight_tensors[0]->device_type();
        }
    }
}
