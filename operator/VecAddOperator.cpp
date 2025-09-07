//
// Created by 19373 on 2025/9/4.
//
#include <stdexcept>
#include "operator/Factory.h"
#include "operator/VecAddOperator.h"

namespace fg42::kernel {
    VecAddOperator::VecAddOperator(DeviceType device_type, std::string name)
            : BaseOperator(device_type, std::move(name)) {}

    Tensor VecAddOperator::forward(const std::vector<const Tensor*>& input_tensors, void* stream) {
        if (!this->check(input_tensors)) {
            throw std::runtime_error("VecAddOperator: check failed");
        }

        auto f = VecAddKernelFunc(input_tensors[0]->device_type());
        return f(*input_tensors[0], *input_tensors[1], stream);
    }

    bool VecAddOperator::check(const std::vector<const Tensor*>& input_tensors) const {
        if (input_tensors.size() != 2) {
            return false;
        }

        if (input_tensors[0]->empty() || input_tensors[1]->empty()) {
            return false;
        }

        if (this->device_type() != input_tensors[0]->device_type() ||
            this->device_type() != input_tensors[1]->device_type()) {
            return false;
       }

        if (!Tensor::shape_equal(input_tensors[0]->shape(), input_tensors[1]->shape())) {
            return false;
        }
        return true;
    }
}
