//
// Created by 19373 on 2025/9/4.
//
#include <stdexcept>
#include "operator/Factory.h"
#include "cpu/add.h"
#include "cuda/add.cuh"

namespace fg42::kernel {
    VecAddKernelFunc::VecAddKernelFunc(DeviceType device_type) : device_type(device_type) {}

    void VecAddKernelFunc::operator()(const Tensor& input1, const Tensor& input2, Tensor& output, void* stream) const {
        switch (device_type) {
        case DeviceType::CPU:
            add_kernel_cpu(input1, input2, output);
            return;
        case DeviceType::NvidiaGPU:
            add_kernel_cuda(input1, input2, output, stream);
            return;
        default:
            throw std::runtime_error("Unknown device type");
        }
    }
}
