//
// Created by 19373 on 2025/9/4.
//
#include <stdexcept>
#include "operator/Factory.h"
#include "operator/cpu/add.h"
#include "operator/cuda/add.cuh"
#include "operator/cpu/embedding.h"

namespace fg42::kernel {
    VecAddKernelFunc::VecAddKernelFunc(DeviceType device_type) : device_type(device_type) {}

    Tensor VecAddKernelFunc::operator()(const Tensor& input1, const Tensor& input2, void* stream) const {
        switch (device_type) {
        case DeviceType::CPU:
            return add_kernel_cpu(input1, input2);
        case DeviceType::NvidiaGPU:
            return add_kernel_cuda(input1, input2, stream);
        default:
            throw std::runtime_error("Unknown device type");
        }
    }

    EmbeddingKernelFunc::EmbeddingKernelFunc(DeviceType device_type) : device_type(device_type) {}

    Tensor EmbeddingKernelFunc::operator()(const Tensor* weight_tensor,
            const Tensor& input_tensor, void* stream) const {
        switch (device_type) {
        case DeviceType::CPU:
            return embedding_kernel_cpu(weight_tensor, input_tensor);
        case DeviceType::NvidiaGPU:
            // return embedding_kernel_cuda(weight_tensor, input_tensor, stream);
        default:
            throw std::runtime_error("Unknown device type");
        }
    }
}
