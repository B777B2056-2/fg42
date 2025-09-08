//
// Created by 19373 on 2025/9/4.
//
#include <stdexcept>
#include "operator/Factory.h"
#include "operator/cpu/kernel.h"

#ifdef HAVE_CUDA
#include "operator/cuda/kernel.cuh"

#define DEVICE_TYPE_SWITCH(op_name, device_type, stream, ...) \
    {   \
        switch (device_type) { \
        case DeviceType::CPU:   \
            return op_name##_kernel_cpu(__VA_ARGS__);  \
        case DeviceType::NvidiaGPU: \
            return op_name##_kernel_cuda(__VA_ARGS__, stream); \
        default:    \
            throw std::runtime_error("Unknown device type");    \
        }   \
    }
#else
#define DEVICE_TYPE_SWITCH(op_name, device_type, stream, ...) \
    {   \
        switch (device_type) { \
        case DeviceType::CPU:   \
            return op_name##_kernel_cpu(__VA_ARGS__);  \
        case DeviceType::NvidiaGPU: \
        default:    \
            throw std::runtime_error("Unknown device type");    \
        }   \
    }
#endif

namespace fg42::kernel {
    Tensor VecAddKernelFunc(const Tensor& input1, const Tensor& input2, void* stream) {
        DEVICE_TYPE_SWITCH(add, input1.device_type(), stream, input1, input2);
    }

    Tensor EmbeddingKernelFunc(const Tensor* weight_tensor, const Tensor& input_tensor, void* stream) {
        DEVICE_TYPE_SWITCH(embedding, weight_tensor->device_type(), stream, weight_tensor, input_tensor);
    }
}
