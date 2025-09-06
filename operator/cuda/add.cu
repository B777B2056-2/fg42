//
// Created by 19373 on 2025/9/3.
//
#include <stdexcept>
#include <cuda_bf16.h>
#include "operator/cuda/add.cuh"

__global__ void vec_add_kernel_int8(std::int32_t size, const void* in1, const void* in2, void* out) {
    auto tid = static_cast<std::int32_t>(threadIdx.x) + blockDim.x * blockIdx.x;
    if (tid >= size) {
        return;
    }

    const auto* input1 = static_cast<const std::int8_t*>(in1);
    const auto* input2 = static_cast<const std::int8_t*>(in2);
    auto* output = static_cast<std::int8_t*>(out);

    const std::int8_t in_val1 = input1[tid];
    const std::int8_t in_val2 = input2[tid];
    output[tid] = in_val1 + in_val2;
}

__global__ void vec_add_kernel_bf16(std::int32_t size, const void* in1, const void* in2, void* out) {
    auto tid = static_cast<std::int32_t>(threadIdx.x) + blockDim.x * blockIdx.x;
    if (tid >= size) {
        return;
    }

    const auto* input1 = static_cast<const __nv_bfloat16*>(in1);
    const auto* input2 = static_cast<const __nv_bfloat16*>(in2);
    auto* output = static_cast<__nv_bfloat16*>(out);

    const __nv_bfloat16 in_val1 = input1[tid];
    const __nv_bfloat16 in_val2 = input2[tid];
    output[tid] = __hadd(in_val1, in_val2);
}

__global__ void vec_add_kernel_fp32(std::int32_t size, const void* in1, const void* in2, void* out) {
    auto tid = static_cast<std::int32_t>(threadIdx.x) + blockDim.x * blockIdx.x;
    if (tid >= size) {
        return;
    }

    const auto* input1 = static_cast<const float*>(in1);
    const auto* input2 = static_cast<const float*>(in2);
    auto* output = static_cast<float*>(out);

    const float in_val1 = input1[tid];
    const float in_val2 = input2[tid];
    output[tid] = in_val1 + in_val2;
}

namespace fg42::kernel {
    void add_kernel_cuda(const Tensor& input1, const Tensor& input2, Tensor& output, void* stream) {
        auto size = static_cast<std::int32_t>(input1.bytes_size());
        auto data_type = input1.data_type();

        std::int32_t thread_num = 512;
        std::int32_t block_num = (size + thread_num - 1) / thread_num;

        switch (data_type) {
        case DataType::Int8:
            if (stream) {
                vec_add_kernel_int8<<<block_num, thread_num, 0, static_cast<cudaStream_t>(stream)>>>(
                    size, input1.raw_ptr(), input2.raw_ptr(), output.raw_ptr());
            } else {
                vec_add_kernel_int8<<<block_num, thread_num>>>(
                    size, input1.raw_ptr(), input2.raw_ptr(), output.raw_ptr());
            }
            break;

        case DataType::BF16:
            if (stream) {
                vec_add_kernel_bf16<<<block_num, thread_num, 0, static_cast<cudaStream_t>(stream)>>>(
                    size, input1.raw_ptr(), input2.raw_ptr(), output.raw_ptr());
            } else {
                vec_add_kernel_bf16<<<block_num, thread_num>>>(
                    size, input1.raw_ptr(), input2.raw_ptr(), output.raw_ptr());
            }
            break;
        case DataType::FP32:
            if (stream) {
                vec_add_kernel_fp32<<<block_num, thread_num, 0, static_cast<cudaStream_t>(stream)>>>(
                    size, input1.raw_ptr(), input2.raw_ptr(), output.raw_ptr());
            } else {
                vec_add_kernel_fp32<<<block_num, thread_num>>>(
                    size, input1.raw_ptr(), input2.raw_ptr(), output.raw_ptr());
            }
            break;
        default:
            throw std::runtime_error("unsupported data type");
        }
    }
}