//
// Created by B777B2056-2 on 2025/9/8.
//
#include <stdexcept>
#include <cuda_bf16.h>
#include "operator/cuda/kernel.cuh"

#define VEC_ADD_KERNEL_CUDA_FUNC(data_type) \
__global__ void vec_add_kernel_##data_type(std::int32_t size, const void* in1, const void* in2, void* out) {   \
        auto tid = static_cast<std::int32_t>(threadIdx.x) + blockDim.x * blockIdx.x;    \
            if (tid >= size) {  \
                return; \
            }   \
        const auto* in1_ptr = static_cast<const data_type*>(in1);  \
        const auto* in2_ptr = static_cast<const data_type*>(in2);  \
        auto* output_ptr = static_cast<data_type*>(out);  \
        const data_type in_val1 = in1_ptr[tid];    \
        const data_type in_val2 = in2_ptr[tid];    \
        output_ptr[tid] = in_val1 + in_val2;    \
    }

VEC_ADD_KERNEL_CUDA_FUNC(int8_t)
VEC_ADD_KERNEL_CUDA_FUNC(uint8_t)
VEC_ADD_KERNEL_CUDA_FUNC(int32_t)
VEC_ADD_KERNEL_CUDA_FUNC(__nv_bfloat16)
VEC_ADD_KERNEL_CUDA_FUNC(float)

#define VEC_ADD_KERNEL_CUDA_IMPL(data_type, block_num, thread_num, stream, ...) \
    {   \
        if (stream) {   \
            vec_add_kernel_##data_type<<<block_num, thread_num, 0, static_cast<cudaStream_t>(stream)>>>(__VA_ARGS__);    \
        } else {    \
            vec_add_kernel_##data_type<<<block_num, thread_num>>>(__VA_ARGS__);   \
        }   \
    }

namespace fg42::kernel {
    Tensor add_kernel_cuda(const Tensor& input1, const Tensor& input2, void* stream) {
        auto size = static_cast<std::int32_t>(input1.bytes_size());
        auto data_type = input1.data_type();

        std::int32_t thread_num = 512;
        std::int32_t block_num = (size + thread_num - 1) / thread_num;

        Tensor output(input1.data_type(), input1.device_type(), input1.shape());

        switch (data_type) {
        case DataType::Int8:
            VEC_ADD_KERNEL_CUDA_IMPL(int8_t, block_num, thread_num, stream,
                size, input1.raw_ptr(), input2.raw_ptr(), output.raw_ptr());
            break;
        case DataType::UInt8:
            VEC_ADD_KERNEL_CUDA_IMPL(uint8_t, block_num, thread_num, stream,
                size, input1.raw_ptr(), input2.raw_ptr(), output.raw_ptr());
            break;
        case DataType::Int32:
            VEC_ADD_KERNEL_CUDA_IMPL(int32_t, block_num, thread_num, stream,
                size, input1.raw_ptr(), input2.raw_ptr(), output.raw_ptr());
        case DataType::BF16:
            VEC_ADD_KERNEL_CUDA_IMPL(__nv_bfloat16, block_num, thread_num, stream,
                size, input1.raw_ptr(), input2.raw_ptr(), output.raw_ptr());
            break;
        case DataType::FP32:
            VEC_ADD_KERNEL_CUDA_IMPL(float, block_num, thread_num, stream,
                size, input1.raw_ptr(), input2.raw_ptr(), output.raw_ptr());
            break;
        default:
            throw std::runtime_error("unsupported data type");
        }
        return output;
    }

    Tensor embedding_kernel_cuda(const Tensor* weight_tensor, const Tensor& input_tensors, void* stream) {
        throw std::runtime_error("not implemented");
    }
}