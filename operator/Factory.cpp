//
// Created by B777B2056-2 on 2025/9/4.
//
#include <stdexcept>
#include "operator/Factory.h"

#include "cpu/kernel_impl.h"
#include "operator/cpu/kernel.h"

#define UNUSED(x) (void)(x)

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
        UNUSED(stream); \
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
    Tensor vec_add_kernel_func(const Tensor& input1, const Tensor& input2, void* stream) {
        DEVICE_TYPE_SWITCH(add, input1.device_type(), stream, input1, input2);
    }

    Tensor negate_kernel_func(const Tensor& input, void* stream) {
        DEVICE_TYPE_SWITCH(negate, input.device_type(), stream, input);
    }

    Tensor vec_outer_kernel_func(const Tensor& input1, const Tensor& input2, void* stream) {
        DEVICE_TYPE_SWITCH(vec_outer, input1.device_type(), stream, input1, input2);
    }

    Tensor mul_kernel_func(const Tensor& input1, const Tensor& input2, void* stream) {
        DEVICE_TYPE_SWITCH(mul, input1.device_type(), stream, input1, input2);
    }

    Tensor mul_with_constant_value_kernel_func(float value, const Tensor& input2, void* stream) {
        DEVICE_TYPE_SWITCH(mul_with_constant_value, input2.device_type(), stream, value, input2);
    }

    Tensor matrix_matmul_kernel_func(const Tensor& input1, const Tensor& input2, void* stream) {
        DEVICE_TYPE_SWITCH(matmul, input1.device_type(), stream, input1, input2);
    }

    Tensor transpose_kernel_func(const Tensor &input, std::size_t dim0, std::size_t dim1, void *stream) {
        DEVICE_TYPE_SWITCH(transpose, input.device_type(), stream, input, dim0, dim1);
    }

    Tensor embedding_kernel_func(const Tensor* weight_tensor, const Tensor& input_tensor, void* stream) {
        DEVICE_TYPE_SWITCH(embedding, weight_tensor->device_type(), stream, weight_tensor, input_tensor);
    }

    Tensor repeat_kv_kernel_func(const Tensor& x, std::size_t n_rep, void* stream) {
        DEVICE_TYPE_SWITCH(repeat_kv, x.device_type(), stream, x, n_rep);
    }

    Tensor softmax_kernel_func(const Tensor& input, std::optional<float> t, void* stream) {
        DEVICE_TYPE_SWITCH(softmax, input.device_type(), stream, input, t);
    }

    Tensor silu_kernel_func(const Tensor& input, void* stream) {
        DEVICE_TYPE_SWITCH(silu, input.device_type(), stream, input);
    }

    Tensor rms_norm_kernel_func(const Tensor& input, float eps, void* stream) {
        DEVICE_TYPE_SWITCH(rme_norm, input.device_type(), stream, input, eps);
    }

    Tensor vec_or_matrix_argmax_kernel_func(const Tensor &input, std::size_t n, void *stream) {
        DEVICE_TYPE_SWITCH(vec_or_matrix_argmax, input.device_type(), stream, input, n);
    }

    Tensor rotate_half_kernel_func(const Tensor &input, void *stream) {
        DEVICE_TYPE_SWITCH(rotate_half, input.device_type(), stream, input);
    }

    Tensor cos_kernel_func(const Tensor &input, void *stream) {
        DEVICE_TYPE_SWITCH(cos, input.device_type(), stream, input);
    }

    Tensor sin_kernel_func(const Tensor &input, void *stream) {
        DEVICE_TYPE_SWITCH(sin, input.device_type(), stream, input);
    }

    Tensor concat_by_col_wise_kernel_func(const Tensor& x1, const Tensor& x2, void* stream) {
        DEVICE_TYPE_SWITCH(concat_by_col_wise, x1.device_type(), stream, x1, x2);
    }

    Tensor concat_by_row_wise_kernel_func(const Tensor& x1, const Tensor& x2, void* stream) {
        DEVICE_TYPE_SWITCH(concat_by_row_wise, x1.device_type(), stream, x1, x2);
    }

    Tensor causal_mask_kernel_func(DataType data_type, DeviceType device_type,
        std::size_t l, std::size_t s, void* stream) {
        DEVICE_TYPE_SWITCH(causal_mask, device_type, stream, data_type, l, s);
    }

    Tensor multinomial_kernel_func(const Tensor &x, std::size_t num_samples,
        const std::function<std::size_t(std::size_t)> &row_end_pos, void *stream) {
        DEVICE_TYPE_SWITCH(multinomial, x.device_type(), stream, x, num_samples, row_end_pos);
    }
}
