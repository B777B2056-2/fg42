//
// Created by B777B2056-2 on 2025/9/3.
//

#ifndef FG42_FACTORY_H
#define FG42_FACTORY_H
#include "tensor/Tensor.h"
#include <optional>

namespace fg42::kernel {
    Tensor vec_add_kernel_func(const Tensor& input1, const Tensor& input2, void* stream);
    Tensor negate_kernel_func(const Tensor& input, void* stream);
    Tensor vec_outer_kernel_func(const Tensor& input1, const Tensor& input2, void* stream);
    Tensor mul_kernel_func(const Tensor& input1, const Tensor& input2, void* stream);
    Tensor mul_with_constant_value_kernel_func(float value, const Tensor& input2, void* stream);
    Tensor matrix_matmul_kernel_func(const Tensor& input1, const Tensor& input2, void* stream);
    Tensor transpose_kernel_func(const Tensor& input1, std::size_t dim0, std::size_t dim1, void* stream);
    Tensor embedding_kernel_func(const Tensor* weight_tensor, const Tensor& input_tensor, void* stream);
    Tensor repeat_kv_kernel_func(const Tensor& x, std::size_t n_rep, void* stream);
    Tensor softmax_kernel_func(const Tensor& input, std::optional<float> t, void* stream);
    Tensor silu_kernel_func(const Tensor& input, void* stream);
    Tensor rms_norm_kernel_func(const Tensor& input, float eps, void* stream);
    Tensor vec_or_matrix_argmax_kernel_func(const Tensor& input, std::size_t n, void* stream);
    Tensor rotate_half_kernel_func(const Tensor& input, void *stream);
    Tensor cos_kernel_func(const Tensor& input, void* stream);
    Tensor sin_kernel_func(const Tensor& input, void* stream);
    Tensor concat_by_col_wise_kernel_func(const Tensor& x1, const Tensor& x2, void* stream);
    Tensor concat_by_row_wise_kernel_func(const Tensor& x1, const Tensor& x2, void* stream);
    Tensor causal_mask_kernel_func(DataType data_type, DeviceType device_type,
        std::size_t l, std::size_t s, void* stream);
    Tensor multinomial_kernel_func(const Tensor& x, std::size_t num_samples,
        const std::function<std::size_t(std::size_t)>& row_end_pos, void* stream);
}

#endif //FG42_FACTORY_H
