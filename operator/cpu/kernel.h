//
// Created by B777B2056-2 on 2025/9/8.
//

#ifndef FG42_KERNEL_H
#define FG42_KERNEL_H
#include <optional>
#include "tensor/Tensor.h"

namespace fg42::kernel {
    Tensor add_kernel_cpu(const Tensor& input1, const Tensor& input2);
    Tensor negate_kernel_cpu(const Tensor& input);
    Tensor vec_outer_kernel_cpu(const Tensor& input1, const Tensor& input2);
    Tensor mul_kernel_cpu(const Tensor& input1, const Tensor& input2);
    Tensor mul_with_constant_value_kernel_cpu(float value, const Tensor& input2);
    Tensor matmul_kernel_cpu(const Tensor& input1, const Tensor& input2);
    Tensor transpose_kernel_cpu(const Tensor& input, std::size_t dim0, std::size_t dim1);
    Tensor embedding_kernel_cpu(const Tensor* weight_tensor, const Tensor& input_tensors);
    Tensor repeat_kv_kernel_cpu(const Tensor& x, std::size_t n_rep);
    Tensor softmax_kernel_cpu(const Tensor& input, std::optional<float> t);
    Tensor silu_kernel_cpu(const Tensor& input);
    Tensor rme_norm_kernel_cpu(const Tensor& input, float eps);
    Tensor vec_or_matrix_argmax_kernel_cpu(const Tensor& input, std::size_t n);
    Tensor rotate_half_kernel_cpu(const Tensor& input);
    Tensor cos_kernel_cpu(const Tensor& input);
    Tensor sin_kernel_cpu(const Tensor& input);
    Tensor concat_by_col_wise_kernel_cpu(const Tensor& x1, const Tensor& x2);
    Tensor concat_by_row_wise_kernel_cpu(const Tensor& x1, const Tensor& x2);
    Tensor causal_mask_kernel_cpu(DataType data_type, std::size_t l, std::size_t s);
    Tensor multinomial_kernel_cpu(const Tensor& x, std::size_t num_samples,
        const std::function<std::size_t(std::size_t)>& row_end_pos);
}
#endif //FG42_KERNEL_H