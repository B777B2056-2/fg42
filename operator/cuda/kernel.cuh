//
// Created by B777B2056-2 on 2025/9/8.
//

#ifndef FG42_KERNEL_CUH
#define FG42_KERNEL_CUH
#include <functional>
#include <optional>
#include "tensor/Tensor.h"

namespace fg42::kernel {
    Tensor add_kernel_cuda(const Tensor& input1, const Tensor& input2, void* stream);
    Tensor negate_kernel_cuda(const Tensor& input, void* stream);
    Tensor vec_outer_kernel_cuda(const Tensor& input1, const Tensor& input2, void* stream);
    Tensor mul_kernel_cuda(const Tensor& input1, const Tensor& input2, void* stream);
    Tensor mul_with_constant_value_kernel_cuda(float value, const Tensor& input2, void* stream);
    Tensor matmul_kernel_cuda(const Tensor& input1, const Tensor& input2, void* stream);
    Tensor transpose_kernel_cuda(const Tensor& input, std::size_t dim0, std::size_t dim1, void* stream);
    Tensor embedding_kernel_cuda(const Tensor* weight_tensor, const Tensor& input_tensors, void* stream);
    Tensor repeat_kv_kernel_cuda(const Tensor& x, std::size_t n_rep, void* stream);
    Tensor softmax_kernel_cuda(const Tensor& input, std::optional<float> t, void* stream);
    Tensor silu_kernel_cuda(const Tensor& input, void* stream);
    Tensor rme_norm_kernel_cuda(const Tensor& input, float eps, void* stream);
    Tensor vec_or_matrix_argmax_kernel_cuda(const Tensor& input, std::size_t n, void* stream);
    Tensor rotate_half_kernel_cuda(const Tensor& input, void* stream);
    Tensor cos_kernel_cuda(const Tensor& input, void* stream);
    Tensor sin_kernel_cuda(const Tensor& input, void* stream);
    Tensor concat_by_col_wise_kernel_cuda(const Tensor& x1, const Tensor& x2, void* stream);
    Tensor concat_by_row_wise_kernel_cuda(const Tensor& x1, const Tensor& x2, void* stream);
    Tensor causal_mask_kernel_cuda(DataType data_type, std::size_t l, std::size_t s, void* stream);
    Tensor multinomial_kernel_cuda(const Tensor& x, std::size_t num_samples,
        const std::function<std::size_t(std::size_t)>& row_end_pos, void* stream);
}
#endif //FG42_KERNEL_CUH