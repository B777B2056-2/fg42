//
// Created by B777B2056-2 on 2025/9/8.
//

#ifndef FG42_KERNEL_CUH
#define FG42_KERNEL_CUH
#include "tensor/Tensor.h"

namespace fg42::kernel {
    Tensor add_kernel_cuda(const Tensor& input1, const Tensor& input2, void* stream);
    Tensor embedding_kernel_cuda(const Tensor* weight_tensor, const Tensor& input_tensors, void* stream);
}
#endif //FG42_KERNEL_CUH