//
// Created by 19373 on 2025/9/8.
//

#ifndef FG42_KERNEL_H
#define FG42_KERNEL_H
#include "tensor/Tensor.h"

namespace fg42::kernel {
    Tensor add_kernel_cpu(const Tensor& input1, const Tensor& input2);
    Tensor embedding_kernel_cpu(const Tensor* weight_tensor, const Tensor& input_tensors);
}
#endif //FG42_KERNEL_H