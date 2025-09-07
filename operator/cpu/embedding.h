//
// Created by 19373 on 2025/9/7.
//

#ifndef FG42_EMBEDDING_H
#define FG42_EMBEDDING_H
#include "tensor/Tensor.h"

namespace fg42::kernel {
    Tensor embedding_kernel_cpu(const Tensor* weight_tensor, const Tensor& input_tensors);
}
#endif //FG42_EMBEDDING_H