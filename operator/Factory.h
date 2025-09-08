//
// Created by 19373 on 2025/9/3.
//

#ifndef FG42_FACTORY_H
#define FG42_FACTORY_H
#include "tensor/Tensor.h"

namespace fg42::kernel {
    Tensor VecAddKernelFunc(const Tensor& input1, const Tensor& input2, void* stream);
    Tensor EmbeddingKernelFunc(const Tensor* weight_tensor, const Tensor& input_tensor, void* stream);
}

#endif //FG42_FACTORY_H
