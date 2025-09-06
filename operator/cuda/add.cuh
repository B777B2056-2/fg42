//
// Created by 19373 on 2025/9/3.
//

#ifndef FG42_ADD_CUH
#define FG42_ADD_CUH
#include "tensor/Tensor.h"

namespace fg42::kernel {
    void add_kernel_cuda(const Tensor& input1, const Tensor& input2, Tensor& output, void* stream);
}

#endif //FG42_ADD_CUH