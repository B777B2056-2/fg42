//
// Created by 19373 on 2025/9/3.
//

#ifndef FG42_ADD_H
#define FG42_ADD_H
#include "tensor/Tensor.h"

namespace fg42::kernel {
    void add_kernel_cpu(const Tensor& input1, const Tensor& input2, Tensor& output);
}
#endif //FG42_ADD_H