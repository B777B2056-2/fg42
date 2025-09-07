//
// Created by 19373 on 2025/9/3.
//

#ifndef FG42_FACTORY_H
#define FG42_FACTORY_H
#include "tensor/Tensor.h"

namespace fg42::kernel {
    struct VecAddKernelFunc {
        DeviceType device_type;

        explicit VecAddKernelFunc(DeviceType device_type);
        Tensor operator()(const Tensor& input1, const Tensor& input2, void* stream) const;
    };

    struct EmbeddingKernelFunc {
        DeviceType device_type;

        explicit EmbeddingKernelFunc(DeviceType device_type);
        Tensor operator()(const Tensor* weight_tensor, const Tensor& input_tensor, void* stream) const;
    };
}

#endif //FG42_FACTORY_H
