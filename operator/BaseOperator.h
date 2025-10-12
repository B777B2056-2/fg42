//
// Created by B777B2056-2 on 2025/9/3.
//

#ifndef FG42_BASEOPERATOR_H
#define FG42_BASEOPERATOR_H
#include "tensor/Tensor.h"

namespace fg42::kernel {
    class BaseOperator {
    public:
        BaseOperator() = default;
        BaseOperator(const BaseOperator&) = default;
        BaseOperator(BaseOperator&&) = default;
        BaseOperator& operator=(const BaseOperator&) = default;
        BaseOperator& operator=(BaseOperator&&) = default;
        virtual ~BaseOperator() = default;

        virtual Tensor forward(const std::vector<const Tensor*>& input_tensors, void* stream) = 0;

    protected:
        virtual void check(const std::vector<const Tensor*>& input_tensors);
        bool is_same_device_and_data_type(const std::vector<const Tensor*>& input_tensors) const;
        bool is_all_same_shape(const std::vector<const Tensor*>& input_tensors) const;
    };
} // kernel
// fg42

#endif //FG42_BASEOPERATOR_H