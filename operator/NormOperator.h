//
// Created by B777B2056-2 on 2025/9/10.
//

#ifndef FG42_NORMOPERATOR_H
#define FG42_NORMOPERATOR_H
#include "operator/BaseOperator.h"

namespace fg42::kernel {
    class RMSNormOperator : public BaseOperator {
    public:
        explicit RMSNormOperator(float eps = 1e-6);
        ~RMSNormOperator() override = default;

        Tensor forward(const std::vector<const Tensor*>& input_tensors, void* stream) override;

    private:
        float eps_;

        using BaseOperator::BaseOperator;
        void check(const std::vector<const Tensor*>& input_tensors) override;
    };
} // kernel
// fg42

#endif //FG42_NORMOPERATOR_H