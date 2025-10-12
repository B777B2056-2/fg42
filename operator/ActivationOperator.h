//
// Created by B777B2056-2 on 2025/9/9.
//

#ifndef FG42_ACTIVATIONOPERATOR_H
#define FG42_ACTIVATIONOPERATOR_H
#include "operator/BaseOperator.h"
#include <optional>

namespace fg42::kernel {
    class SoftmaxActivationOperator : public BaseOperator {
    public:
        explicit SoftmaxActivationOperator(std::optional<float> t=std::nullopt);
        ~SoftmaxActivationOperator() override = default;

        Tensor forward(const std::vector<const Tensor*>& input_tensors, void* stream) override;

    private:
        std::optional<float> t_;
        void check(const std::vector<const Tensor*>& input_tensors) override;
    };

    class SiLUActivationOperator : public BaseOperator {
    public:
        explicit SiLUActivationOperator();
        ~SiLUActivationOperator() override = default;
        Tensor forward(const std::vector<const Tensor*>& input_tensors, void* stream) override;

    private:
        void check(const std::vector<const Tensor*>& input_tensors) override;
    };
} // kernel
// fg42

#endif //FG42_ACTIVATIONOPERATOR_H