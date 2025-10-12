//
// Created by B777B2056-2 on 2025/9/7.
//

#ifndef FG42_EMBEDDINGOPERATOR_H
#define FG42_EMBEDDINGOPERATOR_H
#include "operator/BaseOperator.h"

namespace fg42::kernel {
    class EmbeddingOperator : public BaseOperator {
    private:
        const Tensor* weight_tensor_;

    public:
        explicit EmbeddingOperator(const Tensor& weight_tensor);
        ~EmbeddingOperator() override = default;

        Tensor forward(const std::vector<const Tensor*>& input_tensors, void* stream) override;

    private:
        void check(const std::vector<const Tensor*>& input_tensors) override;
    };
} // kernel
// fg42

#endif //FG42_EMBEDDINGOPERATOR_H